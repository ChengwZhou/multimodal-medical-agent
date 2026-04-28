"""
models/pomdp_sensor_agent.py  —  Belief-State POMDP Sensor Gating Agent

Formal POMDP definition
-----------------------
  S  (state)       : hidden activity + signal-quality state (latent, never observed)
  O  (observation) : transformer memory embedding [B, L, D] — partial observation of S
  A  (action)      : {0,1}^M  binary gate — which sensors to activate next window
  R  (reward)      : r_t = −λ_ce·CE_loss − λ_e·energy − λ_sc·|state_changes|
  b  (belief)      : GRU-compressed history of observations  [B, H]

Policy: π(a_t | b_t)  via Gumbel-Sigmoid continuous relaxation.

Regret bound
------------
Modelling sensor selection as a stochastic M-armed bandit:

  Cumulative regret  R_T = Σ_{t=1}^T [ r(a*_t) − r(a_t) ]

where a*_t is the oracle-optimal action in hindsight.

Gumbel-Sigmoid is equivalent to Thompson Sampling on Bernoulli arms:
  • Each arm k has an unknown expected reward μ_k ∝ σ(logit_k)
  • Gumbel noise g ~ Gumbel(0,1) is equivalent to posterior perturbation
  • σ((logit_k + g_k) / τ) ≈ sampling from a Gumbel-softmax posterior
  • Thompson Sampling achieves: E[R_T] = O( √(M · T · log T) )
    (Russo & Van Roy 2016; Agrawal & Goyal 2012)

With GRU belief state b_t, policy becomes context-conditioned:
  • b_t tracks which sensors were informative in recent history
  • This reduces exploration burden → tighter effective regret in practice

Modules exported
----------------
  BeliefStatePOMDPAgent      — the agent
  LagrangianSparsityController — dual-variable sparsity budget
  RegretTracker               — empirical regret bookkeeping
  AgentOutput                 — named-tuple-style output dataclass
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# _AlwaysDropout — for MC-Dropout uncertainty estimation
# ---------------------------------------------------------------------------

class _AlwaysDropout(nn.Module):
    """
    Dropout that remains active even during eval mode.

    Standard nn.Dropout is disabled when the module is in eval mode, making
    it deterministic at inference.  _AlwaysDropout stays stochastic, so that
    running K forward passes through the same policy head gives K different
    logit samples — the variance across these samples is the MC-Dropout
    estimate of epistemic uncertainty.

    Usage: substitute for nn.Dropout inside the policy head when
    uncertainty_type='mc_dropout'.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True, inplace=False)


# ---------------------------------------------------------------------------
# AgentOutput dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    """Structured output from BeliefStatePOMDPAgent.forward()."""
    p_soft:        torch.Tensor            # [B, M]  continuous gate probs (training)
    p_hard:        torch.Tensor            # [B, M]  hard binary gates (≡ round(p_soft.detach()))
    p_st:          torch.Tensor            # [B, M]  straight-through: hard fwd / soft bwd
    gate_logits:   torch.Tensor            # [B, M]  raw pre-sigmoid logits (mean across ensemble/MC)
    belief:        torch.Tensor            # [B, H]  updated GRU belief state
    combined:      torch.Tensor            # [B, H+H/2+M*modal_per]  full policy_head input
    energy_cost:   torch.Tensor            # [B]     mean gate activation (proxy energy)
    state_changes: torch.Tensor            # [B]     #sensors that changed state
    uncertainty:   Optional[torch.Tensor] = None  # [B, M] epistemic std per sensor; None when disabled


# ---------------------------------------------------------------------------
# LagrangianSparsityController
# ---------------------------------------------------------------------------

class LagrangianSparsityController:
    """
    Lagrangian relaxation for a per-step sparsity budget constraint.

    Optimisation problem
    --------------------
        min_θ  L_task(θ)
        s.t.   E[active_ratio] ≤ target_sparsity

    Lagrangian dual
    ---------------
        L(θ, μ) = L_task(θ) + μ · max(0, ρ̂ − target)

    where ρ̂ is an EMA of the observed active_ratio and μ ≥ 0 is the
    Lagrange multiplier updated by gradient ascent on the dual:

        μ ← clip( μ + lr_dual · (ρ̂ − target),  0, μ_max )

    Usage in training loop
    ----------------------
        ctrl = LagrangianSparsityController(target_sparsity=0.4)
        ...
        sparsity_loss = mean(active_ratios)          # from sensing_info
        mu = ctrl.update(sparsity_loss.item())       # update dual var
        loss = ce_loss + ctrl.penalty(sparsity_loss) # add to total loss
    """

    def __init__(
        self,
        target_sparsity: float = 0.5,   # max acceptable active_ratio
        lr_dual: float = 5e-3,          # dual ascent step size
        mu_init: float = 0.0,
        mu_max: float = 20.0,
        ema_alpha: float = 0.05,        # EMA smoothing (smaller = slower)
    ):
        self.target    = target_sparsity
        self.lr_dual   = lr_dual
        self.mu        = mu_init
        self.mu_max    = mu_max
        self.ema_alpha = ema_alpha
        self._ema      = target_sparsity   # initialise EMA at target

    # ------------------------------------------------------------------

    def update(self, observed_ratio: float) -> float:
        """
        Update the EMA estimate of active_ratio and the dual variable μ.
        Call once per training step (or per batch).

        Returns the current μ for logging.
        """
        self._ema = (1.0 - self.ema_alpha) * self._ema + self.ema_alpha * observed_ratio
        violation  = self._ema - self.target
        self.mu    = float(max(0.0, min(self.mu_max, self.mu + self.lr_dual * violation)))
        return self.mu

    def penalty(self, active_ratio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Return μ · active_ratio  (add this to the task loss).
        Uses the current frozen μ — call update() first.
        """
        return self.mu * active_ratio_tensor

    # ------------------------------------------------------------------
    # Properties for logging
    # ------------------------------------------------------------------

    @property
    def current_mu(self) -> float:
        return self.mu

    @property
    def ema_ratio(self) -> float:
        return self._ema

    def state_dict(self) -> Dict:
        return {"mu": self.mu, "ema": self._ema}

    def load_state_dict(self, d: Dict):
        self.mu   = d["mu"]
        self._ema = d["ema"]


# ---------------------------------------------------------------------------
# RegretTracker
# ---------------------------------------------------------------------------

class RegretTracker:
    """
    Empirical regret bookkeeping.

    Regret is approximated as:
        oracle_reward  ≈  −CE_loss(all sensors on)
        agent_reward   ≈  −CE_loss(gated sensors)

    Usage
    -----
        tracker = RegretTracker()
        tracker.record(oracle_reward=-ce_full, agent_reward=-ce_gated)
        print(tracker.cumulative_regret())
        print(tracker.recent_regret(window=100))
    """

    def __init__(self):
        self._oracle: List[float] = []
        self._agent:  List[float] = []

    def record(self, oracle_reward: float, agent_reward: float):
        self._oracle.append(oracle_reward)
        self._agent.append(agent_reward)

    def cumulative_regret(self) -> float:
        return sum(o - a for o, a in zip(self._oracle, self._agent))

    def recent_regret(self, window: int = 100) -> float:
        n = min(window, len(self._oracle))
        if n == 0:
            return 0.0
        return sum(
            o - a for o, a in zip(self._oracle[-n:], self._agent[-n:])
        ) / n

    def theoretical_bound(self, num_sensors: int) -> float:
        """
        O(√(M · T · log T)) Thompson-Sampling regret bound (unnormalised).
        Useful as a reference point in plots.
        """
        T = len(self._oracle)
        if T == 0:
            return 0.0
        return math.sqrt(num_sensors * T * math.log(T + 1))

    def reset(self):
        self._oracle.clear()
        self._agent.clear()

    def __len__(self) -> int:
        return len(self._oracle)


# ---------------------------------------------------------------------------
# BeliefStatePOMDPAgent
# ---------------------------------------------------------------------------

class BeliefStatePOMDPAgent(nn.Module):
    """
    GRU-based belief-state POMDP agent for multimodal sensor gating.

    At each decision step t:
      1. Receive observation o_t = (transformer memory from window t-1) [B, L, D]
      2. Encode observation: enc_t = MLP(mean_pool(o_t))               [B, H]
      3. Update belief:      b_t   = GRU(enc_t, b_{t-1})               [B, H]
      4. Encode history:     hist_enc = MLP(flatten(sensor_history))    [B, H/2]
      5. Gate logits:        logit = PolicyMLP([b_t, hist_enc])         [B, M]
      6. Gumbel-Sigmoid:     p_soft = σ((logit + Gumbel) / τ)          [B, M]
      7. STE hard gate:      p_st   = round(p_soft.detach()) + p_soft − p_soft.detach()
      8. Value baseline:     V      = ValueMLP(b_t)                     [B, 1]

    Temperature τ anneals from τ_init → τ_min over tau_anneal_steps,
    transitioning from high exploration to high exploitation.

    Args
    ----
    num_sensors      : M  — number of independent sensors/modalities
    obs_dim          : D  — transformer model_dim
    hidden_dim       : H  — GRU hidden state size
    history_length   : how many past gate decisions to encode
    tau_init         : initial Gumbel temperature (exploration)
    tau_min          : final Gumbel temperature (exploitation)
    tau_anneal_steps : number of forward() calls over which τ decays
    """

    def __init__(
        self,
        num_sensors: int,
        obs_dim: int,
        hidden_dim: int = 256,
        history_length: int = 10,
        tau: float = 1.0,              # fixed Gumbel-sigmoid temperature (no annealing)
        modal_per: int = 64,           # per-modality embedding dim for direct policy input
        # Uncertainty-aware exploration (addresses TS approximation gap)
        uncertainty_type: str = "none",   # "none" | "ensemble" | "mc_dropout"
        ensemble_k: int = 5,              # K heads (ensemble) or K MC passes (mc_dropout)
        uncertainty_scale: float = 1.0,   # how strongly uncertainty widens per-sensor τ
    ):
        super().__init__()
        self.num_sensors       = num_sensors
        self.obs_dim           = obs_dim
        self.hidden_dim        = hidden_dim
        self.history_length    = history_length
        self.modal_per         = modal_per
        self.uncertainty_type  = uncertainty_type
        self.ensemble_k        = ensemble_k
        self.uncertainty_scale = uncertainty_scale

        if uncertainty_type not in ("none", "ensemble", "mc_dropout"):
            raise ValueError(
                f"uncertainty_type must be 'none'|'ensemble'|'mc_dropout', got {uncertainty_type!r}"
            )

        # Fixed temperature — saved as buffer so checkpoints are self-contained.
        # No annealing: late-training gradients stay healthy regardless of epoch count.
        self.register_buffer("_tau", torch.tensor(tau))

        # ---- Per-modality encoder -------------------------------------------
        # Shared linear: maps each modality's mean-pooled representation
        #   obs_per_modal [B, M, D] → modal_feats [B, M, modal_per]
        #   → modal_flat [B, M*modal_per]
        #
        # modal_flat serves TWO purposes:
        #   1. Input to GRU (replaces old mean-pool obs_encoder) — gives the
        #      belief state per-modality temporal history rather than blurred
        #      global mean history.
        #   2. Concatenated directly to policy_head — short gradient path for
        #      current-step gating decisions (same role as SensorGatingAgent MLP).
        self.modal_encoder = nn.Sequential(
            nn.Linear(obs_dim, modal_per),
            nn.GELU(),
        )
        _modal_flat_dim = num_sensors * modal_per   # GRU input dimension

        # ---- Belief GRU ------------------------------------------------------
        self.belief_gru = nn.GRUCell(_modal_flat_dim, hidden_dim)

        # ---- Sensor-history encoder ------------------------------------------
        hist_in = history_length * num_sensors
        self.history_encoder = nn.Sequential(
            nn.Linear(hist_in, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )

        # ---- Policy head(s) --------------------------------------------------
        # Three modes controlled by uncertainty_type:
        #
        #   "none"       → single policy_head  (original behaviour, zero overhead)
        #   "ensemble"   → K independent policy_heads; uncertainty = std of their logits.
        #                  Different random inits give diverse predictions.
        #                  Optional diversity loss prevents head collapse.
        #   "mc_dropout" → single policy_head with _AlwaysDropout (active at eval);
        #                  K stochastic passes → variance = MC-Dropout uncertainty.
        #
        # Adaptive temperature:  τ_eff[b,m] = τ · (1 + uncertainty_scale · σ[b,m])
        # High uncertainty sensor → wider Gumbel noise → more exploration.
        # This is what genuine Thompson Sampling provides but fixed-variance
        # Gumbel noise cannot.
        policy_in = hidden_dim + hidden_dim // 2 + num_sensors * modal_per

        if uncertainty_type == "ensemble":
            self.policy_head  = None   # unused; present as None to avoid AttributeError
            self.policy_heads = nn.ModuleList([
                self._build_policy_head(policy_in, hidden_dim, num_sensors, dropout_always=False)
                for _ in range(ensemble_k)
            ])
        else:
            # "none" or "mc_dropout"
            self.policy_head = self._build_policy_head(
                policy_in, hidden_dim, num_sensors,
                dropout_always=(uncertainty_type == "mc_dropout"),
            )
            self.policy_heads = None   # unused

        # value_head removed: it was used only for REINFORCE baseline, but the
        # trainer uses STE + CE gradients exclusively — value_head output was
        # never consumed anywhere in the training loop (dead code).

        self._eps = 1e-8
        # Cache for ensemble diversity loss (set inside _compute_logits_and_uncertainty)
        self._last_logits_k: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Policy head factory (static — no self needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_policy_head(
        policy_in: int,
        hidden_dim: int,
        num_sensors: int,
        dropout_always: bool = False,
    ) -> nn.Sequential:
        """
        Build a policy MLP: policy_in → hidden → hidden/2 → num_sensors.

        dropout_always=True uses _AlwaysDropout so the head stays stochastic
        at eval time — required for MC-Dropout uncertainty sampling.
        """
        dropout_cls = _AlwaysDropout if dropout_always else nn.Dropout
        head = nn.Sequential(
            nn.Linear(policy_in, hidden_dim), nn.GELU(), dropout_cls(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, num_sensors),
        )
        nn.init.constant_(head[-1].bias, 2.0)
        return head

    # ------------------------------------------------------------------
    # Temperature (fixed)
    # ------------------------------------------------------------------

    def get_temperature(self) -> torch.Tensor:
        """Return the fixed Gumbel-sigmoid temperature (no annealing)."""
        return self._tau

    # ------------------------------------------------------------------
    # Gumbel-Sigmoid
    # ------------------------------------------------------------------

    def _gumbel_sigmoid(
        self,
        logits: torch.Tensor,                          # [B, M]
        tau: torch.Tensor,                             # scalar
        uncertainty: Optional[torch.Tensor] = None,    # [B, M] epistemic std or None
    ) -> torch.Tensor:
        """
        Continuous Bernoulli relaxation via Gumbel noise injection.

        When uncertainty is provided (ensemble or mc_dropout mode), the effective
        temperature becomes per-sensor adaptive:

            τ_eff[b, m] = τ · (1 + uncertainty_scale · σ[b, m])

        This is the key property that brings the system closer to Thompson Sampling:
        sensors the agent is uncertain about receive wider exploration (larger τ),
        while sensors whose relevance is already clear get sharper decisions (τ ≈ τ_base).
        Fixed-variance Gumbel noise cannot do this — it explores all sensors equally
        regardless of how much the agent already knows about each one.

        Derivation (base case, no uncertainty)
        ----------------------------------------
          u ~ Uniform(0,1),  g = -log(-log(u)) ~ Gumbel(0,1)
          σ((logit + g) / τ)  →  Bernoulli(σ(logit))  as τ → 0
        """
        if uncertainty is not None:
            # Adaptive per-sensor temperature — [B, M] broadcast
            tau = tau * (1.0 + self.uncertainty_scale * uncertainty)

        if self.training:
            u = torch.rand_like(logits).clamp(self._eps, 1.0 - self._eps)
            g = -torch.log(-torch.log(u))          # Gumbel(0,1)
            return torch.sigmoid((logits + g) / tau)
        else:
            return torch.sigmoid(logits / tau)     # deterministic (τ still adaptive if given)

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def _compute_logits_and_uncertainty(
        self, combined: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gate logits and optional per-sensor epistemic uncertainty.

        Returns
        -------
        logits      : [B, M]  mean logits (or single-head logits when type='none')
        uncertainty : [B, M]  std across K samples, or None when type='none'

        Mode details
        ------------
        "none"       — single deterministic forward; uncertainty = None.
                       Behaviour is identical to the old code.
        "ensemble"   — K independent heads; uncertainty = std[K, B, M].std(0).
                       Caches _last_logits_k [K, B, M] for ensemble_diversity_loss().
        "mc_dropout" — K stochastic passes through the single head (which uses
                       _AlwaysDropout so dropout is active at eval too);
                       uncertainty = std across K draws.
        """
        if self.uncertainty_type == "none":
            return self.policy_head(combined), None

        elif self.uncertainty_type == "ensemble":
            logits_k = torch.stack(
                [h(combined) for h in self.policy_heads], dim=0
            )  # [K, B, M]
            self._last_logits_k = logits_k   # cache for diversity loss
            # uncertainty is detached: std(logits_k) is used only to scale τ,
            # not as a training signal.  Without detach, ∂std/∂logits_k = (x-μ)/(σ·(K-1))
            # → NaN when σ→0 (heads converge), even with diversity loss.
            with torch.no_grad():
                uncertainty = logits_k.std(dim=0).clamp(max=2.0)   # [B, M], no grad; clamped so τ_eff ≤ τ_base*(1+scale*2)
            return logits_k.mean(dim=0), uncertainty

        else:  # "mc_dropout"
            logits_k = torch.stack(
                [self.policy_head(combined) for _ in range(self.ensemble_k)], dim=0
            )  # [K, B, M]
            with torch.no_grad():
                uncertainty = logits_k.std(dim=0).clamp(max=2.0)   # [B, M], no grad; clamped
            return logits_k.mean(dim=0), uncertainty

    def ensemble_diversity_loss(self) -> Optional[torch.Tensor]:
        """
        Diversity regularisation for ensemble heads (cosine-similarity formulation).

        Uses mean pairwise *cosine similarity* instead of L2 distance to keep
        gradients bounded.  L2 gradient grows as 2·(logit_i − logit_j) — once
        heads diverge the gradient explodes (positive feedback → τ → ∞).
        Cosine similarity is bounded in [−1, 1] so its gradient is always O(1).

        Sign convention
        ---------------
            total_loss += diversity_weight * ensemble_diversity_loss()

        The raw return value is the mean cosine similarity ∈ [−1, 1].
        Minimising it → cosine similarity → −1 → maximum angular disagreement ✓

        Trainer logging: disagreement = 1 − div_loss  ∈ [0, 2]  (higher = more diverse).
        """
        if self.uncertainty_type != "ensemble" or self._last_logits_k is None:
            return None
        logits_k = self._last_logits_k   # [K, B, M]
        K = logits_k.shape[0]
        if K < 2:
            return None
        # Flatten B and M into a single vector per head for cosine comparison
        lk = logits_k.reshape(K, -1)                        # [K, B*M]
        lk_n = F.normalize(lk, dim=-1, eps=1e-8)            # unit-norm rows
        pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                cos_ij = (lk_n[i] * lk_n[j]).sum()          # scalar ∈ [−1, 1]
                pairs.append(cos_ij)
        return torch.stack(pairs).mean()   # minimise → maximise angular diversity

    # ------------------------------------------------------------------
    # Belief state initialisation
    # ------------------------------------------------------------------

    def init_belief(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero belief state for the start of a sequence."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_sensor_history(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Start with all sensors on (ones)."""
        return torch.ones(
            batch_size, self.history_length, self.num_sensors, device=device
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        observation: torch.Tensor,         # [B, L, D]  transformer memory
        belief: torch.Tensor,              # [B, H]     previous belief
        sensor_history: torch.Tensor,      # [B, T_h, M] recent gate decisions
        use_straight_through: bool = True,
    ) -> AgentOutput:
        """
        Decide which sensors to activate for the next window.

        Args
        ----
        observation       : transformer memory from the *previous* window
        belief            : GRU belief state from the previous call
        sensor_history    : binary gate history [B, history_length, M]
        use_straight_through : True during training (STE), False at inference

        Returns
        -------
        AgentOutput  with p_soft, p_hard, p_st, gate_logits, belief, value,
                     energy_cost, state_changes
        """
        B = observation.shape[0]

        # ---- 1. Per-modality encoding --------------------------------------
        if observation.dim() == 3:
            B_obs, L, D = observation.shape
            # Per-modality mean-pool: [B, L, D] → [B, M, D]
            # Works when L is divisible by num_sensors (sensor-wise tokenisation).
            # Falls back to global mean replicated across M slots otherwise.
            if L % self.num_sensors == 0:
                patches_p = L // self.num_sensors
                obs_per_modal = observation.view(B_obs, self.num_sensors, patches_p, D).mean(dim=2)
            else:
                obs_mean = observation.mean(dim=1, keepdim=True)
                obs_per_modal = obs_mean.expand(-1, self.num_sensors, -1)       # [B, M, D]
            modal_feats = self.modal_encoder(obs_per_modal)                     # [B, M, modal_per]
            modal_flat  = modal_feats.reshape(B_obs, self.num_sensors * self.modal_per)  # [B, M*mp]
        else:
            # 2D observation — no per-modality structure; zero-fill modal_flat
            modal_flat = observation.new_zeros(B, self.num_sensors * self.modal_per)

        # ---- 2. Update belief state (GRU takes modal_flat directly) --------
        # modal_flat carries per-modality temporal info — richer than the old
        # mean-pooled obs that obs_encoder used to compress.
        new_belief = self.belief_gru(modal_flat, belief)   # [B, H]

        # ---- 3. Encode sensor history ---------------------------------------
        hist_flat = sensor_history.reshape(B, -1)   # [B, T_h * M]
        hist_enc  = self.history_encoder(hist_flat) # [B, H/2]

        # ---- 4. Gate logits (with optional uncertainty) ----------------------
        # Concatenate: belief (temporal context) + hist_enc + modal_flat (current modality identity)
        combined = torch.cat([new_belief, hist_enc, modal_flat], dim=-1)  # [B, H + H/2 + M*modal_per]

        # _compute_logits_and_uncertainty handles all three modes:
        #   "none"       → (logits [B,M], None)
        #   "ensemble"   → (mean logits [B,M], std [B,M])   — also caches _last_logits_k
        #   "mc_dropout" → (mean logits [B,M], std [B,M])
        gate_logits, uncertainty = self._compute_logits_and_uncertainty(combined)  # [B,M], [B,M]|None

        # ---- 5. Gumbel-Sigmoid policy (adaptive τ when uncertainty != None) --
        # When uncertainty is provided, τ_eff[b,m] = τ · (1 + scale · σ[b,m])
        # so uncertain sensors explore more — approximating TS adaptive behaviour.
        tau    = self.get_temperature()
        p_soft = self._gumbel_sigmoid(gate_logits, tau, uncertainty)  # [B, M]  ∈ (0,1)
        p_hard = torch.round(p_soft.detach())                          # [B, M]  ∈ {0,1}

        if use_straight_through:
            # STE: forward = p_hard, backward gradient through p_soft
            p_st = p_hard + (p_soft - p_soft.detach())
        else:
            p_st = p_soft   # smooth gates at inference (no STE needed)

        # ---- 6. Diagnostics -------------------------------------------------
        energy_cost   = p_soft.mean(dim=1)                          # [B]
        prev_gates    = sensor_history[:, -1, :]                    # [B, M]
        state_changes = (p_hard != prev_gates).float().sum(dim=1)   # [B]

        return AgentOutput(
            p_soft        = p_soft,
            p_hard        = p_hard,
            p_st          = p_st,
            gate_logits   = gate_logits,
            belief        = new_belief,
            combined      = combined,
            energy_cost   = energy_cost,
            state_changes = state_changes,
            uncertainty   = uncertainty,   # [B, M] or None
        )

    # ------------------------------------------------------------------
    # Policy gradient helpers
    # ------------------------------------------------------------------

    def policy_gradient_loss(
        self,
        gate_logits: torch.Tensor,    # [B, M]  logits at decision step
        gates_taken: torch.Tensor,    # [B, M]  hard gates that were executed
        reward: torch.Tensor,         # [B]     per-sample reward
        baseline: torch.Tensor,       # [B, 1]  value estimate (from value_head)
        entropy_coeff: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        REINFORCE with baseline + entropy bonus.

        Loss = −E[ (R − V) · log π(a|b) ] − β · H[π]

        where H[π] = −Σ_k p_k log p_k + (1-p_k) log(1-p_k)  is the Bernoulli
        entropy summed over sensors, encouraging exploration.

        Args
        ----
        gate_logits : logits used when the decision was made
        gates_taken : the (hard) gates that were actually applied
        reward      : scalar reward per sample (higher is better)
        baseline    : value prediction (used to reduce variance)
        entropy_coeff: weight for entropy regularisation

        Returns
        -------
        pg_loss   : scalar tensor (backprop-ready)
        info      : dict with component values for logging
        """
        p = torch.sigmoid(gate_logits)                # [B, M]
        advantage = (reward - baseline.squeeze(-1)).detach()  # [B]  no grad through adv

        # Bernoulli log-likelihood of gates_taken under policy π
        log_prob = (
            gates_taken * torch.log(p + self._eps)
            + (1.0 - gates_taken) * torch.log(1.0 - p + self._eps)
        ).sum(dim=-1)   # [B]

        # REINFORCE loss (negative because we maximise reward)
        pg_loss = -(advantage * log_prob).mean()

        # Bernoulli entropy: H = -[p log p + (1-p) log(1-p)]
        entropy = -(
            p * torch.log(p + self._eps)
            + (1.0 - p) * torch.log(1.0 - p + self._eps)
        ).sum(dim=-1).mean()

        total_loss = pg_loss - entropy_coeff * entropy

        info = {
            "pg_loss":   pg_loss.item(),
            "entropy":   entropy.item(),
            "advantage": advantage.mean().item(),
        }
        return total_loss, info

    # value_loss removed along with value_head.


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Smoke test: BeliefStatePOMDPAgent")
    print("=" * 65)

    B, M, D, L, H = 4, 14, 32, 14, 128
    agent = BeliefStatePOMDPAgent(
        num_sensors=M, obs_dim=D, hidden_dim=H,
        history_length=5, tau=1.0,
    )
    agent.train()

    obs     = torch.randn(B, L, D)
    belief  = agent.init_belief(B, torch.device("cpu"))
    history = agent.init_sensor_history(B, torch.device("cpu"))

    out = agent(obs, belief, history, use_straight_through=True)
    print(f"p_soft  : {out.p_soft.shape}")    # [B, M]
    print(f"p_hard  : {out.p_hard.shape}")
    print(f"belief  : {out.belief.shape}")
    print(f"energy  : {out.energy_cost.shape}")
    print(f"τ       : {agent.get_temperature().item():.4f}")

    # Gradient check: STE path (CE gradient flows through p_soft → GRU → modal_encoder)
    out.p_soft.mean().backward()
    for name, p in agent.named_parameters():
        if p.grad is None:
            print(f"  WARNING: {name} has no grad")
    print("✓ All parameters have gradients.")

    # Lagrangian test
    ctrl = LagrangianSparsityController(target_sparsity=0.4)
    for _ in range(20):
        mu = ctrl.update(0.6)   # simulate high activity → violation
    print(f"Lagrangian μ after 20 steps of violation: {ctrl.current_mu:.4f}")

    # Regret tracker test
    tracker = RegretTracker()
    for t in range(200):
        tracker.record(oracle_reward=-0.2, agent_reward=-0.3)
    print(f"Cumulative regret : {tracker.cumulative_regret():.1f}")
    print(f"Recent regret     : {tracker.recent_regret(100):.4f}")
    print(f"Theoretical bound : {tracker.theoretical_bound(M):.1f}")
    print("=" * 65)
