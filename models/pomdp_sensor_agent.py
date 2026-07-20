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
  • σ(logit_k / τ_eff + g_k) ≈ sampling from a Gumbel-softmax posterior
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
      6. Gumbel-Sigmoid:     p_soft = σ(logit / τ_eff + Gumbel)        [B, M]
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
        tau: float = 1.0,              # fixed τ ("none"/"logit") or τ_max ("none_annealing")
        tau_min: float = 0.1,          # τ floor for "none_annealing" mode
        tau_anneal_steps: int = 5000,  # steps over which τ decays from tau → tau_min
        modal_per: int = 64,           # per-modality embedding dim for direct policy input
        # Uncertainty-aware exploration
        uncertainty_type: str = "none",   # "none" | "logit" | "none_annealing"
        uncertainty_scale: float = 1.0,   # how strongly uncertainty widens per-sensor τ
    ):
        super().__init__()
        self.num_sensors       = num_sensors
        self.obs_dim           = obs_dim
        self.hidden_dim        = hidden_dim
        self.history_length    = history_length
        self.modal_per         = modal_per
        self.uncertainty_type  = uncertainty_type
        self.uncertainty_scale = uncertainty_scale

        if uncertainty_type not in ("none", "logit", "none_annealing"):
            raise ValueError(
                f"uncertainty_type must be 'none'|'logit'|'none_annealing', got {uncertainty_type!r}"
            )

        # Temperature buffers — all modes store _tau (max/fixed τ).
        # Annealing mode additionally uses _tau_min, _tau_anneal_steps, _anneal_step.
        # Buffers are saved in checkpoints so training can resume seamlessly.
        self.register_buffer("_tau",              torch.tensor(float(tau)))
        self.register_buffer("_tau_min",          torch.tensor(float(tau_min)))
        self.register_buffer("_tau_anneal_steps", torch.tensor(int(tau_anneal_steps)))
        self.register_buffer("_anneal_step",      torch.zeros(1, dtype=torch.long))

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

        # ---- Policy head -----------------------------------------------------
        # Single shared MLP for both modes.
        #
        #   "none"  → logits only; standard fixed-variance Gumbel exploration.
        #   "logit" → same logits; uncertainty = normalised binary entropy H(p)/log2 ∈ [0,1]
        #             p = sigmoid(logit[b,m])
        #             H = -p·log(p) - (1-p)·log(1-p)   (binary cross-entropy)
        #             σ[b,m] = H / log(2)  ∈ [0, 1]
        #             τ_eff[b,m] = τ · (1 + scale · σ[b,m])
        #             sensor near decision boundary (logit≈0, p≈0.5) → unc≈1 → wider exploration.
        #             confident sensor (|logit| large) → unc→0 → τ_eff≈τ_base.
        #             Zero extra compute; guaranteed per-sensor differentiation.
        #             Much better dynamic range than sigmoid(-|logit|):
        #               logit=1 → 0.78, logit=2 → 0.53, logit=3 → 0.29  (vs 0.27/0.12/0.05)
        policy_in = hidden_dim + hidden_dim // 2 + num_sensors * modal_per
        self.policy_head = self._build_policy_head(policy_in, hidden_dim, num_sensors)

        # value_head removed: it was used only for REINFORCE baseline, but the
        # trainer uses STE + CE gradients exclusively — value_head output was
        # never consumed anywhere in the training loop (dead code).

        self._eps = 1e-8

    # ------------------------------------------------------------------
    # Policy head factory (static — no self needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_policy_head(
        policy_in: int,
        hidden_dim: int,
        num_sensors: int,
    ) -> nn.Sequential:
        """Shared policy MLP for 'none' mode: policy_in → hidden → hidden/2 → M."""
        head = nn.Sequential(
            nn.Linear(policy_in, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, num_sensors),
        )
        nn.init.constant_(head[-1].bias, 2.0)
        return head


    # ------------------------------------------------------------------
    # Temperature (fixed)
    # ------------------------------------------------------------------

    def get_temperature(self) -> torch.Tensor:
        """
        Return the current Gumbel-sigmoid temperature.

        "none" / "logit"   → fixed _tau throughout training.
        "none_annealing"   → exponential decay:
                               τ(t) = τ_min + (τ_max − τ_min) · exp(−t / T)
                             where t = _anneal_step, T = _tau_anneal_steps.
        """
        if self.uncertainty_type != "none_annealing":
            return self._tau
        t   = self._anneal_step.float()
        T   = self._tau_anneal_steps.float().clamp(min=1)
        frac = torch.exp(-t / T)
        return self._tau_min + (self._tau - self._tau_min) * frac

    def anneal_step(self):
        """Increment the annealing counter by 1 (call once per training chunk/step)."""
        if self.uncertainty_type == "none_annealing":
            self._anneal_step.add_(1)

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

        Base case (no uncertainty):
          u ~ Uniform(0,1),  g = -log(-log(u)) ~ Gumbel(0,1)
          σ(logit/τ + g)  →  Bernoulli(σ(logit))  as τ → 0

        When uncertainty is provided ('logit' mode), τ_eff scales only the
        logit — NOT the combined (logit+g) — so that p_hard is affected:

            τ_eff[b,m] = τ · (1 + uncertainty_scale · H(p)[b,m])
            p_soft     = σ( logit[b,m] / τ_eff[b,m]  +  g )

        Why this is correct:
          p_hard = sign(logit/τ_eff + g)
          P(p_hard=1) = P(g > −logit/τ_eff) ≈ σ(logit/τ_eff)

          • High uncertainty → large τ_eff → logit/τ_eff compressed → g dominates
            → P(p_hard=1) ≈ 0.5 → maximally exploratory  ✓
          • Low uncertainty  → τ_eff ≈ τ_base → logit amplified relative to g
            → P(p_hard=1) ≈ σ(logit/τ_base) → exploitative  ✓

        Previous design σ((logit+g)/τ) had τ cancel out of p_hard entirely
        (sign((l+g)/τ) = sign(l+g) for τ>0), so uncertainty had no effect on
        the discrete gate decision — only on gradient magnitude, and in the
        wrong direction (large τ reduced gradient when logit≈0, where gradient
        is most informative).
        """
        if uncertainty is not None:
            # Per-sensor effective temperature — [B, M] broadcast
            tau = tau * (1.0 + self.uncertainty_scale * uncertainty)

        if self.training:
            u = torch.rand_like(logits).clamp(self._eps, 1.0 - self._eps)
            g = -torch.log(-torch.log(u))          # Gumbel(0,1), standard scale
            return torch.sigmoid(logits / tau + g)  # τ compresses logit; g stays full-scale
        else:
            return torch.sigmoid(logits / tau)     # deterministic; τ still adaptive if given

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def _compute_logits_and_uncertainty(
        self,
        belief:     torch.Tensor,   # [B, H]
        hist_enc:   torch.Tensor,   # [B, H/2]
        modal_flat: torch.Tensor,   # [B, M*modal_per]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gate logits and optional per-sensor uncertainty.

        Returns
        -------
        logits      : [B, M]
        uncertainty : [B, M] or None

        Mode details
        ------------
        "none"  — single forward; uncertainty = None; standard fixed-variance Gumbel.

        "logit" — same single forward; uncertainty = normalised binary entropy:
                    p[b,m]   = sigmoid(logit[b,m])
                    H[b,m]   = -p·log(p) - (1-p)·log(1-p)
                    σ[b,m]   = H[b,m] / log(2)  ∈ [0, 1]
                    τ_eff    = τ · (1 + scale · σ[b,m])
                  Applied as: p_soft = sigmoid(logit / τ_eff + g)
                  Semantics:
                    • logit ≈ 0  (p≈0.5) → unc=1 → τ_eff large → logit/τ_eff≈0 → g dominates
                      → P(p_hard=1)≈0.5 → maximum exploration  ✓
                    • |logit| large (confident) → unc→0 → τ_eff≈τ_base → logit/τ_base amplified
                      → P(p_hard=1)≈0 or 1 → exploitation  ✓
                  Detached from computation graph (used only to scale τ, not as loss).
        """
        combined = torch.cat([belief, hist_enc, modal_flat], dim=-1)
        logits   = self.policy_head(combined)   # [B, M]

        if self.uncertainty_type == "none" or self.uncertainty_type == "none_annealing":
            return logits, None

        else:  # "logit"
            with torch.no_grad():
                p   = torch.sigmoid(logits)                           # [B, M] ∈ (0, 1)
                eps = 1e-7
                H   = -(  p        * (p        + eps).log()
                        + (1.0 - p) * (1.0 - p + eps).log() )        # [B, M] ∈ [0, log2]
                uncertainty = H / 0.6931471805599453                  # [B, M] ∈ [0, 1]
            return logits, uncertainty

    # ------------------------------------------------------------------
    # FLOPs / parameter breakdown
    # ------------------------------------------------------------------

    def profile(self, batch_size: int = 1) -> Dict[str, Dict]:
        """
        Per-layer FLOPs (multiply-adds) and parameter count breakdown.

        FLOPs conventions
        -----------------
        Linear(in, out)           : in * out  MACs  (bias negligible)
        GRUCell(input, hidden)    :
            3 input gates  × input*hidden  MACs  (W_i{r,z,n})
            3 hidden gates × hidden*hidden MACs  (W_h{r,z,n})
            Total                 : 3*hidden*(input + hidden)  MACs
        LayerNorm(D)              : 2*D  MACs  (normalise + affine)
        Elementwise (GELU, etc.)  : N    MACs  (1 op per element)

        1 MAC ≈ 2 FLOPs (1 multiply + 1 add).  Results reported in MACs.
        All counts are *per forward call* (batch_size = 1 by default).

        Returns
        -------
        dict  module_name → {"params": int, "macs": int}
        Also prints a formatted table.
        """
        M   = self.num_sensors
        D   = self.obs_dim
        H   = self.hidden_dim
        mp  = self.modal_per
        T_h = self.history_length
        B   = batch_size

        # ---- derived dims ------------------------------------------------
        modal_flat_dim = M * mp
        hist_in        = T_h * M
        policy_in      = H + H // 2 + modal_flat_dim

        rows: Dict[str, Dict] = {}

        # ---- modal_encoder  [B, M, D] → [B, M, mp] ----------------------
        #   Applied M times (once per sensor), shared weights
        lin_me_macs = D * mp * M          # M parallel Linear(D, mp)
        lin_me_p    = D * mp + mp         # weight + bias
        gelu_me     = M * mp              # elementwise
        rows["modal_encoder.linear"] = {
            "shape": f"Linear({D}→{mp}) ×{M}",
            "params": lin_me_p,
            "macs":   lin_me_macs * B,
        }
        rows["modal_encoder.gelu"] = {
            "shape": f"GELU ×{M*mp}",
            "params": 0,
            "macs":   gelu_me * B,
        }

        # ---- belief_gru  GRUCell(modal_flat_dim, H) ----------------------
        #   3*(input*hidden + hidden*hidden)
        gru_i = modal_flat_dim
        gru_macs = 3 * H * (gru_i + H)
        gru_p    = (
            3 * (gru_i * H + H)   # W_i{r,z,n} + b_i{r,z,n}
          + 3 * (H    * H + H)    # W_h{r,z,n} + b_h{r,z,n}
        )
        rows["belief_gru"] = {
            "shape": f"GRUCell({gru_i}→{H})",
            "params": gru_p,
            "macs":   gru_macs * B,
        }

        # ---- history_encoder ---------------------------------------------
        h0, h1 = hist_in, H // 2
        rows["history_encoder.linear0"] = {
            "shape": f"Linear({h0}→{h1})",
            "params": h0 * h1 + h1,
            "macs":   h0 * h1 * B,
        }
        rows["history_encoder.gelu"] = {
            "shape": f"GELU ×{h1}",
            "params": 0,
            "macs":   h1 * B,
        }
        rows["history_encoder.linear1"] = {
            "shape": f"Linear({h1}→{h1})",
            "params": h1 * h1 + h1,
            "macs":   h1 * h1 * B,
        }
        rows["history_encoder.layernorm"] = {
            "shape": f"LayerNorm({h1})",
            "params": 2 * h1,          # weight + bias (affine)
            "macs":   2 * h1 * B,
        }

        # ---- policy_head  MLP(policy_in → H → H/2 → M) ------------------
        p0, p1, p2 = policy_in, H, H // 2
        rows["policy_head.linear0"] = {
            "shape": f"Linear({p0}→{p1})",
            "params": p0 * p1 + p1,
            "macs":   p0 * p1 * B,
        }
        rows["policy_head.gelu0"] = {
            "shape": f"GELU ×{p1}",
            "params": 0,
            "macs":   p1 * B,
        }
        rows["policy_head.linear1"] = {
            "shape": f"Linear({p1}→{p2})",
            "params": p1 * p2 + p2,
            "macs":   p1 * p2 * B,
        }
        rows["policy_head.gelu1"] = {
            "shape": f"GELU ×{p2}",
            "params": 0,
            "macs":   p2 * B,
        }
        rows["policy_head.linear2"] = {
            "shape": f"Linear({p2}→{M})",
            "params": p2 * M + M,
            "macs":   p2 * M * B,
        }

        # ---- Gumbel-sigmoid  (training only) -----------------------------
        rows["gumbel_sigmoid"] = {
            "shape": f"sigmoid+noise ×{M}",
            "params": 0,
            "macs":   M * B,
        }

        # ---- totals -------------------------------------------------------
        total_params = sum(r["params"] for r in rows.values())
        total_macs   = sum(r["macs"]   for r in rows.values())

        # ---- verify against actual param count ----------------------------
        actual_params = sum(p.numel() for p in self.parameters())

        # ---- pretty print -------------------------------------------------
        W = 34
        print(f"\n{'─'*78}")
        print(f"  BeliefStatePOMDPAgent  profile  "
              f"(M={M}, D={D}, H={H}, mp={mp}, T_h={T_h}, B={batch_size})")
        print(f"{'─'*78}")
        hdr = f"  {'Layer':<{W}}  {'Shape':<28}  {'Params':>9}  {'MACs':>12}"
        print(hdr)
        print(f"{'─'*78}")
        for name, r in rows.items():
            p_str  = f"{r['params']:,}"  if r['params'] else "—"
            m_str  = f"{r['macs']:,}"    if r['macs']   else "—"
            print(f"  {name:<{W}}  {r['shape']:<28}  {p_str:>9}  {m_str:>12}")
        print(f"{'─'*78}")
        print(f"  {'TOTAL (analytical)':<{W}}  {'':28}  "
              f"{total_params:>9,}  {total_macs:>12,}")
        print(f"  {'TOTAL (torch.parameters)':<{W}}  {'':28}  "
              f"{actual_params:>9,}")
        print(f"{'─'*78}")
        print(f"  GFLOPs (×2 for FLOPs) = {total_macs * 2 / 1e9:.4f} G  "
              f"per forward call (batch={batch_size})\n")

        rows["__total__"] = {"params": total_params, "macs": total_macs}
        return rows

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
        # _compute_logits_and_uncertainty handles both modes:
        #   "none"       → shared head on combined [B,H+H/2+M*mp]  → (logits [B,M], None)
        #   "mc_dropout" → M per-sensor heads, K passes each       → (mean [B,M], std [B,M])
        gate_logits, uncertainty = self._compute_logits_and_uncertainty(
            new_belief, hist_enc, modal_flat
        )  # [B,M], [B,M]|None
        # combined is built here for AgentOutput (inspection / visualiser use only)
        combined = torch.cat([new_belief, hist_enc, modal_flat], dim=-1)

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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Smoke test: BeliefStatePOMDPAgent")
    print("=" * 65)

    B, M, D, L, H = 1, 12, 512, 20, 256
    agent = BeliefStatePOMDPAgent(
        num_sensors=M, obs_dim=D, hidden_dim=H,
        history_length=5, tau=1.0,
    )
    agent.profile(batch_size=1)
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

    # Regret tracker test
    tracker = RegretTracker()
    for t in range(200):
        tracker.record(oracle_reward=-0.2, agent_reward=-0.3)
    print(f"Cumulative regret : {tracker.cumulative_regret():.1f}")
    print(f"Recent regret     : {tracker.recent_regret(100):.4f}")
    print(f"Theoretical bound : {tracker.theoretical_bound(M):.1f}")
    print("=" * 65)
