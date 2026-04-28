# Architecture & Theory: Adaptive Multimodal Sensing with POMDP Gating

> **Document status:** reflects codebase as of the current implementation.

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  Window t-1 [B, M, T]                                                  │
│       │                                                                 │
│       ▼  per-modality                                                   │
│  ┌─────────────────────────────────────────────────────┐               │
│  │  SigmaDelta AdaptiveSensingModule  (×M modalities)  │  [optional]   │
│  │   • Learnable threshold  θ_m = exp(log_θ_m)         │               │
│  │   • Learnable temperature τ_m = exp(log_τ_m)         │               │
│  │   • Soft mask: p_active = σ((activity − θ_m) / τ_m) │               │
│  │   • Hard mask: cummax skip propagation               │               │
│  │   • STE: mask = hard + (soft − soft.detach())        │               │
│  └─────────────────────────────────────────────────────┘               │
│       │ masked delta [B, M, T]  (or raw signal if --no_sigma_delta)     │
│       ▼                                                                 │
│  ┌──────────────────────────────┐                                       │
│  │  ConvTokenizer1D  (×M)       │  → [B, L, D]                         │
│  └──────────────────────────────┘                                       │
│       ▼                                                                 │
│  ┌──────────────────────────────┐                                       │
│  │  Fusion Transformer          │  cat + CLS + PE + TransformerBlock    │
│  └──────────────────────────────┘                                       │
│       │ memory o_t [B, L, D]                                            │
│       ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  BeliefStatePOMDPAgent                                           │  │
│  │                                                                  │  │
│  │  Per-modality encoding:                                          │  │
│  │   o_t [B,L,D] → reshape [B,M,L/M,D] → mean → [B,M,D]           │  │
│  │   modal_encoder (shared Linear+GELU): [B,M,D] → [B,M,modal_per] │  │
│  │   modal_flat = flatten → [B, M·modal_per]                       │  │
│  │                                                                  │  │
│  │  Belief update:                                                  │  │
│  │   b_t = GRUCell(modal_flat, b_{t-1})   [B, H]                   │  │
│  │                                                                  │  │
│  │  Gate history encoding:                                          │  │
│  │   h_t = MLP(flatten(gate_history_{t-history:t}))   [B, H/2]     │  │
│  │                                                                  │  │
│  │  Policy head(s) → logit [B, M]:                                  │  │
│  │   input = [b_t ‖ h_t ‖ modal_flat]                              │  │
│  │   none:       1 head → logit,  uncertainty = None               │  │
│  │   ensemble:   K heads → mean(logits),  σ = std(logits)  [B,M]  │  │
│  │   mc_dropout: 1 head (AlwaysDropout), K passes →               │  │
│  │               mean(logits),  σ = std(logits)  [B,M]            │  │
│  │                                                                  │  │
│  │  Adaptive Gumbel-Sigmoid:                                        │  │
│  │   τ_eff[m] = τ · (1 + scale · σ[m])     [B, M]                 │  │
│  │   p_soft   = σ((logit + Gumbel(0,1)) / τ_eff)   [B, M]         │  │
│  │   p_st     = round(p_soft.detach())                             │  │
│  │             + (p_soft − p_soft.detach())   ← STE                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│       │ gate a_t ∈ {0,1}^M  applied to Window t                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Two-Level Gating

### Level 1 — Temporal (sigma-delta, within a modality)

Skips consecutive patches when `|Δsignal| < θ_m`. Granularity: individual patches (~10–30 ms).
Learnable parameters: `θ_m` (threshold), `τ_m` (threshold temperature), skip duration.

Physiological signals are sparse in the delta domain — ECG is near-zero between beats, EDA drifts slowly, IMU is bursty. Different modalities learn different thresholds naturally.

### Level 2 — Modality (POMDP agent, across sensors)

Turns entire sensor streams on/off per window. Granularity: whole windows (~1–30 s).
Learnable parameters: `modal_encoder` weights, GRU weights, policy MLP weights.

The activity state is partially observable. A single sensor in isolation can be misleading. The belief state `b_t` accumulates a compressed history of which sensors have been informative, allowing context-dependent selection.

---

## 3. Theoretical Grounding

### 3.1 Differentiable Sigma-Delta (STE)

Threshold θ and temperature τ are learnable end-to-end via the Straight-Through Estimator:

```
p^m_l = σ( (A^m_l − θ_m) / τ_m )
mask^m_l = H(A^m_l − θ_m) + (p^m_l − stop_grad(p^m_l))

∂L/∂θ_m = Σ_{l,b} (∂L/∂mask) · (−1/τ_m) · σ'((A−θ)/τ)
```

The gradient is non-zero when the loss gradient w.r.t. the masked feature is non-zero and the sigmoid is not saturated — both hold for physiological data with variable activity.

### 3.2 Agent Training via STE (CE gradient through gate)

The policy is trained entirely via CE gradients flowing through the STE gate:

```
window_gated = window * p_st
            = window * (p_hard + p_soft − stop_grad(p_soft))

∂L_CE/∂logit_k = (∂L_CE/∂window_gated) · window_k · σ'(logit_k / τ_eff[k])
```

Gates that allow informative inputs receive positive CE gradient reinforcement; gates that block informative inputs are penalised. No REINFORCE, no value function — the CE gradient is the sole training signal for the agent.

### 3.3 POMDP Framing and GRU Belief Approximation

**Formal POMDP:**
- State S: latent activity + signal quality (unobserved)
- Observation O = o_t: transformer memory [B, L, D]
- Action A = a_t ∈ {0,1}^M
- Reward r_t = −CE(t) − λ_e · Σ_m a_{t,m}
- Belief b_t: compressed history sufficient for optimal action selection

The exact Bayesian belief update
```
P(s_t | o_{1:t}, a_{1:t-1}) ∝ P(o_t | s_t) · Σ_{s_{t-1}} P(s_t | s_{t-1}, a_{t-1}) · P(s_{t-1} | ...)
```
is intractable. We substitute:
```
b_t = GRUCell(modal_flat_t, b_{t-1})
```

The GRU approximation has three acknowledged sources of error: (1) finite capacity `b_t ∈ R^H` cannot represent arbitrary distributions, (2) Markov compression — only `(b_{t-1}, modal_flat_t)` is seen, not full history, (3) deterministic update — GRUCell is a fixed function, not a stochastic sampler. These are standard limitations in the recurrent POMDP literature (Igl et al. 2018; Zintgraf et al. 2020). Section 5.4 describes a belief-state probe experiment to empirically quantify belief quality.

### 3.4 Uncertainty-Adaptive Exploration

The agent estimates epistemic uncertainty σ[b, m] per sample and per sensor, then uses it to adaptively widen the Gumbel-Sigmoid temperature for uncertain sensors:

```
τ_eff[b, m] = τ · (1 + uncertainty_scale · σ[b, m])
```

**Sensors the agent is uncertain about** receive a larger effective temperature → wider Gumbel noise → more exploration. **Sensors whose relevance is already established** converge to τ ≈ τ_base → sharper, more decisive gates.

Two uncertainty estimation modes are supported (selected via `--agent_uncertainty`):

**Ensemble** (`uncertainty_type='ensemble'`):

K independent policy heads are built with different random initialisations. Each head maps `[b_t ‖ h_t ‖ modal_flat]` to logits independently. Their disagreement is the epistemic uncertainty:

```
logits_k ∈ R^{K×B×M}   (K forward passes, one per head)
gate_logits = mean_k(logits_k)       [B, M]
σ           = std_k(logits_k)        [B, M]  ← uncertainty
```

All K heads are trained jointly via the same CE gradient through the mean logit. An optional diversity regularisation loss prevents all heads from collapsing to identical weights:

```
L_diversity = −mean_{i<j} ||logits_i − logits_j||²_F
```

Adding `λ_div · L_diversity` to the total loss maximises pairwise disagreement, keeping the uncertainty estimates meaningful throughout training.

**MC Dropout** (`uncertainty_type='mc_dropout'`):

A single policy head uses `_AlwaysDropout` — a dropout layer that remains active even during eval mode. K stochastic forward passes through the same head produce K different logit samples:

```
logits_k ∈ R^{K×B×M}   (K passes with AlwaysDropout active)
gate_logits = mean_k(logits_k)       [B, M]
σ           = std_k(logits_k)        [B, M]  ← MC uncertainty
```

MC Dropout uncertainty is available at both train time and eval time without any special mode switching.

### 3.5 Regret Bound Scope

The Thompson Sampling regret bound E[R_T] = O(√(M·T·log T)) (Russo & Van Roy 2016) applies when the Transformer encoder is pre-trained and **frozen** — the agent then faces a proper contextual bandit with a fixed reward function, and the bound holds for the agent training phase. In the joint end-to-end training setting, the empirical sparsity-accuracy Pareto curve (Section 5.2) is the appropriate experimental evidence.

---

## 4. Training Objective

```
L_total = λ_ce · CE(logits, y)
        + λ_sparse · penalty(p_soft)               ← agent sparsity
        + [optional] λ_pred · MSE(MLP(b_t), o_t)   ← predictive auxiliary
        + [optional] λ_div  · L_diversity            ← ensemble diversity
```

**Sparsity penalty** (two variants):
- **L1:** `λ_sparse · mean(p_soft)` — pushes active ratio down
- **Squared-error:** `λ_sparse · (mean(p_soft) − target)²` — pushes ratio toward a fixed target from both directions (`--target_agent_ratio`)

**Predictive auxiliary loss** (`--predictive_weight > 0`): A separate MLP predicts the next window's model memory from the current belief. Gradient flows belief → GRU, training the recurrent state to be predictive of future observations independently of the CE task loss.

**Ensemble diversity loss** (`--agent_diversity_weight > 0`, only with `--agent_uncertainty ensemble`): Maximises pairwise disagreement between K policy heads, preventing uncertainty estimates from degenerating to zero as heads converge.

---

## 5. Required Experiments (NeurIPS/ICML standard)

### 5.1 Ablation matrix

| Configuration | Purpose |
|---|---|
| Full model | Baseline |
| `--no_sigma_delta` | Temporal gating contribution |
| `--only_predictive` | Modality gating contribution |
| `--agent_uncertainty none` vs `ensemble` vs `mc_dropout` | Adaptive exploration contribution |
| Vary `--agent_ensemble_k` (3, 5, 10) | Sensitivity to number of heads / MC passes |
| Vary `--agent_uncertainty_scale` (0, 0.5, 1.0, 2.0) | Sensitivity to exploration widening factor |

### 5.2 Sparsity–accuracy Pareto curve

Vary `--agent_sparsity_weight` and `--target_agent_ratio` across {0.2, 0.4, 0.6, 0.8}. Plot accuracy vs. active_ratio. Compare uncertainty modes (none / ensemble / mc_dropout) on the same plot.

### 5.3 Skip-pattern analysis

Per-modality `active_ratio` × activity-label heatmap. Expected: ECG high activity during exercise, near-zero at rest; EDA low variability throughout; IMU high during motion labels.

### 5.4 Belief-state probe

Train a small linear classifier on top of frozen `b_t` to predict the current activity label. Track probe accuracy vs. training epoch. A high probe accuracy shows the GRU belief state encodes task-relevant information, providing empirical evidence of belief quality.

### 5.5 Uncertainty calibration

Plot per-sensor σ (from ensemble or MC Dropout) vs. gate decision entropy. Well-calibrated uncertainty should correlate: sensors with high σ should show high gate entropy (the agent genuinely wavers), sensors with low σ should have near-deterministic gates.

### 5.6 Gate decision interpretability

Visualise `p_hard` decisions per modality vs. activity label. Expected: EEG gates off during non-seizure windows in SeizeIT2; wrist sensors gate off during stationary activities.

---

## 6. File Map

```
models/
  pomdp_sensor_agent.py       — BeliefStatePOMDPAgent
                                  • modal_flat GRU input (per-modality)
                                  • uncertainty_type: none | ensemble | mc_dropout
                                  • adaptive τ via per-sensor epistemic std
                                  • ensemble_diversity_loss()
                                  • _AlwaysDropout for MC Dropout
  pomdp_device_agent.py       — Thin alias: BeliefStatePOMDPDeviceAgent = BeliefStatePOMDPAgent
  sigma_former_sensor.py      — Sigma-delta + sensor-wise transformer
  sigma_former_device.py      — Sigma-delta + device-wise transformer
  former_sensor.py            — No sigma-delta, sensor-wise transformer
  former_device.py            — No sigma-delta, device-wise transformer
  agent_sensor_masking.py     — SensorGatingAgent (MLP, used by agent_trainer.py)
  agent_device_masking.py     — DeviceGatingAgent (MLP, used by agent_trainer.py)

trainer/
  sigma_delta_joint_trainer.py — Joint trainer: STE, DDP, fixed τ, predictive aux,
                                  ensemble diversity loss, uncertainty logging
  agent_trainer.py             — MLP agent trainer (REINFORCE optional, DDP)
```

### 4-way model/agent selection (`sigma_delta_joint_trainer.py`)

| `--no_sigma_delta` | `--use_device_wise_model` | Model backbone | Agent |
|---|---|---|---|
| False | False | `sigma_former_sensor` + DeltaDataset | `BeliefStatePOMDPAgent` (M=num_sensors) |
| True  | False | `former_sensor` | `BeliefStatePOMDPAgent` (M=num_sensors) |
| False | True  | `sigma_former_device` + DeltaDataset | `BeliefStatePOMDPDeviceAgent` (M=num_channels) |
| True  | True  | `former_device` | `BeliefStatePOMDPDeviceAgent` (M=num_channels) |

> **Device-wise gating:** The agent gates at **channel level** (M = total input channels, e.g. 12 for mHealth), not device level (M = 2). Channel-level gating is finer-grained and more stable.

---

## 7. Key CLI Arguments (`sigma_delta_joint_trainer.py`)

```bash
# Single GPU — ensemble uncertainty
python trainer/sigma_delta_joint_trainer.py \
    --dataset mhealth --root /path/to/data \
    --model_lr 3e-4 --agent_lr 3e-4 \
    --bptt_steps 8 --num_epochs 100 --batch_size 16 \
    --agent_hidden 256 --agent_modal_per 64 --agent_tau 1.0 \
    --agent_uncertainty ensemble --agent_ensemble_k 5 \
    --agent_uncertainty_scale 1.0 --agent_diversity_weight 0.01 \
    --agent_sparsity_weight 0.05 --target_agent_ratio 0.6 \
    --save_dir ./checkpoints_joint

# Multi-GPU DDP (4 cards)
torchrun --nproc_per_node=4 trainer/sigma_delta_joint_trainer.py \
    --dataset mhealth --root /path/to/data \
    --batch_size 4   # effective global batch = 16
```

| Arg | Default | Effect |
|---|---|---|
| `--agent_tau` | 1.0 | Base Gumbel temperature (fixed throughout training) |
| `--agent_modal_per` | 64 | Per-modality embedding dim fed to GRU and policy head |
| `--agent_uncertainty` | none | Uncertainty mode: `none` / `ensemble` / `mc_dropout` |
| `--agent_ensemble_k` | 5 | K heads (ensemble) or K MC passes (mc_dropout) |
| `--agent_uncertainty_scale` | 1.0 | Adaptive τ widening factor per unit of σ |
| `--agent_diversity_weight` | 0.0 | Ensemble diversity loss weight (0 = disabled) |
| `--agent_sparsity_weight` | 0.0 | Sparsity penalty weight |
| `--target_agent_ratio` | None | Squared-error sparsity target; None → L1 penalty |
| `--predictive_weight` | 0.0 | Predictive auxiliary loss weight (0 = disabled) |
| `--only_predictive` | False | Agent trains but gate not applied to input |
| `--bptt_steps` | 8 | BPTT chunk length |

---

## 8. Implementation Notes

### Gradient flow

```python
# CE gradient path through gate → agent:
#   CE loss → logits → window_gated = window * p_st
#           → p_soft (via STE) → gate_logits
#           → policy_head → [b_t, h_t, modal_flat]
#           → GRU (b_t) and modal_encoder (modal_flat)
#
# All agent parameters receive CE gradient in a single backward pass.
for name, p in agent.named_parameters():
    assert p.grad is not None, f"No grad: {name}"
```

### Adaptive temperature mechanics

```python
# τ_base is fixed (no annealing) — gradients ∂p_soft/∂logit = σ'(·)/τ stay O(1)
# τ_eff widens per-sensor when uncertainty is non-zero:
tau_eff = tau_base * (1.0 + uncertainty_scale * sigma)   # [B, M]
p_soft  = sigmoid((logit + gumbel) / tau_eff)
```

For `uncertainty_type='none'`, `tau_eff = tau_base` (scalar), behaviour is identical to the single-temperature case.

### Ensemble diversity loss sign convention

```python
# ensemble_diversity_loss() returns NEGATIVE mean pairwise squared difference.
# Adding to total_loss with a positive weight maximises disagreement:
total_loss += diversity_weight * ensemble_diversity_loss()
# → minimise total_loss → maximise pairwise logit difference ✓
```

### Belief state initialisation

```python
b_0           = zeros(B, hidden_dim)           # cold start
sensor_hist_0 = ones(B, history_length, M)     # all sensors on initially
# policy_head bias initialised to +2.0 → p_soft ≈ 0.88 at cold start
# prevents: zero belief → all gates closed → zero input → zero belief (dead-lock)
```

### DDP notes

- Model, agent, and predictive MLP are each wrapped in `DistributedDataParallel`.
- Raw module attributes are accessed via `self._model` / `self._agent` / `self._pred_mlp` properties (`.module` unwrap).
- `validate()` synchronises scalars via `dist.all_reduce` and predictions via `dist.all_gather_object`.
- `DistributedSampler.set_epoch(epoch)` is called before each training epoch so each rank sees a different shuffle.
- `ensemble_diversity_loss()` operates on `self._agent` (raw module) to access `_last_logits_k` which is set inside the raw module's `forward()`.
