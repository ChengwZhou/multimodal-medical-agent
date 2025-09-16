import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SensorGatingAgent(nn.Module):
    """
    SensorGatingAgent with Gumbel-Sigmoid + ST estimator (soft p, hard forward)
    - represent_features: [B, seq_len, D] (seq_len = patches_per_modality * M)
    - sensor_history: [B, T, M]
    - Returns soft gating p_soft and ST-forwarded next_sensor_states (p_st)
    - compute_aux_loss accepts full_task_logits & masked_task_logits to form a soft FN proxy
    """

    def __init__(self,
                 num_modalities: int,
                 feature_dim: int,            # D (single patch dimension)
                 hidden_dim: int = 128,
                 history_length: int = 5,
                 min_on_segments: int = 2,
                 min_off_segments: int = 1,
                 cooldown_segments: int = 1,
                 gumbel_tau: float = 1.0,
                 thresh_init: float = 0.0,
                 thresh_range: Tuple[float, float] = (0.0, 1.0)
                 ):
        super().__init__()

        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.min_on_segments = min_on_segments
        self.min_off_segments = min_off_segments
        self.cooldown_segments = cooldown_segments
        self.thresh_range = thresh_range
        self.gumbel_tau = gumbel_tau
        self._eps = 1e-8

        # gate_feature_extractor: input = M * D -> outputs M gate features (we treat them as logits)
        self.gate_feature_extractor = nn.Sequential(
            nn.Linear(num_modalities * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modalities)
        )

        # # LSTM: input size = 2M (gate_features + sensor_states)
        # self.lstm = nn.LSTM(
        #     input_size=num_modalities * 2,
        #     hidden_size=hidden_dim,
        #     num_layers=2,
        #     batch_first=True,
        #     dropout=0.1
        # )
        #
        # # decision heads (保留)
        # self.decision_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim // 2),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim // 2, 3)
        #     ) for _ in range(num_modalities)
        # ])

    def _thresh_sigmoid(self, logit_param: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logit_param)
        lo, hi = self.thresh_range
        return lo + (hi - lo) * s

    def _gumbel_sigmoid(self, logits: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
        """
        Gumbel-Sigmoid for Bernoulli-like sampling (continuous in (0,1))
        logits: [B, M]
        returns: p_soft in (0,1)
        """
        if tau is None:
            tau = self.gumbel_tau
        # sample gumbel noise
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + self._eps) + self._eps)
        # apply to logits and sigmoid
        y = torch.sigmoid((logits + g) / (tau + self._eps))
        return y

    def extract_modality_features(self, represent_features: torch.Tensor) -> torch.Tensor:
        """
        从 represent_features 提取 per-modality 聚合特征并通过 gate_feature_extractor 得到 gate logits
        returns gate_logits: [B, M]
        """
        B, seq_len, D = represent_features.shape
        M = self.num_modalities
        assert D == self.feature_dim, f"feature_dim mismatch: got {D} vs {self.feature_dim}"
        patches_per_modality = seq_len // M
        assert patches_per_modality * M == seq_len, "seq_len must be divisible by num_modalities"

        mod_feats = represent_features.view(B, M, patches_per_modality, D).mean(dim=2)  # [B, M, D]
        all_flat = mod_feats.view(B, M * D)  # [B, M*D]
        gate_logits = self.gate_feature_extractor(all_flat)  # [B, M]  <-- treat as logits
        return gate_logits

    def forward(self,
                represent_features: torch.Tensor,
                sensor_history: torch.Tensor,
                use_straight_through: bool = True
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            represent_features: [B, seq_len, D]
            sensor_history: [B, T, M]
        Returns:
            dict with:
              - 'p_soft': [B, M] continuous probabilities (train)
              - 'p_st': [B, M] ST forward (hard in forward, grad flows from p_soft)
              - 'current_gate_logits': [B, M]
              - 'energy_cost': [B] (approx)
              - 'state_changes': [B]
        """
        current_sensor_states = sensor_history[:, -1, :]  # [B, M]

        # gate logits from represent features (via extractor)
        gate_logits = self.extract_modality_features(represent_features)  # [B, M]

        # ---------- Gumbel-Sigmoid gating ----------
        if self.training:
            # training: use gumbel noise for exploration (continuous p_soft)
            p_soft = self._gumbel_sigmoid(gate_logits)  # [B, M]
        else:
            # eval: deterministic prob (sigmoid of logits)
            p_soft = torch.sigmoid(gate_logits)

        # Hard decision for forward but gradient via p_soft (ST trick)
        p_hard = torch.round(p_soft)
        if use_straight_through:
            p_st = (p_hard - p_soft).detach() + p_soft  # forward behaves like p_hard, backward via p_soft
        else:
            # optionally, don't use ST: purely soft
            p_st = p_soft

        # energy approx: average p across modalities (per-sample)
        energy_cost = p_soft.mean(dim=1)  # [B]

        # state changes proxy (compare to current_sensor_states using hard decisions)
        # count how many modalities flipped relative to current state
        # use p_st rounded as forward state representation
        forward_next_state = torch.round(p_st)  # will be p_hard
        state_changes = (forward_next_state != current_sensor_states).float().sum(dim=1)  # [B]

        return {
            'p_soft': p_soft,                      # [B, M] continuous probabilities
            'p_hard': p_hard,                      # [B, M] hard rounded
            'p_st': p_st,                          # [B, M] straight-through state (forward hard, backward soft)
            'current_gate_logits': gate_logits,    # [B, M]
            'energy_cost': energy_cost,            # [B]
            'state_changes': state_changes,        # [B]
        }

    def _compute_state_duration(self, sensor_history: torch.Tensor) -> torch.Tensor:
        """
        计算每个模态当前状态持续segment数量（可选，若需要）
        sensor_history: [B, T, M]
        """
        B, T, M = sensor_history.shape
        last = sensor_history[:, -1:, :]
        eq = (sensor_history == last)
        durations = torch.zeros((B, M), device=sensor_history.device, dtype=torch.float32)
        for t in range(T - 1, -1, -1):
            cont = eq[:, t, :]
            durations = durations + cont.float()
            if not cont.any():
                break
        return durations

    def compute_trigger_penalty(self, state_changes: torch.Tensor, segments_per_hour: float = 3600.0) -> torch.Tensor:
        """
        触发成本 proxy: state_changes * (segments_per_hour / 1.0)
        state_changes: [B] or scalar
        """
        return state_changes * (segments_per_hour / 1.0)

    def compute_aux_loss(self,
                         p_soft: torch.Tensor,
                         masked_task_logits: torch.Tensor,
                         full_task_logits: torch.Tensor,
                         labels: torch.Tensor,
                         state_changes: torch.Tensor,
                         lambda_fn: float = 0.5,
                         gamma_trigger: float = 0.01,
                         energy_coeff: float = 1.0,
                         segments_per_hour: float = 3600.0,
                         fn_cap: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算联合损失：
        - p_soft: [B, M] continuous gating probabilities
        - masked_task_logits: [B, C] logits produced by backbone with gating applied (this is what you'd actually use to compute task loss)
        - full_task_logits: [B, C] logits produced by backbone with all sensors on (baseline)
        - labels: [B] class labels for task
        Returns:
            total_loss, diagnostics
        说明：
            - masked_task_loss: 真实被优化的任务 loss (cross entropy on masked logits)
            - soft_FN_proxy: ReLU(masked_loss - full_loss) (continuous) -> 反应由于 gating 导致的性能下降
            - energy ≈ mean(p_soft) (per-sample averaged)
        """
        B = p_soft.shape[0]
        device = p_soft.device

        # task losses (per-sample)
        masked_loss_per_sample = F.cross_entropy(masked_task_logits, labels, reduction='none')  # [B]
        full_loss_per_sample = F.cross_entropy(full_task_logits, labels, reduction='none')      # [B]

        masked_task_loss = masked_loss_per_sample.mean()
        full_task_loss = full_loss_per_sample.mean()

        # soft FN proxy: where masked loss > full loss (i.e., gating hurt performance)
        loss_increase = (masked_loss_per_sample - full_loss_per_sample).clamp(min=0.0)  # [B]
        soft_fn = loss_increase.mean()  # scalar (continuous proxy for "how much gating caused extra error")

        # energy (averaged)
        energy_per_sample = p_soft.mean(dim=1) * float(energy_coeff)  # [B]
        energy_mean = energy_per_sample.mean()  # scalar

        # trigger penalty
        trigger_pen = self.compute_trigger_penalty(state_changes, segments_per_hour=segments_per_hour).mean()

        # normalize soft_fn and energy to combine (simple normalization)
        # use full_task_loss as scale reference to make soft_fn in relative terms
        fn_norm = soft_fn / (full_task_loss + 1e-8)
        energy_norm = energy_mean  # p in [0,1], already normalized

        # score combining FN and Energy (lower better)
        score = lambda_fn * fn_norm + (1.0 - lambda_fn) * energy_norm + gamma_trigger * trigger_pen

        # total loss: ensure task performance prioritized by including masked_task_loss
        total_loss = masked_task_loss + score

        # if user enforces fn_cap (interpreted as allowed avg loss increase or allowed number), strongly penalize exceed
        if fn_cap is not None:
            # treat fn_cap as maximum allowed avg loss increase (soft proxy). If exceeded, add big penalty.
            exceed = (soft_fn - float(fn_cap))
            if exceed > 0:
                total_loss = total_loss + 100.0 * (exceed / (full_task_loss + 1e-8))

        diagnostics = {
            'masked_task_loss': float(masked_task_loss.detach().cpu().item()),
            'full_task_loss': float(full_task_loss.detach().cpu().item()),
            'soft_fn': float(soft_fn.detach().cpu().item()),
            'fn_norm': float(fn_norm.detach().cpu().item()),
            'energy_mean': float(energy_mean.detach().cpu().item()),
            'trigger_penalty': float(trigger_pen.detach().cpu().item()),
            'score': float(score.detach().cpu().item()),
            'total_loss': float(total_loss.detach().cpu().item())
        }

        return total_loss, diagnostics
