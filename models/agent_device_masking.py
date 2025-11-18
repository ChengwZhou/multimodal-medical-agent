import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from utils.modality_config import ModalityConfig

class DeviceGatingAgent(nn.Module):
    """
    SensorGatingAgent with Gumbel-Sigmoid + ST estimator (soft p, hard forward)
    - represent_features: [B, seq_len, feature_dim] from MultimodalActivityTransformer before head
    - seq_len = num_device * patches_per_sensor
    - sensor_history: [B, T, num_device] where num_device is the number of unique device_idx
    - Returns soft gating p_soft and ST-forwarded next_sensor_states (p_st)
    - compute_aux_loss accepts full_task_logits & masked_task_logits to form a soft FN proxy
    """
    def __init__(self,
                 num_modalities: int,
                 modalities: List[ModalityConfig],
                 feature_dim: int,            # model_dim from MultimodalActivityTransformer
                 hidden_dim: int = 512,
                 history_length: int = 5,
                 min_on_segments: int = 2,
                 min_off_segments: int = 1,
                 cooldown_segments: int = 1,
                 gumbel_tau: float = 1.0,
                 thresh_init: float = 0.0,
                 thresh_range: Tuple[float, float] = (0.0, 1.0)
                 ):
        super().__init__()

        # Number of sensors is the number of unique device_idx values
        self.num_modalities = num_modalities
        self.modalities = modalities
        self.num_device = len(set(m.device_idx for m in modalities))
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.min_on_segments = min_on_segments
        self.min_off_segments = min_off_segments
        self.cooldown_segments = cooldown_segments
        self.thresh_range = thresh_range
        self.gumbel_tau = gumbel_tau
        self._eps = 1e-8

        # gate_feature_extractor: input449input = num_device * feature_dim -> outputs num_device gate features (logits)
        self.gate_feature_extractor = nn.Sequential(
            nn.Linear(self.num_device * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_modalities)
        )

    def _thresh_sigmoid(self, logit_param: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logit_param)
        lo, hi = self.thresh_range
        return lo + (hi - lo) * s

    def _gumbel_sigmoid(self, logits: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
        """
        Gumbel-Sigmoid for Bernoulli-like sampling (continuous in (0,1))
        logits: [B, num_device]
        returns: p_soft in (0,1)
        """
        if tau is None:
            tau = self.gumbel_tau
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + self._eps) + self._eps)
        y = torch.sigmoid((logits + g) / (tau + self._eps))
        return y

    def extract_modality_features(self, represent_features: torch.Tensor) -> torch.Tensor:
        """
        Extract aggregated features for each sensor from represent_features.
        represent_features: [B, seq_len, feature_dim] where seq_len = num_device * patches_per_sensor
        returns gate_logits: [B, num_device]
        """
        B, seq_len, D = represent_features.shape
        patches_per_sensor = seq_len // self.num_device
        assert D == self.feature_dim, f"feature_dim mismatch: got {D} vs {self.feature_dim}"

        # Reshape to [B, num_device, patches_per_sensor, feature_dim] and mean over patches
        sensor_feats = represent_features.view(B, self.num_device, patches_per_sensor, D).mean(dim=2)  # [B, num_device, D]
        all_flat = sensor_feats.view(B, self.num_device * D)  # [B, num_device * D]
        gate_logits = self.gate_feature_extractor(all_flat)  # [B, num_modalities]
        return gate_logits

    def forward(self,
                represent_features: torch.Tensor,
                sensor_history: torch.Tensor,
                use_straight_through: bool = True
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            represent_features: [B, seq_len, feature_dim] from transformer before head
            sensor_history: [B, T, num_device]
        Returns:
            dict with:
              - 'p_soft': [B, num_device] continuous probabilities (train)
              - 'p_hard': [B, num_device] hard rounded
              - 'p_st': [B, num_device] ST forward (hard in forward, grad flows from p_soft)
              - 'current_gate_logits': [B, num_device]
              - 'energy_cost': [B] (approx)
              - 'state_changes': [B]
        """
        current_sensor_states = sensor_history[:, -1, :]  # [B, num_device*T, D]
        gate_logits = self.extract_modality_features(represent_features)  # [B, num_modalities]

        # Gumbel-Sigmoid gating
        if self.training:
            p_soft = self._gumbel_sigmoid(gate_logits)  # [B, num_modalities]
        else:
            p_soft = torch.sigmoid(gate_logits)

        p_hard = torch.round(p_soft)
        if use_straight_through:
            p_st = (p_hard - p_soft).detach() + p_soft
        else:
            p_st = p_soft

        energy_cost = p_soft.mean(dim=1)  # [B]
        forward_next_state = torch.round(p_st)
        state_changes = (forward_next_state != current_sensor_states).float().sum(dim=1)  # [B]

        return {
            'p_soft': p_soft,
            'p_hard': p_hard,
            'p_st': p_st,
            'current_gate_logits': gate_logits,
            'energy_cost': energy_cost,
            'state_changes': state_changes,
        }

    def _compute_state_duration(self, sensor_history: torch.Tensor) -> torch.Tensor:
        """
        Calculate the number of segments for which each sensor's current state persists
        sensor_history: [B, T*num_modalities, D]
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
        Trigger Cost Proxy: Number of state changes * (Number of hourly segments / 1.0)
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
        Calculate the jointed loss:
        - p_soft: [B, num_device] continuous gating probabilities
        - masked_task_logits: [B, C] logits produced by backbone with gating applied
        - full_task_logits: [B, C] logits produced by backbone with all sensors on
        - labels: [B] class labels for task
        Returns:
            total_loss, diagnostics
        """
        B = p_soft.shape[0]
        device = p_soft.device

        masked_loss_per_sample = F.cross_entropy(masked_task_logits, labels, reduction='none')  # [B]
        full_loss_per_sample = F.cross_entropy(full_task_logits, labels, reduction='none')      # [B]

        masked_task_loss = masked_loss_per_sample.mean()
        full_task_loss = full_loss_per_sample.mean()

        loss_increase = (masked_loss_per_sample - full_loss_per_sample).clamp(min=0.0)  # [B]
        soft_fn = loss_increase.mean()

        energy_per_sample = p_soft.mean(dim=1) * float(energy_coeff)  # [B]
        energy_mean = energy_per_sample.mean()

        trigger_pen = self.compute_trigger_penalty(state_changes, segments_per_hour=segments_per_hour).mean()

        fn_norm = soft_fn / (full_task_loss + 1e-8)
        energy_norm = energy_mean

        score = lambda_fn * fn_norm + (1.0 - lambda_fn) * energy_norm + gamma_trigger * trigger_pen
        total_loss = masked_task_loss + score

        if fn_cap is not None:
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