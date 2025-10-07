import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SensorGatingAgent(nn.Module):
    """
    Simplified SensorGatingAgent with learnable thresholds for energy saving
    - Uses input features directly compared against learnable thresholds
    - No gate_logits, p_soft, p_hard, p_st needed
    - Energy savings computed by counting sensors below threshold
    """

    def __init__(self,
                 num_modalities: int,
                 feature_dim: int,  # D (single patch dimension)
                 hidden_dim: int = 128,
                 history_length: int = 5,
                 thresh_init: float = 0.5,  # 初始阈值
                 thresh_range: Tuple[float, float] = (0.1, 0.9),  # 阈值约束范围
                 thresh_temp: float = 1.0,  # 阈值温度参数（用于稳定训练）
                 input_aggregation: str = 'mean',  # 'mean', 'max', 'sum' - 如何聚合每个modality的patches
                 ):
        super().__init__()

        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.thresh_range = thresh_range
        self.thresh_temp = thresh_temp
        self.input_aggregation = input_aggregation
        self._eps = 1e-8

        # threshold_heads: 为每个modality生成可学习阈值
        # 每个head接收该modality的聚合特征，输出一个阈值参数
        self.threshold_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 4, hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, 1),  # 输出单个阈值参数
            ) for _ in range(num_modalities)
        ])

        # 初始化阈值参数偏置，使初始阈值接近thresh_init
        thresh_init_logit = self._inverse_constrain_threshold(torch.tensor(thresh_init))
        for head in self.threshold_heads:
            nn.init.constant_(head[-1].bias, thresh_init_logit.item())

    def _inverse_constrain_threshold(self, threshold: torch.Tensor) -> torch.Tensor:
        """将约束范围内的阈值转换为logit参数"""
        lo, hi = self.thresh_range
        normalized = (threshold - lo) / (hi - lo)
        normalized = torch.clamp(normalized, self._eps, 1 - self._eps)
        return torch.log(normalized / (1 - normalized))

    def _constrain_threshold(self, logit_param: torch.Tensor) -> torch.Tensor:
        """将logit参数约束到指定阈值范围"""
        s = torch.sigmoid(logit_param / self.thresh_temp)
        lo, hi = self.thresh_range
        return lo + (hi - lo) * s

    def extract_modality_thresholds(self, represent_features: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从 represent_features 提取每个modality的特征，计算activation levels和阈值
        Args:
            represent_features: [B, seq_len, D]
        Returns:
            mod_features: [B, M, D] - 每个modality的聚合特征
            thresholds: [B, M] - 每个modality的可学习阈值
        """
        B, seq_len, D = represent_features.shape
        M = self.num_modalities
        assert D == self.feature_dim, f"feature_dim mismatch: got {D} vs {self.feature_dim}"
        patches_per_modality = seq_len // M
        assert patches_per_modality * M == seq_len, "seq_len must be divisible by num_modalities"

        # 重塑并聚合每个modality的patches
        reshaped = represent_features.view(B, M, patches_per_modality, D)  # [B, M, patches_per_mod, D]

        if self.input_aggregation == 'mean':
            mod_features = reshaped.mean(dim=2)  # [B, M, D]
        elif self.input_aggregation == 'max':
            mod_features = reshaped.max(dim=2)[0]  # [B, M, D]
        elif self.input_aggregation == 'sum':
            mod_features = reshaped.sum(dim=2)  # [B, M, D]
        else:
            raise ValueError(f"Unknown aggregation method: {self.input_aggregation}")

        # 为每个modality计算阈值
        thresholds = []

        for m in range(M):
            mod_feat = mod_features[:, m]  # [B, D]

            # 计算该modality的自适应阈值
            thresh_logit = self.threshold_heads[m](mod_feat).squeeze(-1)  # [B]
            thresh = self._constrain_threshold(thresh_logit)  # [B]
            thresholds.append(thresh)

        thresholds = torch.stack(thresholds, dim=1)  # [B, M]

        return mod_features, thresholds

    def compute_energy_savings(self, activation_levels: torch.Tensor, thresholds: torch.Tensor) -> Dict[
        str, torch.Tensor]:
        """
        基于activation levels与阈值的比较计算能耗节省
        Args:
            activation_levels: [B, M] 每个modality的输入activation level
            thresholds: [B, M] 每个modality的阈值
        Returns:
            dict with energy metrics
        """
        # Hard comparison: activation < threshold => sensor is OFF (saving energy)
        sensors_off_hard = (activation_levels < thresholds).float()  # [B, M]

        # Soft comparison: 用于梯度传播
        # 使用sigmoid来平滑"小于"的操作: sigmoid(k * (thresh - activation))
        # 当activation << threshold时，输出接近1 (sensor off)
        # 当activation >> threshold时，输出接近0 (sensor on)
        temp_scale = 10.0  # 控制平滑程度
        diff = (thresholds - activation_levels) * temp_scale
        sensors_off_soft = torch.sigmoid(diff)  # [B, M]

        # Energy savings计算
        energy_saved_hard = sensors_off_hard.mean(dim=1)  # [B] - 实际关闭sensor的比例
        energy_saved_soft = sensors_off_soft.mean(dim=1)  # [B] - 可梯度传播的版本

        # 总的关闭sensor数量
        sensors_off_count = sensors_off_hard.sum(dim=1)  # [B]

        # 阈值正则化：防止阈值过于极端
        thresh_reg = torch.mean((thresholds - 0.5) ** 2)

        # Activation正则化：防止activation过于极端
        activation_reg = torch.mean((activation_levels - 0.5) ** 2)

        return {
            'energy_saved_soft': energy_saved_soft,  # [B] 可梯度传播的能耗节省
            'energy_saved_hard': energy_saved_hard,  # [B] 实际能耗节省（监控用）
            'sensors_off_count': sensors_off_count,  # [B] 关闭的sensor数量
            'sensors_off_soft': sensors_off_soft,  # [B, M] 每个sensor关闭的soft程度
            'sensors_off_hard': sensors_off_hard,  # [B, M] 每个sensor是否关闭
            'threshold_reg': thresh_reg,  # [] 阈值正则化项
            'activation_reg': activation_reg,  # [] activation正则化项
            'avg_threshold': thresholds.mean(),  # [] 平均阈值（监控用）
            'avg_activation': activation_levels.mean(),  # [] 平均activation（监控用）
        }

    def forward(self,
                represent_features: torch.Tensor,
                sensor_history: torch.Tensor,  # 保留用于接口兼容，但不使用
                ) -> Dict[str, torch.Tensor]:
        """
        简化的前向传播：直接基于输入特征和阈值计算能耗节省
        Args:
            represent_features: [B, seq_len, D]
            sensor_history: [B, T, M] - 保留兼容性但不使用
        Returns:
            dict with:
              - 'thresholds': [B, M] 每个modality的可学习阈值
              - 'energy_saved_soft': [B] 可梯度传播的能耗节省 (主要用于loss)
              - 'mod_features': [B, M, D] 每个modality的聚合特征
              - additional energy and monitoring metrics
        """

        # 提取modality特征、activation levels和阈值
        mod_features, thresholds = self.extract_modality_features_and_thresholds(represent_features)

        # 计算基于阈值比较的能耗节省
        # energy_metrics = self.compute_energy_savings(activation_levels, thresholds)

        # 组合所有输出
        output = {
            'mod_features': mod_features,  # [B, M, D] modality聚合特征
            'thresholds': thresholds,  # [B, M] learnable thresholds
            # 'energy_saved_soft': energy_metrics['energy_saved_soft'],  # [B] 主要用于loss的能耗节省
        }

        # 添加其他energy相关metrics
        # output.update({k: v for k, v in energy_metrics.items() if k != 'energy_saved_soft'})

        return output