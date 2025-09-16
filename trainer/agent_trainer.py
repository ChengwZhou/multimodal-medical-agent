import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
import math

from models.sensor_gating_agent import SensorGatingAgent


class SensorAgentTrainer:
    """
    序列化传感器Agent训练器
    """

    def __init__(self,
                 agent: SensorGatingAgent,
                 base_model: nn.Module,
                 lambda_energy: float = 0.2,
                 lambda_switch: float = 0.1,
                 lambda_consistency: float = 0.3,
                 fn_penalty_weight: float = 5.0):
        self.agent = agent
        self.base_model = base_model
        self.lambda_energy = lambda_energy
        self.lambda_switch = lambda_switch
        self.lambda_consistency = lambda_consistency
        self.fn_penalty_weight = fn_penalty_weight

        # 优化器
        self.agent_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=1e-3, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.agent_optimizer, mode='min', patience=10, factor=0.5
        )

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.agent.train()
        self.base_model.eval()

        total_losses = {
            'total_loss': 0.0,
            'prediction_loss': 0.0,
            'energy_loss': 0.0,
            'switch_penalty': 0.0,
            'consistency_loss': 0.0,
            'fn_penalty': 0.0
        }

        num_batches = 0

        for batch in train_loader:
            his_feature = None
            predicted_states = torch.zeros([B, M])
            sensor_history = torch.zeros([B, T, M])

            for seg in batch:       # seg: [B, M, N]
                xb, yb = seg
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                mask = yb != -100

                B, M, N = xb.shape


                xb_masked = xb.clone()
                for b in range(B):
                    for m in range(self.agent.num_modalities):
                        if predicted_states[b, m] == 0:
                            xb_masked[b, m, :] = 0

                self.agent_optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits_ref, current_features_ref = self.base_model(xb, his_feature)
                    logits, current_features = self.base_model(xb, his_feature)

                    # Agent预测下一个segment的sensor状态
                    agent_output = self.agent(
                        current_features,
                        sensor_history
                    )
                predicted_states = agent_output['next_sensor_states']
                sensor_history = nn.concate(sensor_history, predicted_states)





            # 使用预测的sensor状态创建下一个segment的输入
            # 这里我们需要模拟下一个segment，实际中应该用真实的下一个segment
            next_segments_masked = current_segments.clone()
            for b in range(B):
                for m in range(self.agent.num_modalities):
                    if predicted_states[b, m] == 0:
                        next_segments_masked[b, m, :] = 0

            # 用masked输入进行预测
            with torch.no_grad():
                logits_original, _ = self.base_model(current_segments)
            logits_masked, _ = self.base_model(next_segments_masked)

            # 计算各种损失
            losses = self._compute_losses(
                logits_original, logits_masked, current_labels, next_labels,
                agent_output
            )

            # 反向传播
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.agent_optimizer.step()

            # 累积损失
            for key, value in losses.items():
                total_losses[key] += value.item()
            num_batches += 1

        # 平均损失
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        # 更新学习率
        self.scheduler.step(avg_losses['total_loss'])

        return avg_losses

    def _compute_losses(self, logits_original, logits_masked, current_labels, next_labels, agent_output):
        """计算复合损失函数"""

        # 1. 预测损失 - 用masked输入预测下一个标签
        prediction_loss = F.cross_entropy(logits_masked, next_labels)

        # 2. 能耗损失 - 鼓励关闭更多sensor
        energy_loss = agent_output['energy_cost'].mean()

        # 3. 开关惩罚 - 减少频繁开关
        switch_penalty = agent_output['state_changes'].mean()

        # 4. 一致性损失 - 确保重要信息不丢失
        with torch.no_grad():
            original_probs = F.softmax(logits_original, dim=1)
        masked_probs = F.softmax(logits_masked, dim=1)
        consistency_loss = F.kl_div(
            F.log_softmax(logits_masked, dim=1),
            original_probs,
            reduction='batchmean'
        )

        # 5. 假阴性惩罚
        with torch.no_grad():
            preds_original = torch.argmax(logits_original, dim=1)
            preds_masked = torch.argmax(logits_masked, dim=1)
            # 当原始预测正确但masked预测错误时的惩罚
            fn_mask = (preds_original == next_labels) & (preds_masked != next_labels)
            fn_penalty = fn_mask.float().mean()

        # 总损失
        total_loss = (prediction_loss +
                      self.lambda_energy * energy_loss +
                      self.lambda_switch * switch_penalty +
                      self.lambda_consistency * consistency_loss +
                      self.fn_penalty_weight * fn_penalty)

        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'energy_loss': energy_loss,
            'switch_penalty': switch_penalty,
            'consistency_loss': consistency_loss,
            'fn_penalty': fn_penalty
        }

    def evaluate(self, val_loader) -> Dict[str, float]:
        """评估模型"""
        self.agent.eval()
        self.base_model.eval()

        total_metrics = {
            'accuracy_original': 0.0,
            'accuracy_masked': 0.0,
            'avg_energy_saved': 0.0,
            'avg_switches': 0.0,
            'fn_rate': 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                current_segments = batch['current_segments']
                current_features = batch['current_features']
                next_labels = batch['next_labels']
                sensor_history = batch['sensor_history']

                # Agent预测
                agent_output = self.agent(current_features, sensor_history)
                predicted_states = agent_output['next_sensor_states']

                # 创建masked输入
                next_segments_masked = current_segments.clone()
                B, M, N = current_segments.shape
                for b in range(B):
                    for m in range(M):
                        if predicted_states[b, m] == 0:
                            next_segments_masked[b, m, :] = 0

                # 预测
                logits_original, _ = self.base_model(current_segments)
                logits_masked, _ = self.base_model(next_segments_masked)

                # 计算指标
                preds_original = torch.argmax(logits_original, dim=1)
                preds_masked = torch.argmax(logits_masked, dim=1)

                acc_original = (preds_original == next_labels).float().mean()
                acc_masked = (preds_masked == next_labels).float().mean()

                # 能耗节省
                sensors_on_ratio = predicted_states.float().mean()
                energy_saved = 1.0 - sensors_on_ratio

                # 开关次数
                avg_switches = agent_output['state_changes'].mean()

                # 假阴性率
                fn_mask = (preds_original == next_labels) & (preds_masked != next_labels)
                fn_rate = fn_mask.float().mean()

                total_metrics['accuracy_original'] += acc_original.item()
                total_metrics['accuracy_masked'] += acc_masked.item()
                total_metrics['avg_energy_saved'] += energy_saved.item()
                total_metrics['avg_switches'] += avg_switches.item()
                total_metrics['fn_rate'] += fn_rate.item()

                num_batches += 1

        # 平均化指标
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return total_metrics
