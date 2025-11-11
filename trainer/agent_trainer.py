# =====================================
# File: agent_sequential_trainer.py
# Sequential trainer with SensorGatingAgent for ScientISSTMOVE dataset
# Supports BPTT, mixed precision, sensor gating, contrastive alignment, and predictive coding losses
# =====================================
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, is_initialized
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
import time
from tqdm import tqdm
from dataclasses import dataclass
import json
from collections import deque, defaultdict

# Import base components
from sequential_trainer import (
    SequentialDataset,
    SequentialBatch,
    collate_sequential_batch,
    setup_ddp,
    cleanup_ddp,
    log_info
)

from models.former_sensor import build_former
from models.former_device import build_former_device
from models.agent_sensor_masking import SensorGatingAgent
from models.agent_device_masking import DeviceGatingAgent
from models.FMformer import build_FMformer, ModalityConfig

from dataset.ScientISST_MOVE_loader import ScientISSTMOVEDataset, filter_labels
from dataset.mHEALTH_loader import MHealthDataset
from dataset.WESAD_loader import MultiModalWESADDataset
from utils.metrics import compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSequentialTrainer:
    """
    Trainer for sequential prediction with SensorGatingAgent
    Jointly trains model and agent using BPTT, with optional contrastive alignment and predictive coding losses
    """

    def __init__(
            self,
            model: nn.Module,
            agent: SensorGatingAgent,
            train_dataset: SequentialDataset,
            val_dataset: Optional[SequentialDataset] = None,
            batch_size: int = 8,
            learning_rate: float = 1e-3,
            agent_learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            device: str = "auto",
            save_dir: str = "./checkpoints",
            log_interval: int = 1,
            val_interval: int = 1,
            use_amp: bool = True,
            grad_clip_norm: float = 1.0,
            warmup_steps: int = 50,
            mem_context_cache_length: int = 10,
            # BPTT settings
            bptt_steps: int = 5,  # Number of windows for BPTT
            # Loss weights
            ce_weight: float = 1.0,
            gating_weight: float = 0.1,  # Weight for sensor gating loss
            # New loss parameters
            use_contrastive_loss: bool = False,
            contrastive_weight: float = 0.1,
            contrastive_tau: float = 0.1,
            memory_bank_size: int = 1000,
            use_predictive_loss: bool = False,
            predictive_weight: float = 0.1,
            predictive_offset: int = 1,
            # DDP settings
            ddp_rank: Optional[int] = None,
            ddp_world_size: Optional[int] = None,
            ddp_port: str = "12355"
    ):
        self.model = model
        self.agent = agent
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip_norm = grad_clip_norm
        self.warmup_steps = warmup_steps
        self.mem_context_cache_length = mem_context_cache_length
        self.bptt_steps = bptt_steps
        self.ce_weight = ce_weight
        self.gating_weight = gating_weight
        # New loss configurations
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_weight = contrastive_weight
        self.contrastive_tau = contrastive_tau
        self.memory_bank_size = memory_bank_size
        self.use_predictive_loss = use_predictive_loss
        self.predictive_weight = predictive_weight
        self.predictive_offset = predictive_offset

        # DDP setup
        self.is_ddp = ddp_rank is not None
        self.rank = ddp_rank if ddp_rank is not None else 0
        self.world_size = ddp_world_size if ddp_world_size is not None else 1
        self.is_main_process = self.rank == 0

        if self.is_ddp:
            setup_ddp(self.rank, self.world_size, ddp_port)

        # Setup device
        if device == "auto":
            if self.is_ddp:
                self.device = torch.device(f"cuda:{self.rank}")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.agent.to(self.device)

        # Wrap with DDP if needed
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
            self.agent = DDP(self.agent, device_ids=[self.rank], find_unused_parameters=True)

        # Setup optimizers
        self.model_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.agent_optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=agent_learning_rate,
            weight_decay=weight_decay
        )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Setup data loaders
        if self.is_ddp:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(train_dataset, rank=self.rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, rank=self.rank, shuffle=False) if val_dataset else None
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_sequential_batch,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_sequential_batch,
                num_workers=1,
                pin_memory=True,
                persistent_workers=True
            )

        # Create save directory
        if self.is_main_process:
            os.makedirs(save_dir, exist_ok=True)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_acc = 0.0

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Memory bank for contrastive loss
        if self.use_contrastive_loss:
            num_modalities = self.agent.module.num_modalities if self.is_ddp else self.agent.num_modalities
            self.memory_bank = deque(maxlen=memory_bank_size)
            self.memory_bank_labels = deque(maxlen=memory_bank_size)
            self.memory_bank_modality = deque(maxlen=memory_bank_size)
            # self.modalities_per_sensor = self._compute_modalities_per_sensor()

        # Predictive MLP for predictive coding loss
        if self.use_predictive_loss:
            model_dim = self.model.module.model_dim if self.is_ddp else self.model.model_dim
            self.predictive_mlp = nn.Sequential(
                nn.Linear(model_dim, model_dim * 2),
                nn.ReLU(),
                nn.Linear(model_dim * 2, model_dim)
            ).to(self.device)
            if self.is_ddp:
                self.predictive_mlp = DDP(self.predictive_mlp, device_ids=[self.rank], find_unused_parameters=True)
            self.predictive_optimizer = torch.optim.AdamW(
                self.predictive_mlp.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        if self.is_main_process:
            log_info(f"AgentSequentialTrainer initialized - Device: {self.device}")
            log_info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            log_info(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
            if self.use_predictive_loss:
                log_info(f"Predictive MLP parameters: {sum(p.numel() for p in self.predictive_mlp.parameters()):,}")
            log_info(f"BPTT steps: {bptt_steps}, CE weight: {ce_weight}, Gating weight: {gating_weight}")
            log_info(f"Contrastive loss: {use_contrastive_loss}, weight: {contrastive_weight}, tau: {contrastive_tau}")
            log_info(
                f"Predictive loss: {use_predictive_loss}, weight: {predictive_weight}, offset: {predictive_offset}")

    def _compute_modalities_per_sensor(self) -> Dict[int, List[int]]:
        """Group modality indices by device_idx."""
        modalities = self.model.module.modalities if self.is_ddp else self.model.modalities
        sensor_mods = defaultdict(list)
        for idx, modality in enumerate(modalities):
            sensor_mods[modality.device_idx].append(idx)
        return sensor_mods

    def get_lr_schedulers(self, total_steps: int):
        """Get learning rate schedulers with warmup for model, agent, and predictive MLP"""
        model_scheduler = OneCycleLR(
            self.model_optimizer,
            max_lr=self.model_optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=self.warmup_steps / total_steps,
            anneal_strategy='cos'
        )

        agent_scheduler = OneCycleLR(
            self.agent_optimizer,
            max_lr=self.agent_optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=self.warmup_steps / total_steps,
            anneal_strategy='cos'
        )

        schedulers = {"model": model_scheduler, "agent": agent_scheduler}
        if self.use_predictive_loss:
            predictive_scheduler = OneCycleLR(
                self.predictive_optimizer,
                max_lr=self.predictive_optimizer.param_groups[0]['lr'],
                total_steps=total_steps,
                pct_start=self.warmup_steps / total_steps,
                anneal_strategy='cos'
            )
            schedulers["predictive"] = predictive_scheduler

        return schedulers

    def compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor,
                                 modality_indices: List[int]) -> torch.Tensor:
        """
        Compute contrastive alignment loss using memory bank.
        embeddings: [B, D] embeddings for a specific modality at current timestep
        labels: [B] class labels
        modality_indices: List of modality indices for the current sensor
        """
        B, D = embeddings.shape
        device = embeddings.device
        contrastive_loss = torch.tensor(0.0, device=device)

        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, dim=-1)

        # Update memory bank
        for i in range(B):
            self.memory_bank.append(embeddings[i].detach().cpu())
            self.memory_bank_labels.append(labels[i].detach().cpu())
            self.memory_bank_modality.append(modality_indices)  # Store first modality index for simplicity

        if len(self.memory_bank) < 2:
            return contrastive_loss

        # Sample from memory bank
        bank_embeddings = torch.stack(list(self.memory_bank)).to(device)  # [N, D]
        bank_labels = torch.stack(list(self.memory_bank_labels)).to(device)  # [N]
        bank_modalities = torch.tensor(list(self.memory_bank_modality), device=device)  # [N]
        # print("bank_embeddings size:", bank_embeddings.size())

        # Compute contrastive loss for each sample
        for i in range(B):
            pos_mask = (bank_labels == labels[i]) & (bank_modalities != modality_indices)
            neg_mask = (bank_labels != labels[i]) | (bank_modalities == modality_indices)

            if not pos_mask.any():
                continue

            pos_embeds = bank_embeddings[pos_mask]
            neg_embeds = bank_embeddings[neg_mask]

            pos_sim = F.cosine_similarity(embeddings[i:i + 1], pos_embeds, dim=-1) / self.contrastive_tau
            neg_sim = F.cosine_similarity(embeddings[i:i + 1], neg_embeds,
                                          dim=-1) / self.contrastive_tau if neg_embeds.size(0) > 0 else torch.tensor(
                []).to(device)

            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim).sum() if neg_sim.numel() > 0 else torch.tensor(0.0, device=device)

            loss = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp + 1e-8))
            contrastive_loss += loss

        return contrastive_loss / max(1, B)

    def compute_predictive_loss(self, embeddings_t: torch.Tensor,
                                embeddings_t_plus_delta: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive coding loss.
        embeddings_t: [B, D] embeddings at time t
        embeddings_t_plus_delta: [B, D] embeddings at time t+delta
        """
        pred_embeddings = self.predictive_mlp(embeddings_t)  # [B, D]
        loss = F.mse_loss(pred_embeddings, embeddings_t_plus_delta, reduction='mean')
        return loss

    def train_step_bptt(self, batch: SequentialBatch) -> Dict[str, float]:
        """
        Training step with BPTT over multiple windows, including contrastive and predictive losses
        """
        self.model.train()
        self.agent.train()
        if self.use_predictive_loss:
            self.predictive_mlp.train()

        # Move to device
        sequences = batch.sequences.to(self.device)  # [B, max_seq_len, C, T]
        labels = batch.labels.to(self.device)  # [B, max_seq_len]
        seq_lengths = batch.seq_lengths.to(self.device)  # [B]

        B, max_seq_len, C, T = sequences.shape
        M = self.agent.module.num_modalities if self.is_ddp else self.agent.num_modalities
        if args.use_device_wise_model:
            num_features = self.agent.module.num_device if self.is_ddp else self.agent.num_device
        else:
            num_features = self.agent.module.num_modalities if self.is_ddp else self.agent.num_modalities

        # Initialize tracking variables
        total_ce_loss = 0.0
        total_gating_loss = 0.0
        total_contrastive_loss = 0.0
        total_predictive_loss = 0.0
        total_loss = 0.0
        correct_count = 0
        total_samples = 0
        total_sensor_usage = 0.0

        # Initialize memory and sensor history
        mem = None
        mem_cache = []
        mem_running_context = None
        sensor_history_length = self.agent.module.history_length if self.is_ddp else self.agent.history_length
        sensor_history = torch.ones(B, sensor_history_length, M).to(self.device)  # Start with all sensors on

        # Process sequences in chunks of bptt_steps
        num_chunks = (max_seq_len + self.bptt_steps - 1) // self.bptt_steps

        for chunk_idx in range(num_chunks):
            # Get chunk boundaries
            start_idx = chunk_idx * self.bptt_steps
            end_idx = min(start_idx + self.bptt_steps, max_seq_len)
            chunk_size = end_idx - start_idx

            # Zero gradients at start of each BPTT chunk
            self.model_optimizer.zero_grad(set_to_none=True)
            self.agent_optimizer.zero_grad(set_to_none=True)
            if self.use_predictive_loss:
                self.predictive_optimizer.zero_grad(set_to_none=True)

            # Accumulate losses for this chunk
            chunk_ce_loss = 0.0
            chunk_gating_loss = 0.0
            chunk_contrastive_loss = 0.0
            chunk_predictive_loss = 0.0
            chunk_samples = 0

            # Store intermediate states for BPTT
            chunk_mems = []
            chunk_p_softs = []
            prev_embeddings = []  # For predictive loss

            # Forward pass through chunk
            for i in range(start_idx, end_idx):
                window = sequences[:, i, :, :]  # [B, C, T]
                label = labels[:, i]  # [B]
                mask = label != -100

                if not mask.any():
                    continue

                # If we have memory from previous window, use agent to get gating
                if mem is not None and i > start_idx:
                    # Agent decides which sensors to use based on previous memory
                    with autocast(enabled=self.use_amp):
                        agent_output = self.agent(
                            represent_features=mem,  # Use memory as representation
                            sensor_history=sensor_history,
                            use_straight_through=True
                        )

                    p_soft = agent_output['p_soft']  # [B, M]
                    p_st = agent_output['p_st']  # [B, M] (hard forward, soft backward)

                    # Apply gating mask to input
                    # Reshape p_st to match window dimensions
                    # Assuming each modality corresponds to specific channels
                    channels_per_modality = C // M
                    p_st_expanded = p_st.unsqueeze(-1).unsqueeze(-1)  # [B, M, 1, 1]
                    p_st_expanded = p_st_expanded.repeat(1, 1, channels_per_modality, T)  # [B, M, channels_per_mod, T]
                    p_st_expanded = p_st_expanded.view(B, C, T)  # [B, C, T]

                    window_masked = window * p_st_expanded

                    # Update sensor history
                    sensor_history = torch.cat([
                        sensor_history[:, 1:, :],
                        torch.round(p_st).unsqueeze(1)
                    ], dim=1)

                    chunk_p_softs.append(p_soft)
                else:
                    # First window or no memory yet - use all sensors
                    window_masked = window
                    p_st = None
                    p_soft = None

                # Forward through model
                with autocast(enabled=self.use_amp):
                    logits, mem = self.model(window_masked, mem_running_context if i > start_idx else None)

                    # Store memory for context input (detached)
                    if mem_running_context is None:
                        mem_running_context = mem.detach()
                    else:
                        mem_cache.append(mem.detach())
                        if len(mem_cache) > self.mem_context_cache_length:
                            mem_cache.pop(0)
                        mem_running_context = torch.stack(mem_cache).mean(0)
                    # Store memory for BPTT
                    chunk_mems.append(mem)

                    # Compute cross-entropy loss
                    if mask.any():
                        ce_loss = self.criterion(logits[mask], label[mask])
                        chunk_ce_loss += ce_loss

                        # Track accuracy
                        with torch.no_grad():
                            preds = logits[mask].argmax(dim=1)
                            correct = preds.eq(label[mask]).sum().item()
                            correct_count += correct
                            chunk_samples += mask.sum().item()
                            total_samples += mask.sum().item()

                    # Add gating loss if we used the agent
                    if p_soft is not None:
                        gating_loss = p_soft.mean()
                        chunk_gating_loss += gating_loss
                        with torch.no_grad():
                            total_sensor_usage += p_st.detach().mean().item()

                    # Compute contrastive alignment loss
                    if self.use_contrastive_loss and mem is not None:
                        # mem: [B, L, D], where L = num_features * patches_per_feature
                        patches_per_device = mem.shape[1] // num_features
                        sensor_feats = mem.view(B, num_features, patches_per_device, -1).mean(dim=2)  # [B, n_d, D]
                        for device_idx in range(num_features):
                            sensor_embeds = sensor_feats[:, device_idx]  # [B, D]
                            contrastive_loss = self.compute_contrastive_loss(sensor_embeds[mask], label[mask],
                                                                             device_idx)
                            chunk_contrastive_loss += contrastive_loss

                    # Compute predictive coding loss
                    if self.use_predictive_loss and mem is not None:
                        prev_embeddings.append(mem.mean(dim=1))  # [B, D]
                        if len(prev_embeddings) > self.predictive_offset:
                            emb_t = prev_embeddings[-self.predictive_offset - 1]
                            emb_t_plus_delta = mem.mean(dim=1)  # Current embedding
                            predictive_loss = self.compute_predictive_loss(emb_t, emb_t_plus_delta)
                            chunk_predictive_loss += predictive_loss

            # Backward pass for this chunk
            if chunk_samples > 0:
                # Combine losses
                chunk_total_loss = (
                        self.ce_weight * chunk_ce_loss +
                        self.gating_weight * chunk_gating_loss +
                        self.contrastive_weight * chunk_contrastive_loss +
                        self.predictive_weight * chunk_predictive_loss
                )

                # Scale and backward
                self.scaler.scale(chunk_total_loss).backward()

                # Accumulate metrics
                total_ce_loss += chunk_ce_loss.item() if isinstance(chunk_ce_loss, torch.Tensor) else chunk_ce_loss
                total_gating_loss += chunk_gating_loss.item() if isinstance(chunk_gating_loss,
                                                                            torch.Tensor) else chunk_gating_loss
                total_contrastive_loss += chunk_contrastive_loss.item() if isinstance(chunk_contrastive_loss,
                                                                                      torch.Tensor) else chunk_contrastive_loss
                total_predictive_loss += chunk_predictive_loss.item() if isinstance(chunk_predictive_loss,
                                                                                    torch.Tensor) else chunk_predictive_loss
                total_loss += chunk_total_loss.item()

            # Detach memory for next chunk
            if chunk_mems:
                mem = chunk_mems[-1].detach() if chunk_mems[-1] is not None else None

            # Gradient clipping and optimization
            if chunk_samples > 0:
                # Unscale gradients
                self.scaler.unscale_(self.model_optimizer)
                self.scaler.unscale_(self.agent_optimizer)
                if self.use_predictive_loss:
                    self.scaler.unscale_(self.predictive_optimizer)

                # Gradient clipping
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
                    if self.use_predictive_loss:
                        torch.nn.utils.clip_grad_norm_(self.predictive_mlp.parameters(), self.grad_clip_norm)

                # Optimizer steps
                self.scaler.step(self.model_optimizer)
                self.scaler.step(self.agent_optimizer)
                if self.use_predictive_loss:
                    self.scaler.step(self.predictive_optimizer)
                self.scaler.update()

                if hasattr(self, 'schedulers'):
                    self.schedulers["model"].step()
                    self.schedulers["agent"].step()
                    if self.use_predictive_loss:
                        self.schedulers["predictive"].step()

        # Compute average metrics
        avg_ce_loss = total_ce_loss / num_chunks if num_chunks > 0 else 0.0
        avg_gating_loss = total_gating_loss / num_chunks if num_chunks > 0 else 0.0
        avg_contrastive_loss = total_contrastive_loss / num_chunks if num_chunks > 0 else 0.0
        avg_predictive_loss = total_predictive_loss / num_chunks if num_chunks > 0 else 0.0
        avg_total_loss = total_loss / num_chunks if num_chunks > 0 else 0.0
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        avg_sensor_usage = total_sensor_usage / num_chunks if num_chunks > 0 else 0.0

        return {
            "loss": avg_total_loss,
            "ce_loss": avg_ce_loss,
            "gating_loss": avg_gating_loss,
            "contrastive_loss": avg_contrastive_loss,
            "predictive_loss": avg_predictive_loss,
            "accuracy": accuracy,
            "sensor_usage": avg_sensor_usage,
            "total_samples": total_samples,
            "model_lr": self.model_optimizer.param_groups[0]['lr'],
            "agent_lr": self.agent_optimizer.param_groups[0]['lr'],
            "predictive_lr": self.predictive_optimizer.param_groups[0]['lr'] if self.use_predictive_loss else 0.0
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop with agent gating"""
        if not self.val_dataset:
            return {}

        self.model.eval()
        self.agent.eval()
        if self.use_predictive_loss:
            self.predictive_mlp.eval()

        total_ce_loss = 0.0
        total_gating_loss = 0.0
        correct_count = 0
        total_samples = 0
        total_sensor_usage = 0.0
        sensor_usage_counts = 0

        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Validation", leave=False, disable=not self.is_main_process):
            sequences = batch.sequences.to(self.device)
            labels = batch.labels.to(self.device)
            B, max_seq_len, C, T = sequences.shape
            M = self.agent.module.num_modalities if self.is_ddp else self.agent.num_modalities
            if args.use_device_wise_model:
                num_features = self.agent.module.num_device if self.is_ddp else self.agent.num_device
            else:
                num_features = self.agent.module.num_modalities if self.is_ddp else self.agent.num_modalities

            mem = None
            mem_cache = []
            mem_running_context = None
            sensor_history_length = self.agent.module.history_length if self.is_ddp else self.agent.history_length
            sensor_history = torch.ones(B, sensor_history_length, M).to(self.device)

            for i in range(max_seq_len):
                window = sequences[:, i, :, :]
                label = labels[:, i]
                mask = label != -100

                if not mask.any():
                    continue

                # Agent gating
                if mem is not None:
                    with autocast(enabled=self.use_amp):
                        agent_output = self.agent(
                            represent_features=mem,
                            sensor_history=sensor_history,
                            use_straight_through=False  # Use soft gating in validation
                        )

                    p_soft = agent_output['p_soft']

                    # Apply soft gating
                    channels_per_modality = C // M
                    p_soft_expanded = p_soft.unsqueeze(-1).unsqueeze(-1)
                    p_soft_expanded = p_soft_expanded.repeat(1, 1, channels_per_modality, T)
                    p_soft_expanded = p_soft_expanded.view(B, C, T)
                    window_masked = window * p_soft_expanded

                    # Update history
                    sensor_history = torch.cat([
                        sensor_history[:, 1:, :],
                        torch.round(p_soft).unsqueeze(1)
                    ], dim=1)

                    total_sensor_usage += p_soft.mean().item()
                    sensor_usage_counts += 1
                else:
                    window_masked = window

                # Model forward
                with autocast(enabled=self.use_amp):
                    logits, mem = self.model(window_masked, mem_running_context)

                    if mem_running_context is None:
                        mem_running_context = mem.detach()
                    else:
                        mem_cache.append(mem.detach())
                        if len(mem_cache) > self.mem_context_cache_length:
                            mem_cache.pop(0)
                        mem_running_context = torch.stack(mem_cache).mean(0)

                    if mask.any():
                        ce_loss = self.criterion(logits[mask], label[mask])
                        preds = logits[mask].argmax(dim=1)
                        correct = preds.eq(label[mask]).sum().item()

                        batch_size = mask.sum().item()
                        total_ce_loss += ce_loss.item() * batch_size
                        correct_count += correct
                        total_samples += batch_size

                        all_preds.append(preds.cpu())
                        all_labels.append(label[mask].cpu())
                        # print(preds.shape, label[mask].shape)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Synchronize across processes if DDP
        if self.is_ddp:
            metrics_tensor = torch.tensor(
                [total_ce_loss, correct_count, total_samples, total_sensor_usage, sensor_usage_counts],
                device=self.device
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_ce_loss, correct_count, total_samples, total_sensor_usage, sensor_usage_counts = metrics_tensor.tolist()

            all_preds_list = [torch.zeros_like(all_preds) for _ in range(self.world_size)]
            all_labels_list = [torch.zeros_like(all_labels) for _ in range(self.world_size)]
            torch.distributed.all_gather(all_preds_list, all_preds.to(self.device))
            torch.distributed.all_gather(all_labels_list, all_labels.to(self.device))
            all_preds = torch.cat(all_preds_list).numpy()
            all_labels = torch.cat(all_labels_list).numpy()

        if total_samples == 0:
            return {"val_loss": 0.0, "val_accuracy": 0.0, "val_sensor_usage": 1.0}

        metrics = compute_metrics(all_labels, all_preds, average='macro')
        print_metrics(metrics)

        avg_loss = total_ce_loss / total_samples
        avg_sensor_usage = total_sensor_usage / max(sensor_usage_counts, 1)

        return {
            "val_loss": round(avg_loss, 4),
            "val_sensor_usage": round(avg_sensor_usage, 4),
            "val_accuracy": round(metrics['accuracy'], 4),
            # "val_f1-score": round(metrics['f1_score'], 4),
            # "Confusion Matrix": metrics["confusion_matrix"]
        }

    def train(self, num_epochs: int = 10):
        """Main training loop with BPTT"""
        if self.is_main_process:
            log_info(f"Starting training for {num_epochs} epochs with BPTT (steps={self.bptt_steps})")

        # Calculate total steps for lr scheduler
        max_seq_len = int(max(next(iter(self.train_loader)).seq_lengths))
        steps_per_epoch = len(self.train_loader) * max_seq_len // self.bptt_steps
        total_steps = int(steps_per_epoch * num_epochs * 1.05)
        self.schedulers = self.get_lr_schedulers(total_steps)

        for epoch in range(num_epochs):
            log_info("="*70)
            log_info(f"Current Epoch: {epoch}")
            self.epoch = epoch

            # Set sampler epoch for DDP
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_losses = []
            epoch_accuracies = []
            epoch_sensor_usages = []
            epoch_contrastive_losses = []
            epoch_predictive_losses = []
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step_bptt(batch)

                if metrics["total_samples"] > 0:
                    epoch_losses.append(metrics["loss"])
                    epoch_accuracies.append(metrics["accuracy"])
                    epoch_sensor_usages.append(metrics["sensor_usage"])
                    epoch_contrastive_losses.append(metrics["contrastive_loss"])
                    epoch_predictive_losses.append(metrics["predictive_loss"])

                    if self.is_main_process and batch_idx % self.log_interval == 0:
                        log_info(f"Step {batch_idx}: loss={metrics['loss']:.4f}, "
                                 f"ce_loss={metrics['ce_loss']:.4f}, "
                                 f"gating_loss={metrics['gating_loss']:.4f}, "
                                 f"contrastive_loss={metrics['contrastive_loss']:.4f}, "
                                 f"predictive_loss={metrics['predictive_loss']:.4f}, "
                                 f"acc={metrics['accuracy']:.4f}, "
                                 f"sensor_usage={metrics['sensor_usage']:.3f}, "
                                 f"m_lr={metrics['model_lr']:.2e}, "
                                 f"a_lr={metrics['agent_lr']:.2e}, "
                                 f"p_lr={metrics['predictive_lr']:.2e}" if self.use_predictive_loss else ""
                        )

            if epoch_losses and self.is_main_process:
                epoch_time = time.time() - epoch_start_time
                epoch_loss = np.mean(epoch_losses)
                epoch_acc = np.mean(epoch_accuracies)
                epoch_sensors = np.mean(epoch_sensor_usages)
                epoch_contrastive = np.mean(epoch_contrastive_losses)
                epoch_predictive = np.mean(epoch_predictive_losses)
                log_info(f"Epoch {epoch} Summary: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, "
                         f"sensor_usage={epoch_sensors:.3f}, contrastive_loss={epoch_contrastive:.4f}, "
                         f"predictive_loss={epoch_predictive:.4f}, time={epoch_time:.1f}s")

            if self.step % self.val_interval == 0 and self.val_dataset:
                val_metrics = self.validate()
                if self.is_main_process:
                    log_info(f"Epoch {epoch} Validation: {val_metrics}")
                    if val_metrics.get("val_accuracy", 0) > self.best_val_acc:
                        self.best_val_acc = val_metrics["val_accuracy"]
                        self.save_checkpoint("best_model.pth")
                    log_info(f"The best model's performance: val_acc={self.best_val_acc:.4f}, "
                             f"sensor_usage={val_metrics['val_sensor_usage']:.3f}")

        if self.is_main_process:
            log_info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save model, agent, and predictive MLP checkpoint"""
        if not self.is_main_process:
            return

        # Get state dicts (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
        agent_state = self.agent.module.state_dict() if self.is_ddp else self.agent.state_dict()

        checkpoint = {
            "model_state_dict": model_state,
            "agent_state_dict": agent_state,
            "model_optimizer_state_dict": self.model_optimizer.state_dict(),
            "agent_optimizer_state_dict": self.agent_optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_val_acc": self.best_val_acc
        }

        if hasattr(self, 'schedulers'):
            checkpoint["model_lr_scheduler_state_dict"] = self.schedulers["model"].state_dict()
            checkpoint["agent_lr_scheduler_state_dict"] = self.schedulers["agent"].state_dict()
            if self.use_predictive_loss:
                checkpoint["predictive_lr_scheduler_state_dict"] = self.schedulers["predictive"].state_dict()
                checkpoint[
                    "predictive_mlp_state_dict"] = self.predictive_mlp.module.state_dict() if self.is_ddp else self.predictive_mlp.state_dict()
                checkpoint["predictive_optimizer_state_dict"] = self.predictive_optimizer.state_dict()

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str):
        """Load model, agent, and predictive MLP checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load state dicts (handle DDP)
        if self.is_ddp:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
            self.agent.module.load_state_dict(checkpoint["agent_state_dict"])
            if self.use_predictive_loss and "predictive_mlp_state_dict" in checkpoint:
                self.predictive_mlp.module.load_state_dict(checkpoint["predictive_mlp_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            if self.use_predictive_loss and "predictive_mlp_state_dict" in checkpoint:
                self.predictive_mlp.load_state_dict(checkpoint["predictive_mlp_state_dict"])

        self.model_optimizer.load_state_dict(checkpoint["model_optimizer_state_dict"])
        self.agent_optimizer.load_state_dict(checkpoint["agent_optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]

        if "model_lr_scheduler_state_dict" in checkpoint and hasattr(self, 'schedulers'):
            self.schedulers["model"].load_state_dict(checkpoint["model_lr_scheduler_state_dict"])
            self.schedulers["agent"].load_state_dict(checkpoint["agent_lr_scheduler_state_dict"])
            if self.use_predictive_loss and "predictive_lr_scheduler_state_dict" in checkpoint:
                self.schedulers["predictive"].load_state_dict(checkpoint["predictive_lr_scheduler_state_dict"])
                self.predictive_optimizer.load_state_dict(checkpoint["predictive_optimizer_state_dict"])

        if self.is_main_process:
            log_info(f"Checkpoint loaded from {filepath}")

    def cleanup(self):
        """Cleanup resources"""
        if self.is_ddp:
            cleanup_ddp()


def create_agent_trainer_from_dataset(
        dataset,
        train_subject_ids: List[str],
        val_subject_ids: Optional[List[str]] = None,
        model: Optional[nn.Module] = None,
        agent: Optional[SensorGatingAgent] = None,
        trainer_config: Optional[Dict] = None,
        ddp_config: Optional[Dict] = None
) -> AgentSequentialTrainer:
    """
    Convenience function to create agent trainer from ScientISSTMOVEDataset
    """
    train_sequential = SequentialDataset(dataset, train_subject_ids)
    val_sequential = None
    if val_subject_ids:
        val_sequential = SequentialDataset(dataset, val_subject_ids)

    trainer_config = trainer_config or {}
    ddp_config = ddp_config or {}
    trainer_config.update(ddp_config)

    trainer = AgentSequentialTrainer(
        model=model,
        agent=agent,
        train_dataset=train_sequential,
        val_dataset=val_sequential,
        **trainer_config
    )

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default="12355")
    parser.add_argument('--dataset', type=str, default='siscientisst')
    parser.add_argument('--root', type=str,
                        default='C:\Coding\Research\multimodal-medical-agent\dataset\mhealth+dataset')
    parser.add_argument('--model_lr', type=float, default=5e-5)
    parser.add_argument('--agent_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--bptt_steps', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--window_sec', type=float, default=1.0)
    parser.add_argument('--stride_sec', type=float, default=10)
    # Model Config
    parser.add_argument('--use_device_wise_model', action='store_true', help='use device embedded tokenizer based \
                        Transformer model(embeddings are not embeded in sensor-wise)')
    parser.add_argument('--use_fm_mdoel', action='store_true', help='use foundation model, can not setup fm model while\
                        use_device_wise_model is True')
    parser.add_argument('--modal_fusion', type=str, default='concate', help='Choose to use CrossModalAttention or \
                        concatenation')
    # Loss Config
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--gating_weight', type=float, default=0.05)
    parser.add_argument('--mem_context_cache_length', type=float, default=10)
    # New arguments for contrastive and predictive losses
    parser.add_argument('--use_contrastive_loss', action='store_true')
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    parser.add_argument('--contrastive_tau', type=float, default=0.1)
    parser.add_argument('--memory_bank_size', type=int, default=1000)

    parser.add_argument('--use_predictive_loss', action='store_true')
    parser.add_argument('--predictive_weight', type=float, default=0.1)
    parser.add_argument('--predictive_offset', type=int, default=1)

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    is_distributed = world_size > 1

    data_root = args.root
    if args.dataset == "siscientisst":
        dataset = ScientISSTMOVEDataset(root=data_root, window_sec=args.window_sec, stride_sec=args.stride_sec,
                                        none_policy="ignore")
        dataset = filter_labels(dataset, remove_labels=["sprint", "jumps"])
        print(dataset.label_map)
        train_subjects = dataset.subjects[:int(0.8 * len(dataset.subjects))]
        val_subjects = dataset.subjects[int(0.8 * len(dataset.subjects)):]
        num_classes = len(dataset.label_map)
        if not args.use_device_wise_model:
            modalities = [
                ModalityConfig('acc_c', 3, 10, 0), ModalityConfig('ecg_g_c', 1, 10, 0),
                ModalityConfig('ecg_t_c', 1, 10, 0),
                ModalityConfig('eda_f', 1, 10, 1), ModalityConfig('ppg_f', 1, 10, 1),
                ModalityConfig('emg_f', 1, 10, 1),
                ModalityConfig('eda_w', 1, 10, 2), ModalityConfig('ppg_w', 1, 10, 2),
                ModalityConfig('temp_w', 1, 10, 2), ModalityConfig('acc_w', 3, 10, 2),
            ]
        else:
            modalities = [
                ModalityConfig('c', 5, 10, 0),
                ModalityConfig('f', 3, 10, 1),
                ModalityConfig('w', 6, 10, 2),
            ]
    elif args.dataset == "mhealth":
        dataset = MHealthDataset(args.root, subjects=[i for i in range(0, 11)], time_steps=100, step=50,
                                 balance=False, majority_n=500)
        train_subjects = [i for i in range(1, 8)]
        val_subjects = [8, 9, 10]
        num_classes = 12
        weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        weights = weights.to(torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'))
        num_modal = 12
        if not args.use_device_wise_model:
            modalities = [
                ModalityConfig(f'm{i}', 1, 10, 0) for i in range(num_modal)
            ]
        else:
            modalities = [
                ModalityConfig('al', 3, 10, 0), ModalityConfig('gl', 3, 10, 0),
                ModalityConfig('ar', 3, 10, 1), ModalityConfig('gr', 3, 10, 1),
            ]
    elif args.dataset == "wesad":
        dataset = MultiModalWESADDataset(args.root, [2,3], window_sec=10, target_fs=64)
        train_subjects = [2]
        val_subjects = [3]
        num_classes = 3
        num_modal = 14
        if not args.use_device_wise_model:
            modalities = [
                # RespiBAN chest sensor (device 0)
                ModalityConfig('chest_acc', 3, 10, 0),
                ModalityConfig('chest_ecg', 1, 10, 0),
                ModalityConfig('chest_emg', 1, 10, 0),
                ModalityConfig('chest_eda', 1, 10, 0),
                ModalityConfig('chest_temp', 1, 10, 0),
                ModalityConfig('chest_resp', 1, 10, 0),

                # Empatica E4 wrist sensor (device 1)
                ModalityConfig('wrist_acc', 3, 10, 1),
                ModalityConfig('wrist_bvp', 1, 10, 1),
                ModalityConfig('wrist_eda', 1, 10, 1),
                ModalityConfig('wrist_temp', 1, 10, 1),
            ]
        else:
            # Device-wise grouping (optional)
            modalities = [
                ModalityConfig('chest', 8, 10, 0),
                ModalityConfig('wrist', 6, 10, 1),
            ]

    if args.use_device_wise_model:
        model = build_former_device(num_classes=num_classes, model_dim=512, return_mem=True, modalities=modalities, modal_fusion=args.modal_fusion)
        agent = DeviceGatingAgent(num_modalities=num_modal, modalities=modalities, feature_dim=512)
    else:
        if args.use_fm_mdoel:
            model = build_FMformer(num_classes=num_classes, model_dim=512, return_mem=True, modalities=modalities)
        else:
            model = build_former(num_classes=num_classes, model_dim=512, return_mem=True, modalities=modalities, modal_fusion=args.modal_fusion)
        agent = SensorGatingAgent(num_modalities=num_modal, feature_dim=512)

    ddp_config = {}
    if is_distributed:
        ddp_config = {
            "ddp_rank": local_rank,
            "ddp_world_size": world_size,
            "ddp_port": args.port
        }

    trainer = create_agent_trainer_from_dataset(
        dataset=dataset,
        train_subject_ids=train_subjects,
        val_subject_ids=val_subjects,
        model=model,
        agent=agent,
        trainer_config={
            "batch_size": args.batch_size,
            "bptt_steps": args.bptt_steps,
            "ce_weight": args.ce_weight,
            "gating_weight": args.gating_weight,
            "learning_rate": args.model_lr,
            "agent_learning_rate": args.agent_lr,
            "use_amp": True,
            "grad_clip_norm": 1.0,
            "val_interval": 1,
            "mem_context_cache_length": args.mem_context_cache_length,
            "use_contrastive_loss": args.use_contrastive_loss,
            "contrastive_weight": args.contrastive_weight,
            "contrastive_tau": args.contrastive_tau,
            "memory_bank_size": args.memory_bank_size,
            "use_predictive_loss": args.use_predictive_loss,
            "predictive_weight": args.predictive_weight,
            "predictive_offset": args.predictive_offset
        },
        ddp_config=ddp_config
    )

    trainer.train(num_epochs=args.num_epochs)
    trainer.cleanup()