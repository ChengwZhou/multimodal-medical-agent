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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR
import torch.distributed as dist
from torch.amp import GradScaler
from torch.amp import autocast

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
import time
from tqdm import tqdm
from dataclasses import dataclass
import json
from collections import deque, defaultdict

# Import base components
from models.former_sensor import build_former
from models.former_device import build_former_device
from models.agent_sensor_masking import SensorGatingAgent
from models.agent_device_masking import DeviceGatingAgent
from models.FMformer import build_FMformer, ModalityConfig

from sequential_trainer import (
    setup_ddp,
    cleanup_ddp,
    log_info
)
from dataset.sequential_dataset import (
    SequentialDataset,
    SequentialBatch,
    collate_sequential_batch,
    collate_sequential_batch_sync
)
from dataset.ScientISST_MOVE_loader import ScientISSTMOVEDataset, filter_labels
from dataset.mHEALTH_loader import MHealthDataset
from dataset.hmc_loader import HMCSleepDataset
from dataset.seizeit2_loader import (SeizeIT2Dataset, SeizeIT2IterableDataset,
                                     SeizeIT2PreextractedDataset,
                                     SeizeIT2PreextractedIterableDataset,
                                     preextract_to_dir)
from dataset.emowear_loader import EmoWearDataset, emowear_modality_configs
# from dataset.WESAD_loader import MultiModalWESADDataset
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
            train_dataset: Optional[SequentialDataset] = None,
            val_dataset: Optional[SequentialDataset] = None,
            # Pre-built loaders for large datasets that bypass SequentialDataset
            # (e.g. SeizeIT2 with ~41 M windows).  When supplied, train_dataset /
            # val_dataset are ignored and each batch is treated as a length-1 sequence.
            train_loader: Optional[DataLoader] = None,
            val_loader: Optional[DataLoader] = None,
            only_predictive: bool = False,
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
        self.only_predictive = only_predictive

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
            # find_unused_parameters=False: no unused params in our graph (verified by runtime warning).
            # _set_static_graph(): required for BPTT — each BPTT chunk calls model.forward() multiple
            # times before one backward(), which confuses DDP's per-iteration hook tracking.
            # _set_static_graph() tells DDP the compute graph is static so it doesn't reset ready-state
            # on every forward() call, allowing multiple forward+backward rounds per optimizer step.
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=False)
            self.model._set_static_graph()
            self.agent = DDP(self.agent, device_ids=[self.rank], find_unused_parameters=False)
            self.agent._set_static_graph()

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
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Setup data loaders
        if train_loader is not None:
            # --- Plain-loader path (large datasets, e.g. SeizeIT2) ---
            # Each batch is a plain (x [B,C,T], y [B]) tuple.
            # train_step_bptt / validate will wrap it into a length-1 SequentialBatch.
            self.train_loader = train_loader
            self.val_loader   = val_loader   # may be None
            self.max_len      = 1
        else:
            # --- SequentialDataset path (default) ---
            if self.is_ddp:
                from torch.utils.data.distributed import DistributedSampler
                train_sampler = DistributedSampler(train_dataset, rank=self.rank, shuffle=True)
                val_sampler = DistributedSampler(val_dataset, rank=self.rank, shuffle=False) if val_dataset else None
            else:
                train_sampler = None
                val_sampler = None

            max_len = train_dataset.max_length
            if self.is_ddp:
                max_tensor = torch.tensor([max_len], device=self.device)
                dist.all_reduce(max_tensor, op=dist.ReduceOp.MIN)
                max_len = max_tensor.item()
            self.max_len = max_len

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                collate_fn=collate_sequential_batch,
                num_workers=4,
                pin_memory=False,
                persistent_workers=True,
            )

            if val_dataset:
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=val_sampler,
                    collate_fn=collate_sequential_batch,
                    num_workers=4,
                    pin_memory=False,
                    persistent_workers=True
                )

        # Create save directory
        if self.is_main_process:
            os.makedirs(save_dir, exist_ok=True)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_acc = 0.0
        self.best_val_agent_ratio = 0.0
        self.global_step  = 0

        # Visualizer (attached via attach_visualizer before train())
        self.visualizer   = None
        self.viz_interval = 50

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Memory bank for contrastive loss
        if self.use_contrastive_loss:
            num_modalities = self.agent.module.num_modalities if self.is_ddp else self.agent.num_modalities
            self.memory_bank = deque(maxlen=memory_bank_size)
            self.memory_bank_labels = deque(maxlen=memory_bank_size)
            self.memory_bank_modality = deque(maxlen=memory_bank_size)

        # Predictive MLP for predictive coding loss
        if self.use_predictive_loss:
            model_dim = self.model.module.model_dim if self.is_ddp else self.model.model_dim
            self.predictive_mlp = nn.Sequential(
                nn.Linear(model_dim, model_dim * 2),
                nn.ReLU(),
                nn.Linear(model_dim * 2, model_dim)
            ).to(self.device)
            if self.is_ddp:
                self.predictive_mlp = DDP(self.predictive_mlp, device_ids=[self.rank], find_unused_parameters=False)
                self.predictive_mlp._set_static_graph()
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

    def attach_visualizer(self, visualizer, viz_interval: int = 50):
        """Attach a PomdpVisualizer for training visualization."""
        self.visualizer   = visualizer
        self.viz_interval = viz_interval

    def get_lr_schedulers(self, steps_per_epoch: int):
        """Get learning rate schedulers — no total_steps required.

        Uses CosineAnnealingWarmRestarts: LR decays cosine-style within each
        epoch and restarts at the beginning of the next.  ``T_0`` = one epoch
        in batches so the cycle aligns with epoch boundaries.
        A short linear warmup is applied via a LambdaLR multiplicative factor
        for the first ``warmup_steps`` batches.
        """
        def _make(optimizer):
            warmup = LambdaLR(
                optimizer,
                lr_lambda=lambda s: min(1.0, (s + 1) / max(self.warmup_steps, 1))
            )
            cosine = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(steps_per_epoch, 1),
                T_mult=1,
                eta_min=optimizer.param_groups[0]['lr'] * 0.01,
            )
            return SequentialLR(optimizer, schedulers=[warmup, cosine],
                                milestones=[self.warmup_steps])

        schedulers = {
            "model": _make(self.model_optimizer),
            "agent": _make(self.agent_optimizer),
        }
        if self.use_predictive_loss:
            schedulers["predictive"] = _make(self.predictive_optimizer)
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

    @staticmethod
    def _wrap_plain_batch(batch) -> SequentialBatch:
        """Wrap a plain batch as a SequentialBatch.

        Handles two shapes from ``SeizeIT2IterableDataset``:
        * ``x [B, C, T]``         — single window per item  → seq_len = 1
        * ``x [B, bptt, C, T]``   — BPTT sequence per item → seq_len = bptt
        """
        x, y = batch
        if x.dim() == 3:
            # Single-window batch: [B, C, T] → [B, 1, C, T]
            return SequentialBatch(
                sequences   = x.unsqueeze(1),
                labels      = y.unsqueeze(1),
                seq_lengths = torch.ones(x.shape[0], dtype=torch.long),
                subject_ids = [str(i) for i in range(x.shape[0])],
            )
        else:
            # BPTT batch: x is [B, bptt_steps, C, T], y is [B, bptt_steps]
            B, T_seq = x.shape[0], x.shape[1]
            return SequentialBatch(
                sequences   = x,                                           # [B, bptt, C, T]
                labels      = y,                                           # [B, bptt]
                seq_lengths = torch.full((B,), T_seq, dtype=torch.long),
                subject_ids = [str(i) for i in range(B)],
            )

    def train_step_bptt(self, batch) -> Dict[str, float]:
        """
        Training step with BPTT over multiple windows, including contrastive and predictive losses.
        Accepts either a :class:`SequentialBatch` (default) or a plain ``(x, y)`` tuple
        (ImageFolder-style datasets such as SeizeIT2).
        """
        if not isinstance(batch, SequentialBatch):
            batch = self._wrap_plain_batch(batch)

        self.model.train()
        self.agent.train()
        if self.use_predictive_loss:
            self.predictive_mlp.train()

        # Move to device
        sequences = batch.sequences.to(self.device)  # [B, max_seq_len, C, T]
        labels = batch.labels.to(self.device)  # [B, max_seq_len]
        seq_lengths = batch.seq_lengths.to(self.device)  # [B]

        B, max_seq_len, C, T = sequences.shape
        print("sequences.shape:", sequences.shape)

        # DDP requires every rank to call backward() (and thus all-reduce) the exact same
        # number of times.  Different recordings have different window counts, so each rank's
        # batch may have a different max_seq_len → different num_chunks → deadlock.
        # All-reduce to the global MAX so every rank uses identical num_chunks.
        # Ranks with shorter sequences pad sequences with zeros and labels with -100
        # (masked out in the loss), so extra chunks produce zero loss/gradient.
        if self.is_ddp:
            _len_t = torch.tensor(max_seq_len, device=self.device)
            dist.all_reduce(_len_t, op=dist.ReduceOp.MAX)
            global_max = int(_len_t.item())
            if global_max > max_seq_len:
                pad = global_max - max_seq_len
                sequences = F.pad(sequences, (0, 0, 0, 0, 0, pad))       # pad seq dim
                labels    = F.pad(labels,    (0, pad), value=-100)
            max_seq_len = global_max

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

        all_p_softs = []   # collect per-step p_soft [B, M] for viz

        for chunk_idx in range(num_chunks):
            # Get chunk boundaries
            start_idx = chunk_idx * self.bptt_steps
            end_idx = min(start_idx + self.bptt_steps, max_seq_len)
            chunk_size = end_idx - start_idx
            # print("chunk_size",  chunk_size)
            if chunk_size == 0:
                continue

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
            n_valid_windows = 0   # windows with at least one valid label
            n_agent_calls   = 0   # windows where agent was actually invoked

            # Store intermediate states for BPTT
            chunk_mems = []
            chunk_p_softs = []
            prev_embeddings = []  # For predictive loss

            # Forward pass through chunk
            for i in range(start_idx, end_idx):
                window = sequences[:, i, :, :]  # [B, C, T]
                label = labels[:, i]  # [B]
                mask = label != -100

                # if not mask.any():
                #     continue

                # If we have memory from previous window, use agent to get gating.
                # Gate whenever memory exists: chunk 0 starts with mem=None (natural
                # cold-start skip); chunk 1+ starts with detached mem from previous
                # chunk — gradient is truncated but gate decision is still applied,
                # matching validation behaviour.
                if mem is not None:
                    # Agent decides which sensors to use based on previous memory
                    with autocast('cuda', enabled=self.use_amp):
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

                    if not self.only_predictive:
                        window_masked = window * p_st_expanded
                    else:
                        window_masked = window

                    # Update sensor history
                    sensor_history = torch.cat([
                        sensor_history[:, 1:, :],
                        torch.round(p_st).unsqueeze(1)
                    ], dim=1)

                    chunk_p_softs.append(p_soft)
                    n_agent_calls += 1
                else:
                    # Cold start: no memory available yet
                    window_masked = window
                    p_st = None
                    p_soft = None

                # Forward through model
                # mem_running_context is always detached before storing, so passing
                # it here never creates cross-chunk gradient chains.
                with autocast('cuda', enabled=self.use_amp):
                    logits, mem = self.model(window_masked, mem_running_context)

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

                    # Always maintain a gradient-connected zero term so that even
                    # fully-padded chunks (all label==-100) produce a tensor rooted
                    # in model params.  AMP scaler and DDP hooks require at least one
                    # backward pass through model parameters per chunk.
                    if not isinstance(chunk_ce_loss, torch.Tensor):
                        chunk_ce_loss = logits.sum() * 0.0

                    # Compute cross-entropy loss
                    if mask.any():
                        ce_loss = self.criterion(logits[mask], label[mask])
                        chunk_ce_loss = chunk_ce_loss + ce_loss
                        n_valid_windows += 1

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

            # Accumulate p_softs from this chunk for viz
            all_p_softs.extend([ps.detach() for ps in chunk_p_softs])

            # Normalise each term so gradient magnitude is independent of chunk
            # occupancy.  The logits*0 seed keeps chunk_ce_loss as a tensor even
            # for fully-padded chunks, so AMP scaler and DDP hooks always fire.
            _vw = max(n_valid_windows, 1)
            _ac = max(n_agent_calls,   1)
            chunk_total_loss = (
                self.ce_weight          * chunk_ce_loss          / _vw +
                self.gating_weight      * chunk_gating_loss      / _ac +
                self.contrastive_weight * chunk_contrastive_loss / _vw +
                self.predictive_weight  * chunk_predictive_loss  / _vw
            )

            # print("111111", self.device, chunk_idx, num_chunks, chunk_samples)
            # Scale and backward
            self.scaler.scale(chunk_total_loss).backward()

            # print("111112", self.device, chunk_idx, num_chunks, chunk_samples)

            # Unscale gradients for clipping
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

            # print("111113", self.device, chunk_idx, num_chunks, chunk_samples)

            # Optimizer steps
            self.scaler.step(self.model_optimizer)
            # print("111115", self.device, chunk_idx, num_chunks, chunk_samples)
            self.scaler.step(self.agent_optimizer)
            if self.use_predictive_loss:
                self.scaler.step(self.predictive_optimizer)

            # Update scaler
            self.scaler.update()


            # Learning rate scheduler steps
            if hasattr(self, 'schedulers'):
                self.schedulers["model"].step()
                self.schedulers["agent"].step()
                if self.use_predictive_loss:
                    self.schedulers["predictive"].step()

            # Accumulate metrics
            total_ce_loss += chunk_ce_loss.item() if isinstance(chunk_ce_loss,
                                                                torch.Tensor) else chunk_ce_loss
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


        # Compute average metrics
        avg_ce_loss = total_ce_loss / num_chunks if num_chunks > 0 else 0.0
        avg_gating_loss = total_gating_loss / num_chunks if num_chunks > 0 else 0.0
        avg_contrastive_loss = total_contrastive_loss / num_chunks if num_chunks > 0 else 0.0
        avg_predictive_loss = total_predictive_loss / num_chunks if num_chunks > 0 else 0.0
        avg_total_loss = total_loss / num_chunks if num_chunks > 0 else 0.0
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        avg_sensor_usage = total_sensor_usage / num_chunks if num_chunks > 0 else 0.0

        # Mean p_soft per sensor across all windows [M]
        if all_p_softs:
            p_soft_per_sensor = torch.stack(all_p_softs).mean(0).mean(0).cpu().numpy()
        else:
            p_soft_per_sensor = None

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
            "predictive_lr": self.predictive_optimizer.param_groups[0]['lr'] if self.use_predictive_loss else 0.0,
            "p_soft_per_sensor": p_soft_per_sensor,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop with agent gating - support single GPU & DDP"""
        if not self.val_dataset and not self.val_loader:
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

        # Gate timeline for visualization (first sequence only)
        self._val_gate_seq  = None
        self._val_label_seq = None
        _seq0_gates  = []   # list of np [M] per step
        _seq0_labels = []   # list of int per step
        _first_batch_done = False

        # 是否使用 DistributedSampler（关键！）
        use_distributed_sampler = self.is_ddp and hasattr(self.val_loader.sampler, 'set_epoch')

        for batch in tqdm(self.val_loader, desc="Validation", leave=False, disable=not self.is_main_process):
            if not isinstance(batch, SequentialBatch):
                batch = self._wrap_plain_batch(batch)
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
                    with autocast('cuda', enabled=self.use_amp):
                        agent_output = self.agent(
                            represent_features=mem,
                            sensor_history=sensor_history,
                            use_straight_through=False
                        )
                    p_soft = agent_output['p_soft']

                    channels_per_modality = C // M
                    p_soft_expanded = p_soft.unsqueeze(-1).unsqueeze(-1)
                    p_soft_expanded = p_soft_expanded.repeat(1, 1, channels_per_modality, T)
                    p_soft_expanded = p_soft_expanded.view(B, C, T)

                    if not self.only_predictive:
                        window_masked = window * p_soft_expanded
                    else:
                        window_masked = window

                    sensor_history = torch.cat([
                        sensor_history[:, 1:, :],
                        torch.round(p_soft).unsqueeze(1)
                    ], dim=1)

                    total_sensor_usage += p_soft.mean().item()
                    sensor_usage_counts += 1

                    # Collect gate timeline for first batch
                    if not _first_batch_done and self.is_main_process:
                        _seq0_gates.append(p_soft[0].cpu().numpy())
                        _seq0_labels.append(label[0].item())
                else:
                    window_masked = window

                # Model forward
                with autocast('cuda', enabled=self.use_amp):
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

            # After the first batch's sequence is done, freeze gate collection
            if not _first_batch_done and _seq0_gates and self.is_main_process:
                self._val_gate_seq  = np.stack(_seq0_gates)   # [T, M]
                self._val_label_seq = np.array(_seq0_labels)  # [T]
            _first_batch_done = True

        # ========== 关键：智能合并预测结果 ==========
        if all_preds:
            all_preds = torch.cat(all_preds)  # [N_local]
            all_labels = torch.cat(all_labels)  # [N_local]
        else:
            all_preds = torch.tensor([], dtype=torch.long, device='cpu')
            all_labels = torch.tensor([], dtype=torch.long, device='cpu')

        # ---------- DDP 模式：同步 ----------
        if self.is_ddp:
            # 1. 同步标量指标
            metrics_tensor = torch.tensor(
                [total_ce_loss, correct_count, total_samples, total_sensor_usage, sensor_usage_counts],
                dtype=torch.float64, device=self.device
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_ce_loss, correct_count, total_samples, total_sensor_usage, sensor_usage_counts = metrics_tensor.tolist()

            # 2. 同步预测结果（设备安全版）
            if all_preds.numel() > 0:
                # 预分配输出 list（每个都在当前 GPU）
                pred_list = [torch.zeros_like(all_preds, device=self.device) for _ in range(self.world_size)]
                label_list = [torch.zeros_like(all_labels, device=self.device) for _ in range(self.world_size)]

                dist.all_gather(pred_list, all_preds.to(self.device))
                dist.all_gather(label_list, all_labels.to(self.device))

                all_preds = torch.cat(pred_list).cpu().numpy()
                all_labels = torch.cat(label_list).cpu().numpy()
            else:
                # 空预测：广播一个空 tensor
                empty_pred = torch.tensor([], dtype=torch.long, device=self.device)
                empty_label = torch.tensor([], dtype=torch.long, device=self.device)
                pred_list = [torch.zeros_like(empty_pred) for _ in range(self.world_size)]
                label_list = [torch.zeros_like(empty_label) for _ in range(self.world_size)]
                dist.all_gather(pred_list, empty_pred)
                dist.all_gather(label_list, empty_label)
                all_preds = np.array([])
                all_labels = np.array([])

        else:
            # ---------- 单卡模式：直接转 CPU ----------
            if all_preds.numel() > 0:
                all_preds = all_preds.cpu().numpy()
                all_labels = all_labels.cpu().numpy()
            else:
                all_preds = np.array([])
                all_labels = np.array([])

        # ========== 计算指标（仅主进程）==========
        if self.is_main_process:
            if total_samples == 0:
                return {"val_loss": 0.0, "val_accuracy": 0.0, "val_sensor_usage": 1.0}

            metrics = compute_metrics(all_labels, all_preds, use_weighted=True)
            print_metrics(metrics)

            avg_loss = total_ce_loss / total_samples
            avg_sensor_usage = total_sensor_usage / max(sensor_usage_counts, 1)

            return {
                "val_loss": round(avg_loss, 4),
                "val_sensor_usage": round(avg_sensor_usage, 4),
                "val_accuracy": round(metrics['accuracy'], 4),
            }
        else:
            return {}

    def train(self, num_epochs: int = 10):
        """Main training loop with BPTT"""
        if self.is_main_process:
            log_info(f"Starting training for {num_epochs} epochs with BPTT (steps={self.bptt_steps})")

        # Calculate total steps for lr scheduler.
        # For IterableDataset (e.g. SeizeIT2IterableDataset) we must NOT call
        # next(iter(loader)) here because:
        #   1. It triggers __iter__ which reads a full EDF recording (~30s on NFS)
        #   2. The iterator is then discarded, so the epoch loop reads it AGAIN
        # Instead, detect the batch type cheaply from dataset metadata.
        steps_per_epoch = len(self.train_loader)
        self.schedulers = self.get_lr_schedulers(steps_per_epoch)

        for epoch in range(num_epochs):
            log_info("="*70)
            log_info(f"Current Epoch: {epoch}")
            self.epoch = epoch

            # Set sampler epoch for DDP (DistributedSampler)
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            # Set epoch for SeizeIT2IterableDataset (controls per-epoch shuffle seed)
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)

            epoch_losses = []
            epoch_accuracies = []
            epoch_sensor_usages = []
            epoch_contrastive_losses = []
            epoch_predictive_losses = []
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step_bptt(batch)
                self.global_step += 1

                if metrics["total_samples"] > 0:
                    epoch_losses.append(metrics["loss"])
                    epoch_accuracies.append(metrics["accuracy"])
                    epoch_sensor_usages.append(metrics["sensor_usage"])
                    epoch_contrastive_losses.append(metrics["contrastive_loss"])
                    epoch_predictive_losses.append(metrics["predictive_loss"])

                    # Visualizer step logging
                    if (self.visualizer is not None and self.is_main_process
                            and self.global_step % self.viz_interval == 0):
                        p_soft_ms = metrics.get("p_soft_per_sensor")
                        if p_soft_ms is not None:
                            self.visualizer.log_step(self.global_step, {"p_soft": p_soft_ms})

                    if self.is_main_process and batch_idx % self.log_interval == 0:
                        log_msg = f"[{batch_idx}/{len(self.train_loader)}] loss={metrics['loss']:.4f}, " \
                                  f"ce_loss={metrics['ce_loss']:.4f}, " \
                                  f"gating_loss={metrics['gating_loss']:.4f}, " \
                                  f"contrastive_loss={metrics['contrastive_loss']:.4f}, " \
                                  f"predictive_loss={metrics['predictive_loss']:.4f}, " \
                                  f"acc={metrics['accuracy']:.4f}, " \
                                  f"sensor_usage={metrics['sensor_usage']:.3f}, " \
                                  f"m_lr={metrics['model_lr']:.2e}, " \
                                  f"a_lr={metrics['agent_lr']:.2e}"

                        if self.use_predictive_loss:
                            log_msg += f", p_lr={metrics['predictive_lr']:.2e}"

                        log_info(log_msg)

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

            if epoch % self.val_interval == 0 and (self.val_dataset or self.val_loader):
                val_metrics = self.validate()
                if self.is_main_process:
                    log_info(f"Epoch {epoch} Validation: {val_metrics}")
                    if val_metrics.get("val_accuracy", 0) > self.best_val_acc:
                        self.best_val_acc = val_metrics["val_accuracy"]
                        self.best_val_agent_ratio = val_metrics.get("val_sensor_usage", 0.0)
                        self.save_checkpoint("best_model.pth")
                    log_info(f"The best model's performance: val_acc={self.best_val_acc:.4f}, "
                             f"agent_ratio={self.best_val_agent_ratio:.3f}")

                    # Visualizer epoch / sequence logging
                    if self.visualizer is not None and epoch_losses:
                        train_m = {
                            "ce_loss":      float(np.mean(epoch_losses)),
                            "accuracy":     float(np.mean(epoch_accuracies)),
                            "sensor_usage": float(np.mean(epoch_sensor_usages)),
                        }
                        self.visualizer.log_epoch(epoch, train_m, val_metrics)
                        if self._val_gate_seq is not None:
                            self.visualizer.log_val_sequence(
                                epoch, self._val_gate_seq, self._val_label_seq)
                        self.visualizer.generate_plots()

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
        dataset=None,
        train_subject_ids: Optional[List[str]] = None,
        val_subject_ids: Optional[List[str]] = None,
        model: Optional[nn.Module] = None,
        agent: Optional[SensorGatingAgent] = None,
        trainer_config: Optional[Dict] = None,
        ddp_config: Optional[Dict] = None,
        # Pre-built loaders for large plain-batch datasets (bypass SequentialDataset)
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
) -> AgentSequentialTrainer:
    """
    Convenience function to create AgentSequentialTrainer.

    Pass ``train_loader`` / ``val_loader`` to bypass SequentialDataset for
    large datasets (e.g. SeizeIT2) that use ImageFolder-style plain batches.
    """
    trainer_config = trainer_config or {}
    ddp_config = ddp_config or {}
    trainer_config.update(ddp_config)

    if train_loader is not None:
        trainer = AgentSequentialTrainer(
            model=model,
            agent=agent,
            train_loader=train_loader,
            val_loader=val_loader,
            **trainer_config
        )
    else:
        train_sequential = SequentialDataset(dataset, train_subject_ids)
        val_sequential = None
        if val_subject_ids:
            val_sequential = SequentialDataset(dataset, val_subject_ids)

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
    parser.add_argument('--dataset', type=str, default='siscientisst',
                        choices=['siscientisst', 'mhealth', 'hmc', 'seizeit2', 'emowear'])
    parser.add_argument('--emowear_label', type=str, default='valence',
                        choices=['valence', 'arousal', 'quadrant'],
                        help='EmoWear label mode: valence/arousal (binary) or quadrant (4-class)')
    parser.add_argument('--root', type=str,
                        default='/Users/chengweizhou/PycharmProjects/data/scientisst-move-annotated-wearable-multimodal-biosignals-recorded-during-everyday-life-activities-in-naturalistic-environments-1.0.1')
    parser.add_argument('--model_lr', type=float, default=3e-4)
    parser.add_argument('--agent_lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--bptt_steps', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--window_sec', type=float, default=1.0)
    parser.add_argument('--stride_sec', type=float, default=10)
    # Model Config
    parser.add_argument('--only_predictive', action='store_true')
    parser.add_argument('--use_device_wise_model', action='store_true', help='use device embedded tokenizer based \
                        Transformer model(embeddings are not embeded in sensor-wise)')
    parser.add_argument('--use_fm_mdoel', action='store_true', help='use foundation model, can not setup fm model while\
                        use_device_wise_model is True')
    parser.add_argument('--modal_fusion', type=str, default='concate', help='Choose to use CrossModalAttention or \
                        concatenation')
    # Loss Config
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--gating_weight', type=float, default=0.1)
    parser.add_argument('--mem_context_cache_length', type=float, default=10)
    # New arguments for contrastive and predictive losses
    parser.add_argument('--use_contrastive_loss', action='store_true')
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    parser.add_argument('--contrastive_tau', type=float, default=0.1)
    parser.add_argument('--memory_bank_size', type=int, default=1000)

    parser.add_argument('--use_predictive_loss', action='store_true')
    parser.add_argument('--predictive_weight', type=float, default=0.1)
    parser.add_argument('--predictive_offset', type=int, default=1)
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Local directory to cache SeizeIT2 window index (avoids '
                             'repeated NFS header scans; recommended: /scratch/$USER/seizeit2_cache)')
    parser.add_argument('--num_data_workers', type=int, default=0,
                        help='DataLoader num_workers for SeizeIT2. Keep 0 on NFS mounts '
                             '(workers enter D-state and hang); set to 2-4 when data is on local SSD.')
    parser.add_argument('--preextracted_dir', type=str, default=None,
                        help='Path to pre-extracted numpy windows directory produced by '
                             '--preprocess_seizeit2. When set, uses SeizeIT2PreextractedDataset '
                             'for fast random access (batch_size=512, no EDF parsing).')
    parser.add_argument('--preprocess_seizeit2', action='store_true',
                        help='Extract all SeizeIT2 windows to --preextracted_dir and exit. '
                             'Run once offline before training. Requires --root and --preextracted_dir.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints.')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable training visualization via PomdpVisualizer.')
    parser.add_argument('--viz_dir', type=str, default='./viz',
                        help='Directory to save visualization outputs.')
    parser.add_argument('--viz_interval', type=int, default=50,
                        help='Log a gate snapshot every N global steps.')

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    is_distributed = world_size > 1

    data_root = args.root

    # Sentinel values — overwritten by the active dataset branch below
    dataset = None
    train_subjects = []
    val_subjects = []
    _seizeit2_train_loader = None
    _seizeit2_val_loader = None
    num_classes = 2
    num_modal = 1
    modalities = []

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
    elif args.dataset == "seizeit2":
        # SeizeIT2 (ds005873): wearable multimodal epilepsy seizure detection.
        # ~41 M windows total / ~330 K per subject — SequentialDataset cannot be used
        # (would require holding several GB per subject in RAM).
        # Instead we build plain shuffled DataLoaders (ImageFolder-style) and pass them
        # directly to AgentSequentialTrainer, which wraps each (x, y) batch into a
        # length-1 SequentialBatch inside train_step_bptt / validate.
        all_subjects = [f"sub-{i:03d}" for i in range(1, 126)]
        np.random.seed(42)
        _perm      = np.random.permutation(len(all_subjects))
        _split     = int(len(all_subjects) * 0.8)
        train_subjects = sorted([all_subjects[i] for i in _perm[:_split]])
        val_subjects   = sorted([all_subjects[i] for i in _perm[_split:]])

        num_classes = 2
        # SeizeIT2 stores modalities in separate sub-directories:
        #   ses-01/eeg/*_eeg.edf  → 2 EEG channels  (256 Hz → resampled to 250)
        #   ses-01/ecg/*_ecg.edf  → 1 ECG channel
        #   ses-01/mov/*_mov.edf  → 12 MOV channels (dual 6-axis IMU)
        # Total: 15 channels per recording run.
        num_modal   = 15
        if not args.use_device_wise_model:
            modalities = [ModalityConfig(f"m{i}", 1, 25, 0) for i in range(num_modal)]
        else:
            modalities = [
                ModalityConfig('eeg',  2, 25, 0),
                ModalityConfig('ecg',  1, 25, 1),
                ModalityConfig('mov', 12, 25, 2),
            ]

        _base_train = SeizeIT2Dataset(
            data_root=data_root, subjects=train_subjects,
            time_steps=500, step=500,
            modalities=('eeg', 'ecg', 'mov'), balance=False,
            cache_dir=args.cache_dir,
        )
        _base_val = SeizeIT2Dataset(
            data_root=data_root, subjects=val_subjects,
            time_steps=500, step=500,
            modalities=('eeg', 'ecg', 'mov'), balance=False,
            cache_dir=args.cache_dir,
        )

        # ------------------------------------------------------------------
        # --preprocess_seizeit2: extract all windows to numpy files and exit
        # ------------------------------------------------------------------
        if args.preprocess_seizeit2:
            assert args.preextracted_dir, "--preextracted_dir must be set with --preprocess_seizeit2"
            if local_rank == 0:
                log_info("=== Pre-extracting train set ===")
                preextract_to_dir(_base_train, args.preextracted_dir)
                log_info("=== Pre-extracting val set ===")
                preextract_to_dir(_base_val,   args.preextracted_dir)
                log_info("Pre-extraction complete. Re-run without --preprocess_seizeit2 to train.")
            import sys; sys.exit(0)

        _nw = args.num_data_workers

        if args.preextracted_dir:
            # ------------------------------------------------------------------
            # Fast path: memmap-backed IterableDataset — same BPTT semantics as
            # the NFS path but reads from local SSD (no EDF parsing, no OOM).
            # Supports larger batch_size (e.g. 16-32) than the NFS path.
            # ------------------------------------------------------------------
            _train_pre = SeizeIT2PreextractedIterableDataset(
                args.preextracted_dir, train_subjects,
                rank=local_rank, world_size=world_size, shuffle=True,
                expected_channels=num_modal,
            )
            _val_pre = SeizeIT2PreextractedIterableDataset(
                args.preextracted_dir, val_subjects,
                rank=local_rank, world_size=world_size, shuffle=False,
                expected_channels=num_modal,
            )

            def _pad_collate_pre(batch):
                xs, ys = zip(*batch)
                max_len = max(x.shape[0] for x in xs)
                C, T = xs[0].shape[1], xs[0].shape[2]
                xs_pad = torch.zeros(len(xs), max_len, C, T)
                ys_pad = torch.full((len(xs), max_len), -100, dtype=torch.long)
                for i, (x, y) in enumerate(zip(xs, ys)):
                    n = x.shape[0]
                    xs_pad[i, :n] = x
                    ys_pad[i, :n] = y
                return xs_pad, ys_pad

            _pf = 1 if _nw > 0 else None
            _seizeit2_train_loader = DataLoader(
                _train_pre, batch_size=args.batch_size, shuffle=False,
                num_workers=_nw, pin_memory=False,
                persistent_workers=(_nw > 0), prefetch_factor=_pf,
                collate_fn=_pad_collate_pre,
            )
            _seizeit2_val_loader = DataLoader(
                _val_pre, batch_size=args.batch_size, shuffle=False,
                num_workers=_nw, pin_memory=False,
                persistent_workers=(_nw > 0), prefetch_factor=_pf,
                collate_fn=_pad_collate_pre,
            )
        else:
            # ------------------------------------------------------------------
            # NFS path: IterableDataset, yields full recordings for BPTT
            # ------------------------------------------------------------------
            _train_ds = SeizeIT2IterableDataset(
                _base_train, bptt_steps=args.bptt_steps,
                rank=local_rank, world_size=world_size, shuffle=True,
                expected_channels=num_modal,
            )
            _val_ds = SeizeIT2IterableDataset(
                _base_val, bptt_steps=args.bptt_steps,
                rank=local_rank, world_size=world_size, shuffle=False,
                expected_channels=num_modal,

            )

            # Pad variable-length recordings so they can be stacked in a batch
            def _pad_collate(batch):
                xs, ys = zip(*batch)
                max_len = max(x.shape[0] for x in xs)
                C, T = xs[0].shape[1], xs[0].shape[2]
                xs_pad = torch.zeros(len(xs), max_len, C, T)
                ys_pad = torch.full((len(xs), max_len), -100, dtype=torch.long)
                for i, (x, y) in enumerate(zip(xs, ys)):
                    n = x.shape[0]
                    xs_pad[i, :n] = x
                    ys_pad[i, :n] = y
                return xs_pad, ys_pad

            # Full recordings are ~180 MB each; limit prefetch to 1 to avoid OOM.
            _pf = 1 if _nw > 0 else None
            _seizeit2_train_loader = DataLoader(
                _train_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=_nw, pin_memory=False,
                persistent_workers=(_nw > 0), prefetch_factor=_pf,
                collate_fn=_pad_collate,
            )
            _seizeit2_val_loader = DataLoader(
                _val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=_nw, pin_memory=False,
                persistent_workers=(_nw > 0), prefetch_factor=_pf,
                collate_fn=_pad_collate,
            )

        dataset = None   # not used — loaders are passed directly

    elif args.dataset == "hmc":
        global_stats = {"mean": np.array([9.1890168e-01, 1.9557451e+00, 2.4014959e+00, 1.6120193e+00,
                        7.8689933e-05, 1.9333732e+00, 3.5932889e+00, 2.4175742e+00]),
                        "std": np.array([ 40.361404, 25.491713, 28.026707, 35.241245, 3.7148967,
                        30.820967, 41.629414, 124.64315])}

        # Final: {0: 23686, 1: 15548, 2: 50083, 3: 26671, 4: 21255} | Total: 137243
        dataset = HMCSleepDataset(
            data_root=data_root,
            subjects=None,
            balance=False,
            remove_wake=False,
            apply_notch=True, notch_freq=50.0,
            apply_emg_hp=True,
            apply_ecg_filter=False,
            global_stats=global_stats
        )  # 30s-windows
        train_ratio = 0.8
        all_subjects = dataset.subjects

        np.random.seed(42)
        np.random.shuffle(all_subjects)

        split_idx = int(len(all_subjects) * train_ratio)
        train_subjects = sorted(all_subjects[:split_idx])
        val_subjects = sorted(all_subjects[split_idx:])

        num_classes = 5
        weights = torch.tensor([1.1, 1.5, 0.5, 1, 1.25])
        weights = weights.to(torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'))
        num_modal = 8
        # if not args.use_device_wise_model:
        #     modalities = [
        #         ModalityConfig(f'm{i}', 1, 10, 0) for i in range(num_modal)
        #     ]
        # else:
        #     modalities = [
        #         ModalityConfig('al', 3, 10, 0), ModalityConfig('gl', 3, 10, 0),
        #         ModalityConfig('ar', 3, 10, 1), ModalityConfig('gr', 3, 10, 1),
        #     ]
        modalities = [
            ModalityConfig(f'm{i}', 1, 30, 0) for i in range(num_modal)
        ]
    elif args.dataset == "emowear":
        # EmoWear (Zenodo 10407279): 49 subjects, 38 trials × 60 s, 5 devices, FS=64 Hz
        # Default modalities: ECG, RSP (BH3); BVP, EDA, SKT, ACC (E4); ACC, GYRO (front STb)
        # → 14 channels total
        _emowear_modalities = ('ecg', 'rsp', 'bvp', 'eda', 'skt', 'acc_e4',
                               'acc_front', 'gyro_front')
        _emowear_label_mode = args.emowear_label
        num_classes = 4 if _emowear_label_mode == 'quadrant' else 2
        modalities = emowear_modality_configs(_emowear_modalities, patch_size=8)
        num_modal = sum(m.in_ch for m in modalities)  # 14

        # Discover all subjects and split 80/20
        _all_emowear_subs = EmoWearDataset(data_root)._discover_subjects()
        np.random.seed(42)
        _perm = np.random.permutation(len(_all_emowear_subs))
        _split = int(len(_all_emowear_subs) * 0.8)
        train_subjects = [_all_emowear_subs[i] for i in _perm[:_split]]
        val_subjects   = [_all_emowear_subs[i] for i in _perm[_split:]]

        dataset = EmoWearDataset(
            data_root=data_root,
            subjects=train_subjects + val_subjects,
            time_steps=384,   # 6 s at 64 Hz
            step=64,          # 1 s stride
            modalities=_emowear_modalities,
            label_mode=_emowear_label_mode,
            balance=False,
        )

    # elif args.dataset == "wesad":
    #     dataset = MultiModalWESADDataset(args.root, [2,3], window_sec=10, target_fs=64)
    #     train_subjects = [2]
    #     val_subjects = [3]
    #     num_classes = 3
    #     num_modal = 14
    #     if not args.use_device_wise_model:
    #         modalities = [
    #             # RespiBAN chest sensor (device 0)
    #             ModalityConfig('chest_acc', 3, 10, 0),
    #             ModalityConfig('chest_ecg', 1, 10, 0),
    #             ModalityConfig('chest_emg', 1, 10, 0),
    #             ModalityConfig('chest_eda', 1, 10, 0),
    #             ModalityConfig('chest_temp', 1, 10, 0),
    #             ModalityConfig('chest_resp', 1, 10, 0),
    #
    #             # Empatica E4 wrist sensor (device 1)
    #             ModalityConfig('wrist_acc', 3, 10, 1),
    #             ModalityConfig('wrist_bvp', 1, 10, 1),
    #             ModalityConfig('wrist_eda', 1, 10, 1),
    #             ModalityConfig('wrist_temp', 1, 10, 1),
    #         ]
    #     else:
    #         # Device-wise grouping (optional)
    #         modalities = [
    #             ModalityConfig('chest', 8, 10, 0),
    #             ModalityConfig('wrist', 6, 10, 1),
    #         ]

    if args.use_device_wise_model:
        model = build_former_device(num_classes=num_classes, model_dim=512, return_mem=True, modalities=modalities, modal_fusion=args.modal_fusion)
        agent = DeviceGatingAgent(num_modalities=num_modal, modalities=modalities, feature_dim=512)
    else:
        if args.use_fm_mdoel:
            model = build_FMformer(num_classes=num_classes, model_dim=512, return_mem=True, modalities=modalities)
        else:
            model = build_former(num_classes=num_classes, model_dim=512, return_mem=True, modalities=modalities, modal_fusion=args.modal_fusion)
            # model = build_former(num_classes=num_classes, model_dim=512, return_mem=True,
            #                                     return_sensing_info=False,
            #                                     skip_steps=5,
            #                                     init_threshold=1e-7,
            #                                     modalities=modalities, modal_fusion=args.modal_fusion)
        agent = SensorGatingAgent(num_modalities=num_modal, feature_dim=512)

    ddp_config = {}
    if is_distributed:
        ddp_config = {
            "ddp_rank": local_rank,
            "ddp_world_size": world_size,
            "ddp_port": args.port
        }

    _trainer_config = {
        "batch_size": args.batch_size,
        "bptt_steps": args.bptt_steps,
        "ce_weight": args.ce_weight,
        "gating_weight": args.gating_weight,
        "learning_rate": args.model_lr,
        "agent_learning_rate": args.agent_lr,
        "use_amp": True,
        "grad_clip_norm": 1.0,
        "val_interval": 1,
        "save_dir": args.save_dir,
        "mem_context_cache_length": args.mem_context_cache_length,
        "use_contrastive_loss": args.use_contrastive_loss,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_tau": args.contrastive_tau,
        "memory_bank_size": args.memory_bank_size,
        "use_predictive_loss": args.use_predictive_loss,
        "predictive_weight": args.predictive_weight,
        "predictive_offset": args.predictive_offset,
        "only_predictive": args.only_predictive
    }

    if args.dataset == "seizeit2":
        # Plain-loader path: bypass SequentialDataset
        trainer = create_agent_trainer_from_dataset(
            model=model, agent=agent,
            train_loader=_seizeit2_train_loader,
            val_loader=_seizeit2_val_loader,
            trainer_config=_trainer_config,
            ddp_config=ddp_config,
        )
    else:
        trainer = create_agent_trainer_from_dataset(
            dataset=dataset,
            train_subject_ids=train_subjects,
            val_subject_ids=val_subjects,
            model=model, agent=agent,
            trainer_config=_trainer_config,
            ddp_config=ddp_config,
        )

    if args.visualize and trainer.is_main_process:
        from visualization.pomdp_viz import PomdpVisualizer
        if args.use_device_wise_model:
            # DeviceGatingAgent gates per channel (num_modalities = total channels)
            gate_names = [f"{m.name}_{c}" for m in modalities for c in range(m.in_ch)]
            gate_label  = "Channel"
        else:
            gate_names = [m.name for m in modalities]
            gate_label  = "Sensor"
        viz = PomdpVisualizer(
            viz_dir=args.viz_dir,
            num_sensors=len(gate_names),
            sensor_names=gate_names,
            gate_label=gate_label,
            uncertainty_mode="none",
        )
        trainer.attach_visualizer(viz, viz_interval=args.viz_interval)

    trainer.train(num_epochs=args.num_epochs)
    trainer.cleanup()