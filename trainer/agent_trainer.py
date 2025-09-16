# =====================================
# File: agent_sequential_trainer.py
# Sequential trainer with SensorGatingAgent for ScientISSTMOVE dataset
# Supports BPTT, mixed precision, and sensor gating
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

# Import base components
from sequential_trainer import (
    SequentialDataset,
    SequentialBatch,
    collate_sequential_batch,
    setup_ddp,
    cleanup_ddp,
    log_info
)

from models.sensor_gating_agent import SensorGatingAgent
from models.former import build_former
from dataset.ScientISST_MOVE_loader import ScientISSTMOVEDataset, filter_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSequentialTrainer:
    """
    Trainer for sequential prediction with SensorGatingAgent
    Jointly trains model and agent using BPTT
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
            log_interval: int = 10,
            val_interval: int = 100,
            use_amp: bool = True,
            grad_clip_norm: float = 1.0,
            warmup_steps: int = 50,
            # BPTT settings
            bptt_steps: int = 5,  # Number of windows for BPTT
            # Loss weights
            ce_weight: float = 1.0,
            gating_weight: float = 0.1,  # Weight for sensor gating loss
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
        self.bptt_steps = bptt_steps
        self.ce_weight = ce_weight
        self.gating_weight = gating_weight

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

        # Setup optimizers (separate for model and agent)
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
            num_workers=4,
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
                num_workers=4,
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

        if self.is_main_process:
            log_info(f"AgentSequentialTrainer initialized - Device: {self.device}")
            log_info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            log_info(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
            log_info(f"BPTT steps: {bptt_steps}, CE weight: {ce_weight}, Gating weight: {gating_weight}")

    def get_lr_schedulers(self, total_steps: int):
        """Get learning rate schedulers with warmup for both model and agent"""
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

        return model_scheduler, agent_scheduler

    def train_step_bptt(self, batch: SequentialBatch) -> Dict[str, float]:
        """
        Training step with BPTT over multiple windows
        """
        self.model.train()
        self.agent.train()

        # Move to device
        sequences = batch.sequences.to(self.device)  # [B, max_seq_len, C, T]
        labels = batch.labels.to(self.device)  # [B, max_seq_len]
        seq_lengths = batch.seq_lengths.to(self.device)  # [B]

        B, max_seq_len, C, T = sequences.shape
        M = self.agent.num_modalities

        # Initialize tracking variables
        total_ce_loss = 0.0
        total_gating_loss = 0.0
        total_loss = 0.0
        correct_count = 0
        total_samples = 0
        total_sensor_usage = 0.0

        # Initialize memory and sensor history
        mem = None
        sensor_history = torch.ones(B, self.agent.history_length, M).to(self.device)  # Start with all sensors on

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

            # Accumulate losses for this chunk
            chunk_ce_loss = 0.0
            chunk_gating_loss = 0.0
            chunk_samples = 0

            # Store intermediate states for BPTT
            chunk_mems = []
            chunk_p_softs = []

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
                    p_soft = None

                # Forward through model
                with autocast(enabled=self.use_amp):
                    logits, mem = self.model(window_masked, mem if i > start_idx else None)

                    # Store memory for BPTT (don't detach within chunk)
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
                        # Encourage turning off sensors (minimize p_soft)
                        gating_loss = p_soft.mean()
                        chunk_gating_loss += gating_loss

                        with torch.no_grad():
                            total_sensor_usage += p_soft.mean().item()

            # Backward pass for this chunk
            if chunk_samples > 0:
                # Combine losses
                chunk_total_loss = self.ce_weight * chunk_ce_loss + self.gating_weight * chunk_gating_loss

                # Scale and backward
                self.scaler.scale(chunk_total_loss).backward()

                # Accumulate metrics
                total_ce_loss += chunk_ce_loss.item() if isinstance(chunk_ce_loss, torch.Tensor) else chunk_ce_loss
                total_gating_loss += chunk_gating_loss.item() if isinstance(chunk_gating_loss,
                                                                            torch.Tensor) else chunk_gating_loss
                total_loss += chunk_total_loss.item()

            # Detach memory for next chunk to break gradient flow
            if chunk_mems:
                mem = chunk_mems[-1].detach() if chunk_mems[-1] is not None else None

            # Gradient clipping and optimization
            if chunk_samples > 0:
                # Unscale gradients
                self.scaler.unscale_(self.model_optimizer)
                self.scaler.unscale_(self.agent_optimizer)

                # Gradient clipping
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)

                # Optimizer steps
                self.scaler.step(self.model_optimizer)
                self.scaler.step(self.agent_optimizer)
                self.scaler.update()

                # Update learning rates
                if hasattr(self, 'model_lr_scheduler'):
                    self.model_lr_scheduler.step()
                if hasattr(self, 'agent_lr_scheduler'):
                    self.agent_lr_scheduler.step()

        # Compute average metrics
        avg_ce_loss = total_ce_loss / num_chunks if num_chunks > 0 else 0.0
        avg_gating_loss = total_gating_loss / num_chunks if num_chunks > 0 else 0.0
        avg_total_loss = total_loss / num_chunks if num_chunks > 0 else 0.0
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        avg_sensor_usage = total_sensor_usage / num_chunks if num_chunks > 0 else 0.0

        return {
            "loss": avg_total_loss,
            "ce_loss": avg_ce_loss,
            "gating_loss": avg_gating_loss,
            "accuracy": accuracy,
            "sensor_usage": avg_sensor_usage,
            "total_samples": total_samples,
            "model_lr": self.model_optimizer.param_groups[0]['lr'],
            "agent_lr": self.agent_optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop with agent gating"""
        if not self.val_dataset:
            return {}

        self.model.eval()
        self.agent.eval()

        total_ce_loss = 0.0
        total_gating_loss = 0.0
        correct_count = 0
        total_samples = 0
        total_sensor_usage = 0.0
        sensor_usage_counts = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False, disable=not self.is_main_process):
            sequences = batch.sequences.to(self.device)
            labels = batch.labels.to(self.device)
            B, max_seq_len, C, T = sequences.shape
            M = self.agent.num_modalities

            mem = None
            sensor_history = torch.ones(B, self.agent.history_length, M).to(self.device)

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
                    logits, mem = self.model(window_masked, mem)

                    if mask.any():
                        ce_loss = self.criterion(logits[mask], label[mask])
                        preds = logits[mask].argmax(dim=1)
                        correct = preds.eq(label[mask]).sum().item()

                        batch_size = mask.sum().item()
                        total_ce_loss += ce_loss.item() * batch_size
                        correct_count += correct
                        total_samples += batch_size

        # Synchronize across processes if DDP
        if self.is_ddp:
            metrics_tensor = torch.tensor(
                [total_ce_loss, correct_count, total_samples, total_sensor_usage, sensor_usage_counts],
                device=self.device
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_ce_loss, correct_count, total_samples, total_sensor_usage, sensor_usage_counts = metrics_tensor.tolist()

        if total_samples == 0:
            return {"val_loss": 0.0, "val_accuracy": 0.0, "val_sensor_usage": 1.0}

        avg_loss = total_ce_loss / total_samples
        avg_accuracy = correct_count / total_samples
        avg_sensor_usage = total_sensor_usage / max(sensor_usage_counts, 1)

        return {
            "val_loss": avg_loss,
            "val_accuracy": avg_accuracy,
            "val_sensor_usage": avg_sensor_usage
        }

    def train(self, num_epochs: int = 10):
        """Main training loop with BPTT"""
        if self.is_main_process:
            log_info(f"Starting training for {num_epochs} epochs with BPTT (steps={self.bptt_steps})")

        # Calculate total steps for lr scheduler
        max_seq_len = int(max(next(iter(self.train_loader)).seq_lengths))
        steps_per_epoch = len(self.train_loader) * max_seq_len // self.bptt_steps
        total_steps = steps_per_epoch * num_epochs
        self.model_lr_scheduler, self.agent_lr_scheduler = self.get_lr_schedulers(total_steps)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Set sampler epoch for DDP
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_losses = []
            epoch_accuracies = []
            epoch_sensor_usages = []
            epoch_start_time = time.time()

            # Training loop
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                        disable=not self.is_main_process)

            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step_bptt(batch)

                if metrics["total_samples"] > 0:
                    epoch_losses.append(metrics["loss"])
                    epoch_accuracies.append(metrics["accuracy"])
                    epoch_sensor_usages.append(metrics["sensor_usage"])

                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "acc": f"{metrics['accuracy']:.4f}",
                        "sensors": f"{metrics['sensor_usage']:.2f}",
                        "m_lr": f"{metrics['model_lr']:.2e}",
                        "a_lr": f"{metrics['agent_lr']:.2e}"
                    })

                    # Logging
                    if self.is_main_process and self.step % self.log_interval == 0:
                        log_info(f"Step {self.step}: loss={metrics['loss']:.4f}, "
                                 f"ce_loss={metrics['ce_loss']:.4f}, "
                                 f"gating_loss={metrics['gating_loss']:.4f}, "
                                 f"acc={metrics['accuracy']:.4f}, "
                                 f"sensor_usage={metrics['sensor_usage']:.3f}")

                    # Validation
                    if self.step % self.val_interval == 0 and self.val_dataset:
                        val_metrics = self.validate()
                        if self.is_main_process:
                            log_info(f"Step {self.step} Validation: {val_metrics}")

                            # Save best model based on accuracy
                            if val_metrics.get("val_accuracy", 0) > self.best_val_acc:
                                self.best_val_acc = val_metrics["val_accuracy"]
                                self.save_checkpoint("best_model.pth")
                                log_info(f"New best model saved with val_acc={self.best_val_acc:.4f}, "
                                         f"sensor_usage={val_metrics['val_sensor_usage']:.3f}")

                    self.step += 1

            # End of epoch summary
            if epoch_losses and self.is_main_process:
                epoch_time = time.time() - epoch_start_time
                epoch_loss = np.mean(epoch_losses)
                epoch_acc = np.mean(epoch_accuracies)
                epoch_sensors = np.mean(epoch_sensor_usages)
                log_info(f"Epoch {epoch + 1} Summary: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, "
                         f"sensor_usage={epoch_sensors:.3f}, time={epoch_time:.1f}s")

                # Save checkpoint
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")

        if self.is_main_process:
            log_info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save model and agent checkpoint"""
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

        if hasattr(self, 'model_lr_scheduler'):
            checkpoint["model_lr_scheduler_state_dict"] = self.model_lr_scheduler.state_dict()
        if hasattr(self, 'agent_lr_scheduler'):
            checkpoint["agent_lr_scheduler_state_dict"] = self.agent_lr_scheduler.state_dict()

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str):
        """Load model and agent checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load state dicts (handle DDP)
        if self.is_ddp:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
            self.agent.module.load_state_dict(checkpoint["agent_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.agent.load_state_dict(checkpoint["agent_state_dict"])

        self.model_optimizer.load_state_dict(checkpoint["model_optimizer_state_dict"])
        self.agent_optimizer.load_state_dict(checkpoint["agent_optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]

        if "model_lr_scheduler_state_dict" in checkpoint and hasattr(self, 'model_lr_scheduler'):
            self.model_lr_scheduler.load_state_dict(checkpoint["model_lr_scheduler_state_dict"])
        if "agent_lr_scheduler_state_dict" in checkpoint and hasattr(self, 'agent_lr_scheduler'):
            self.agent_lr_scheduler.load_state_dict(checkpoint["agent_lr_scheduler_state_dict"])

        if self.is_main_process:
            log_info(f"Checkpoint loaded from {filepath}")

    def cleanup(self):
        """Cleanup resources"""
        if self.is_ddp:
            cleanup_ddp()


def create_agent_trainer_from_dataset(
        scientisst_dataset,
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
    # Create sequential datasets
    train_sequential = SequentialDataset(scientisst_dataset, train_subject_ids)
    val_sequential = None
    if val_subject_ids:
        val_sequential = SequentialDataset(scientisst_dataset, val_subject_ids)

    # Trainer configuration
    trainer_config = trainer_config or {}
    ddp_config = ddp_config or {}

    # Merge DDP config
    trainer_config.update(ddp_config)

    # Create trainer
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
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--bptt_steps', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--gating_weight', type=float, default=0.1)
    parser.add_argument('--root', type=str,
                        default='/Users/chengweizhou/PycharmProjects/data/scientisst-move-annotated-wearable-multimodal-biosignals-recorded-during-everyday-life-activities-in-naturalistic-environments-1.0.1')
    args = parser.parse_args()

    # detect torchrun env
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    is_distributed = world_size > 1

    # build dataset and model (each process will run this; that's fine)
    data_root = args.root

    dataset = ScientISSTMOVEDataset(root=data_root, window_sec=1, stride_sec=10, none_policy="ignore")
    dataset = filter_labels(dataset, remove_labels=["sprint", "jumps"])
    train_subjects = dataset.subjects[:int(0.8 * len(dataset.subjects))]
    val_subjects = dataset.subjects[int(0.8 * len(dataset.subjects)):]

    model = build_former(num_modal=14, num_classes=len(dataset.label_map), model_dim=32, return_mem=True)
    agent = SensorGatingAgent(num_modalities=14, feature_dim=32)

    # pass ddp config into trainer
    ddp_config = {}
    if is_distributed:
        ddp_config = {
            "ddp_rank": local_rank,
            "ddp_world_size": world_size,
            "ddp_port": args.port
        }

    trainer = create_agent_trainer_from_dataset(
        scientisst_dataset=dataset,
        train_subject_ids=train_subjects,
        val_subject_ids=val_subjects,
        model=model,
        agent=agent,
        trainer_config={
            "batch_size": args.batch_size,
            "bptt_steps": args.bptt_steps,
            "ce_weight": args.ce_weight,
            "gating_weight": args.gating_weight,
            "learning_rate": 1e-3,
            "use_amp": True,
            "grad_clip_norm": 1.0
        },
        ddp_config=ddp_config
    )

    trainer.train(num_epochs=args.num_epochs)
    trainer.cleanup()