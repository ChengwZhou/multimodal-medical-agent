# =====================================
# File: sequential_trainer.py
# Sequential trainer for ScientISSTMOVE dataset
# Supports DDP, mixed precision, and feature dependency
# =====================================
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, is_initialized
from torch.optim.lr_scheduler import OneCycleLR

import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
import os
from tqdm import tqdm
from dataclasses import dataclass
import json
import time

import argparse
from models.former import build_former
from dataset.ScientISST_MOVE_loader import ScientISSTMOVEDataset, filter_labels


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_info(msg):
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)

@dataclass
class SequentialBatch:
    """Sequential batch with padding information"""
    sequences: torch.Tensor  # [B, max_seq_len, C, T] - padded sequences
    labels: torch.Tensor  # [B, max_seq_len] - padded labels (-100 for padding)
    seq_lengths: torch.Tensor  # [B] - actual sequence lengths for each subject
    subject_ids: List[str]  # Subject IDs for this batch


class SequentialDataset(Dataset):
    """
    Dataset wrapper that groups data by subject and maintains temporal order
    """

    def __init__(self, scientisst_dataset, subject_ids: Optional[List[str]] = None):
        self.base_dataset = scientisst_dataset
        self.subject_ids = subject_ids or list(scientisst_dataset.subject_to_windows.keys())

        # Group sequences by subject
        self.subject_sequences = {}
        for subject_id in self.subject_ids:
            sequence = scientisst_dataset.get_subject_sequence(subject_id)
            if sequence:  # Only include subjects with data
                self.subject_sequences[subject_id] = sequence

        # Update subject list to only include those with data
        self.subject_ids = list(self.subject_sequences.keys())
        log_info(f"Sequential dataset initialized with {len(self.subject_ids)} subjects")

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Tuple[str, List[Tuple[torch.Tensor, int]]]:
        subject_id = self.subject_ids[idx]
        sequence = self.subject_sequences[subject_id]
        return subject_id, sequence


def collate_sequential_batch(batch: List[Tuple[str, List[Tuple[torch.Tensor, int]]]]) -> SequentialBatch:
    """
    Collate function for sequential batches with padding
    Uses -100 for padding labels (standard ignore_index)
    """
    subject_ids = []
    x_sequences = []
    y_sequences = []
    lengths = []

    for subject_id, sequence in batch:
        subject_ids.append(subject_id)
        lengths.append(len(sequence))

        # Extract x and y from sequence
        x_seq = torch.stack([x for x, y in sequence])  # [seq_len, C, T]
        y_seq = torch.tensor([y for x, y in sequence], dtype=torch.long)  # [seq_len]

        x_sequences.append(x_seq)
        y_sequences.append(y_seq)

    # Pad sequences to same length
    max_len = max(lengths)
    batch_size = len(batch)
    C, T = x_sequences[0].shape[1], x_sequences[0].shape[2]

    # Initialize padded tensors
    sequences = torch.zeros(batch_size, max_len, C, T, dtype=torch.float32)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # Use -100 for padding

    # Fill with actual data
    for i, (x_seq, y_seq, length) in enumerate(zip(x_sequences, y_sequences, lengths)):
        sequences[i, :length] = x_seq
        labels[i, :length] = y_seq

    return SequentialBatch(
        sequences=sequences,
        labels=labels,
        seq_lengths=torch.tensor(lengths, dtype=torch.long),
        subject_ids=subject_ids
    )


def setup_ddp(rank: int, world_size: int, port: str = "12355"):
    """Setup DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


class SequentialTrainer:
    """
    Trainer for sequential real-time prediction with DDP support
    """

    def __init__(
            self,
            model: nn.Module,
            train_dataset: SequentialDataset,
            val_dataset: Optional[SequentialDataset] = None,
            batch_size: int = 8,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            device: str = "auto",
            save_dir: str = "./checkpoints",
            log_interval: int = 10,
            val_interval: int = 100,
            use_amp: bool = True,
            grad_clip_norm: float = 1.0,
            warmup_steps: int = 50,
            # Gradient accumulation
            gradient_accumulation_steps: int = 1,
            # DDP settings
            ddp_rank: Optional[int] = None,
            ddp_world_size: Optional[int] = None,
            ddp_port: str = "12355"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip_norm = grad_clip_norm
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

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

        # Wrap with DDP if needed
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.rank])

        # Setup optimizer with warmup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Setup learning rate scheduler with warmup
        from torch.optim.lr_scheduler import OneCycleLR

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

        # Create save directory (only main process)
        if self.is_main_process:
            os.makedirs(save_dir, exist_ok=True)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_acc = 0.0

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        if self.is_main_process:
            log_info(f"Trainer initialized - Device: {self.device}")
            log_info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            log_info(f"DDP: {self.is_ddp}, Mixed Precision: {self.use_amp}")
            log_info(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
            log_info(f"Effective Batch Size: {batch_size * self.gradient_accumulation_steps * self.world_size}")

    def get_lr_scheduler(self, total_steps: int):
        """Get learning rate scheduler with warmup"""
        return OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=self.warmup_steps / total_steps,
            anneal_strategy='cos'
        )

    def train_step(self, batch: SequentialBatch, accumulation_step: int) -> Dict[str, float]:
        """Single training step with sequential processing and gradient accumulation"""
        self.model.train()

        # Move to device
        sequences = batch.sequences.to(self.device)  # [B, Max_seq_len, C, T]
        labels = batch.labels.to(self.device)  # [B, Max_seq_len]
        # seq_lengths = batch.seq_lengths.to(self.device)  # [B]
        max_seq_len = sequences.size(1)

        total_loss = 0.0
        correct_count = 0
        total_samples = 0

        # Initialize memory/features for each subject in batch
        mem = None  # Will be updated for each window

        # Only zero gradients at the start of accumulation cycle
        if accumulation_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Process each window in sequence
        for i in range(max_seq_len):
            window = sequences[:, i, :, :]  # [B, C, T]
            label = labels[:, i]  # [B]
            # Create mask for valid samples (not padding)
            mask = label != -100

            if not mask.any():
                continue  # Skip if all are padding

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits, mem = self.model(window, mem)  # [B, num_classes], updated_mem
                if mem is not None:
                    mem = mem.detach()

                    # Only compute loss for non-padding samples
                if mask.any():
                    loss = self.criterion(logits[mask], label[mask])
                    # Scale loss by accumulation steps for proper averaging
                    loss = loss / self.gradient_accumulation_steps
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Scale and backward
            if loss.requires_grad and loss.item() > 0:
                self.scaler.scale(loss).backward()

            # Accumulate metrics (unscaled loss for logging)
            with torch.no_grad():
                if mask.any():
                    preds = logits[mask].argmax(dim=1)
                    correct = preds.eq(label[mask]).sum().item()
                    correct_count += correct

                    batch_size = mask.sum().item()
                    # Store unscaled loss for logging
                    total_loss += loss.item() * self.gradient_accumulation_steps * batch_size
                    total_samples += batch_size

        # Only step optimizer at the end of accumulation cycle
        is_last_accumulation_step = (accumulation_step + 1) % self.gradient_accumulation_steps == 0

        if is_last_accumulation_step and total_samples > 0:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update learning rate
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

        # Compute average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_samples": total_samples,
            "lr": self.optimizer.param_groups[0]['lr'],
            "is_optimizer_step": is_last_accumulation_step
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop with sequential processing"""
        if not self.val_dataset:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct_count = 0
        total_samples = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False, disable=not self.is_main_process):
            sequences = batch.sequences.to(self.device)
            labels = batch.labels.to(self.device)
            max_seq_len = sequences.size(1)

            mem = None

            for i in range(max_seq_len):
                window = sequences[:, i, :, :]
                label = labels[:, i]
                mask = label != -100

                if not mask.any():
                    continue

                with autocast(enabled=self.use_amp):
                    logits, mem = self.model(window, mem)

                    if mask.any():
                        loss = self.criterion(logits[mask], label[mask])
                        preds = logits[mask].argmax(dim=1)
                        correct = preds.eq(label[mask]).sum().item()

                        batch_size = mask.sum().item()
                        total_loss += loss.item() * batch_size
                        correct_count += correct
                        total_samples += batch_size

        # Synchronize across processes if DDP
        if self.is_ddp:
            # All-reduce metrics
            metrics_tensor = torch.tensor([total_loss, correct_count, total_samples], device=self.device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_loss, correct_count, total_samples = metrics_tensor.tolist()

        if total_samples == 0:
            return {"val_loss": 0.0, "val_accuracy": 0.0}

        avg_loss = total_loss / total_samples
        avg_accuracy = correct_count / total_samples

        return {"val_loss": avg_loss, "val_accuracy": avg_accuracy}

    def train(self, num_epochs: int = 10):
        """Main training loop with gradient accumulation"""
        if self.is_main_process:
            log_info(f"Starting training for {num_epochs} epochs")
        max_seq_len = int(max(next(iter(self.train_loader)).seq_lengths))
        # Calculate total steps for lr scheduler (accounting for gradient accumulation)
        steps_per_epoch = len(self.train_loader) * max_seq_len // self.gradient_accumulation_steps
        print(steps_per_epoch)
        total_steps = steps_per_epoch * num_epochs
        self.lr_scheduler = self.get_lr_scheduler(total_steps)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Set sampler epoch for DDP
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_losses = []
            epoch_accuracies = []
            epoch_start_time = time.time()

            # Training loop with gradient accumulation
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                        disable=not self.is_main_process)

            accumulation_step = 0
            accumulated_metrics = {"loss": 0.0, "accuracy": 0.0, "total_samples": 0}

            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch, accumulation_step)

                if metrics["total_samples"] > 0:
                    # Accumulate metrics
                    accumulated_metrics["loss"] += metrics["loss"] * metrics["total_samples"]
                    accumulated_metrics["accuracy"] += metrics["accuracy"] * metrics["total_samples"]
                    accumulated_metrics["total_samples"] += metrics["total_samples"]

                    # If optimizer step was taken
                    if metrics["is_optimizer_step"]:
                        # Calculate averaged metrics over accumulation steps
                        if accumulated_metrics["total_samples"] > 0:
                            avg_loss = accumulated_metrics["loss"] / accumulated_metrics["total_samples"]
                            avg_acc = accumulated_metrics["accuracy"] / accumulated_metrics["total_samples"]

                            epoch_losses.append(avg_loss)
                            epoch_accuracies.append(avg_acc)

                            # Update progress bar
                            pbar.set_postfix({
                                "loss": f"{avg_loss:.4f}",
                                "acc": f"{avg_acc:.4f}",
                                "lr": f"{metrics['lr']:.2e}",
                                "step": f"{accumulation_step + 1}/{self.gradient_accumulation_steps}",
                                "samples": accumulated_metrics["total_samples"]
                            })

                            # Logging
                            if self.is_main_process and self.step % self.log_interval == 0:
                                recent_losses = epoch_losses[-self.log_interval:]
                                recent_accs = epoch_accuracies[-self.log_interval:]
                                recent_avg_loss = np.mean(recent_losses)
                                recent_avg_acc = np.mean(recent_accs)
                                log_info(f"Step {self.step}: loss={recent_avg_loss:.4f}, "
                                            f"acc={recent_avg_acc:.4f}, lr={metrics['lr']:.2e}, "
                                            f"eff_bs={accumulated_metrics['total_samples']}")

                            # Validation
                            if self.step % self.val_interval == 0 and self.val_dataset:
                                val_metrics = self.validate()
                                if self.is_main_process:
                                    log_info(f"Step {self.step} Validation: {val_metrics}")

                                    # Save best model
                                    if val_metrics.get("val_accuracy", 0) > self.best_val_acc:
                                        self.best_val_acc = val_metrics["val_accuracy"]
                                        self.save_checkpoint("best_model.pth")
                                        log_info(f"New best model saved with val_acc={self.best_val_acc:.4f}")

                            self.step += 1

                        # Reset accumulated metrics
                        accumulated_metrics = {"loss": 0.0, "accuracy": 0.0, "total_samples": 0}
                        accumulation_step = 0
                    else:
                        accumulation_step += 1

                        # Update progress bar for accumulation steps
                        pbar.set_postfix({
                            "accumulating": f"{accumulation_step + 1}/{self.gradient_accumulation_steps}",
                            "samples": accumulated_metrics["total_samples"]
                        })

            # Handle remaining accumulated gradients at epoch end
            if accumulation_step > 0 and accumulated_metrics["total_samples"] > 0:
                # Force optimizer step for remaining accumulated gradients
                if self.gradient_accumulation_steps > 1:
                    # Unscale and step
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if hasattr(self, 'lr_scheduler'):
                        self.lr_scheduler.step()

                # Log final accumulated metrics
                avg_loss = accumulated_metrics["loss"] / accumulated_metrics["total_samples"]
                avg_acc = accumulated_metrics["accuracy"] / accumulated_metrics["total_samples"]
                epoch_losses.append(avg_loss)
                epoch_accuracies.append(avg_acc)

            # End of epoch summary
            if epoch_losses and self.is_main_process:
                epoch_time = time.time() - epoch_start_time
                epoch_loss = np.mean(epoch_losses)
                epoch_acc = np.mean(epoch_accuracies)
                log_info(f"Epoch {epoch + 1} Summary: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, "
                            f"time={epoch_time:.1f}s, batches={len(epoch_losses)}")

                # Save checkpoint
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")

        if self.is_main_process:
            log_info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint (only main process)"""
        if not self.is_main_process:
            return

        # Get model state dict (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()

        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_val_acc": self.best_val_acc
        }

        if hasattr(self, 'lr_scheduler'):
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state (handle DDP)
        if self.is_ddp:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]

        if "lr_scheduler_state_dict" in checkpoint and hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if self.is_main_process:
            log_info(f"Checkpoint loaded from {filepath}")

    @torch.no_grad()
    def predict_sequence(self, subject_sequence: List[Tuple[torch.Tensor, int]]) -> Dict[str, Any]:
        """
        Predict on a single subject sequence with feature dependency
        Simulates real-time prediction by processing windows sequentially
        """
        self.model.eval()

        predictions = []
        mem = None  # Start with no memory

        for window_x, true_y in subject_sequence:
            # Add batch dimension and move to device
            x = window_x.unsqueeze(0).to(self.device)  # [1, C, T]

            # Predict with memory from previous window
            with autocast(enabled=self.use_amp):
                logits, mem = self.model(x, mem)  # [1, num_classes], updated_mem

            pred_y = logits.argmax(dim=-1).item()
            confidence = F.softmax(logits, dim=-1).max().item()

            predictions.append({
                "prediction": pred_y,
                "true_label": true_y,
                "confidence": confidence
            })

        return {"predictions": predictions}

    def cleanup(self):
        """Cleanup resources"""
        if self.is_ddp:
            cleanup_ddp()


# Convenience function
def create_trainer_from_dataset(
        scientisst_dataset,
        train_subject_ids: List[str],
        val_subject_ids: Optional[List[str]] = None,
        model: Optional[nn.Module] = None,
        trainer_config: Optional[Dict] = None,
        ddp_config: Optional[Dict] = None
) -> SequentialTrainer:
    """
    Convenience function to create trainer from ScientISSTMOVEDataset
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
    trainer = SequentialTrainer(
        model=model,
        train_dataset=train_sequential,
        val_dataset=val_sequential,
        **trainer_config
    )

    return trainer


# DDP training script
def train_ddp(rank: int, world_size: int, train_fn, *args, **kwargs):
    """DDP training wrapper"""
    try:
        kwargs['ddp_rank'] = rank
        kwargs['ddp_world_size'] = world_size
        train_fn(*args, **kwargs)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    # Environment/Parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default="12355")
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--root', type=str, default='/Users/chengweizhou/PycharmProjects/data/scientisst-move-annotated-wearable-multimodal-biosignals-recorded-during-everyday-life-activities-in-naturalistic-environments-1.0.1')
    args = parser.parse_args()

    # detect torchrun env
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    is_distributed = world_size > 1

    # build dataset and model (each process will run this; that's fine)
    data_root = args.root

    dataset = ScientISSTMOVEDataset(root=data_root, window_sec=1, stride_sec=0.5, none_policy="ignore")
    dataset = filter_labels(dataset, remove_labels=["sprint", "jumps"])
    train_subjects = dataset.subjects[:int(0.8 * len(dataset.subjects))]
    val_subjects = dataset.subjects[int(0.8 * len(dataset.subjects)):]

    model = build_former(num_modal=14, num_classes=len(dataset.label_map), model_dim=32, return_mem=True)

    # pass ddp config into trainer
    ddp_config = {}
    if is_distributed:
        ddp_config = {
            "ddp_rank": local_rank,
            "ddp_world_size": world_size,
            "ddp_port": args.port
        }

    trainer = create_trainer_from_dataset(
        scientisst_dataset=dataset,
        train_subject_ids=train_subjects,
        val_subject_ids=val_subjects,
        model=model,
        trainer_config={
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": 1e-3,
            "use_amp": True,
            "grad_clip_norm": 1.0
        },
        ddp_config=ddp_config
    )

    trainer.train(num_epochs=args.num_epochs)
    trainer.cleanup()