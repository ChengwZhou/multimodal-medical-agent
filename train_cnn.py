# =====================================
# File: train_ddp.py
# Patient-sequence aware PyTorch DDP training
# =====================================

import argparse
import time
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler

from models.cnn import CNN1DModel
from utils.ScientISST_MOVE_loader import MoveEDFWindowDataset, split_by_subject
from utils.metrics import compute_metrics, print_metrics


# =======================
# DDP setup
# =======================
def ddp_setup():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world_size
    else:
        return 0, 1


def is_main(rank: int) -> bool:
    return rank == 0


# =======================
# Training one patient sequence
# =======================
def train_patient_sequence(model, seq: List, optimizer, scaler, device, ce):
    """seq: list of (x_dict, y) tuples for one patient in order"""
    model.train()
    h = None
    running_loss, nstep = 0.0, 0
    all_preds, all_labels = [], []

    for x, y in seq:
        # x = {k: v.to(device, non_blocking=True).unsqueeze(0) for k, v in x.items()}  # add batch dim 1
        x = torch.cat([
            x['ecg_chest_gel'],
            x['ecg_chest_textile'],
            x['eda_forearm_scientisst'],
            x['eda_wrist_e4'],
            x['ppg_forearm_scientisst'],
            x['ppg_wrist_e4'],
            x['emg_forearm'],
            x['temp_wrist'],
            x['c_acc'],
            x['w_acc'],
        ], dim=0).unsqueeze(0)
        x = x.to(device)
        y = torch.tensor([y], dtype=torch.long, device=device) if y is not None else torch.tensor([-1], device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = ce(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        nstep += 1

        pred = logits.argmax(dim=1)
        mask = y != -1
        all_preds.append(pred[mask].detach().cpu())
        all_labels.append(y[mask].detach().cpu())

    if nstep == 0:
        return 0.0, 0.0, {}

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(all_labels, all_preds, average="macro")
    acc = metrics["accuracy"]
    return running_loss / nstep, acc, metrics


# =======================
# Evaluation per patient
# =======================
def evaluate_patient(model, seq: List, device, ce):
    model.eval()
    h = None
    running_loss, nstep = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in seq:
            # x = {k: v.to(device, non_blocking=True).unsqueeze(0) for k, v in x.items()}  # add batch dim 1
            x = torch.cat([
                x['ecg_chest_gel'],
                x['ecg_chest_textile'],
                x['eda_forearm_scientisst'],
                x['eda_wrist_e4'],
                x['ppg_forearm_scientisst'],
                x['ppg_wrist_e4'],
                x['emg_forearm'],
                x['temp_wrist'],
                x['c_acc'],
                x['w_acc'],
            ], dim=0).unsqueeze(0)
            x = x.to(device)
            y = torch.tensor([y], dtype=torch.long, device=device) if y is not None else torch.tensor([-1], device=device)

            logits = model(x)
            loss = ce(logits, y)
            running_loss += loss.item()
            nstep += 1

            pred = logits.argmax(dim=1)
            mask = y != -1
            all_preds.append(pred[mask].detach().cpu())
            all_labels.append(y[mask].detach().cpu())

    if nstep == 0:
        return 0.0, 0.0, {}

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(all_labels, all_preds, average="macro")
    acc = metrics["accuracy"]
    return running_loss / nstep, acc, metrics


# =======================
# Main
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--window_sec', type=float, default=3.0)
    parser.add_argument('--stride_sec', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)  # batch_size=1 patient
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    rank, world_size = ddp_setup()
    device = torch.device('cuda', rank % torch.cuda.device_count()) if torch.cuda.is_available() else torch.device('cpu')

    dataset = MoveEDFWindowDataset(root=args.data_root, window_sec=args.window_sec, stride_sec=args.stride_sec, none_policy="extra_class")
    train_idx, val_idx = split_by_subject(dataset, val_ratio=0.2)

    def idx_to_subject_windows(idx_list):
        subj_dict = {}
        for i in idx_list:
            wi = dataset.index[i]
            if wi.subject_id not in subj_dict:
                subj_dict[wi.subject_id] = []
            subj_dict[wi.subject_id].append((dataset[i][0], dataset[i][1]))
        return subj_dict

    train_subj_seq = idx_to_subject_windows(train_idx)
    val_subj_seq = idx_to_subject_windows(val_idx)

    # DDP subject allocation
    train_subj_ids = sorted(train_subj_seq.keys())
    val_subj_ids = sorted(val_subj_seq.keys())
    if world_size > 1:
        train_subj_ids = train_subj_ids[rank::world_size]
        val_subj_ids = val_subj_ids

    # subjects = dataset.subjects
    num_classes = len(dataset.label_map)
    model = CNN1DModel(num_classes=num_classes)
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    os.makedirs(args.save_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_metrics = 0.0, 0.0, {}
        n_subj = 0
        for sid in train_subj_ids:
            seq = dataset.get_subject_sequence(sid)
            l, acc, tr_metrics = train_patient_sequence(model, seq, optimizer, scaler, device, ce)
            train_loss += l
            train_acc += acc
            n_subj += 1
        train_loss /= max(1, n_subj)
        train_acc /= max(1, n_subj)

        # Evaluate
        val_loss, val_acc, val_metrics = 0.0, 0.0, {}
        for sid in val_subj_ids:
            seq = dataset.get_subject_sequence(sid)
            l, acc, va_metrics = evaluate_patient(model, seq, device, ce)
            val_loss += l
            val_acc += acc
        val_loss /= max(1, n_subj)
        val_acc /= max(1, n_subj)

        if is_main(rank):
            print(f"Epoch {ep:03d} | train_loss {train_loss:.4f} acc {train_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f} | {(time.time()-t0):.1f}s")
            # print("\n[Train Metrics]")
            # print_metrics(tr_metrics)

            ckpt_path = os.path.join(args.save_dir, f"epoch{ep:03d}_acc{val_acc:.3f}.pt")
            to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'epoch': ep,
                'model': to_save,
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_sched.state_dict(),
                'scaler': scaler.state_dict(),
            }, ckpt_path)

        lr_sched.step()
    if is_main(rank):
        print("\n[Val Metrics]")
        print_metrics(va_metrics)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
