# train_ddp_batch.py
# DDP batch-based training for MoveEDF windows
import argparse
import os
import time
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler

from models.cnn import CNN1DModel
from models.cnn_res import CNN1DResidual
from utils.loader_filter_after_seg import MoveEDFWindowDataset, split_by_subject, filter_labels
from utils.metrics import compute_metrics, print_metrics


# -----------------------
# DDP setup
# -----------------------
def ddp_setup(master_addr=None, master_port=None):
    """
    Setup DDP with optional manual address specification
    Args:
        master_addr: Master node address (e.g., "localhost", "192.168.1.100")
        master_port: Master node port (e.g., "12355")
    """
    # Use env:// launched by torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set master address and port if provided
        if master_addr is not None:
            os.environ['MASTER_ADDR'] = master_addr
        if master_port is not None:
            os.environ['MASTER_PORT'] = str(master_port)

        # Use tcp:// init method if address is specified, otherwise use env://
        if master_addr is not None and master_port is not None:
            init_method = f'tcp://{master_addr}:{master_port}'
            print(f"Rank {rank}: Initializing DDP with tcp://{master_addr}:{master_port}")
        else:
            init_method = 'env://'
            print(f"Rank {rank}: Initializing DDP with env://")

        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank}/{world_size} initialized on device cuda:{local_rank}")
        return rank, world_size, local_rank
    else:
        # Single process mode
        print("Single process mode - no DDP")
        return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


# -----------------------
# collate fn: convert list[(x_dict, y)] -> (x_batch [B,C,T], y_batch [B])
# -----------------------
def collate_windows(batch: List):
    xs = []
    ys = []
    for x_dict, y in batch:
        # x_dict entries are torch tensors, shapes:
        # single-channel: [1, T]; c_acc: [3, T]; w_acc: [3, T]
        parts = [
            x_dict['ecg_chest_gel'],
            x_dict['ecg_chest_textile'],
            x_dict['eda_forearm_scientisst'],
            x_dict['eda_wrist_e4'],
            x_dict['ppg_forearm_scientisst'],
            x_dict['ppg_wrist_e4'],
            x_dict['emg_forearm'],
            x_dict['temp_wrist'],
        ]
        # ensure all single-channel parts are [1,T]
        parts = [p if p.dim() == 2 else p.unsqueeze(0) for p in parts]

        # expand accs into 3 separate channels each
        c_acc = x_dict['c_acc']  # [3, T] (or maybe zeros)
        w_acc = x_dict['w_acc']  # [3, T]
        # ensure dims
        if c_acc.dim() == 1:
            c_acc = c_acc.unsqueeze(0)
        if w_acc.dim() == 1:
            w_acc = w_acc.unsqueeze(0)
        # now unbind channels
        c_chs = list(torch.unbind(c_acc, dim=0))  # three [T] -> each [T] -> make [1,T]
        c_chs = [ch.unsqueeze(0) if ch.dim() == 1 else ch for ch in c_chs]
        w_chs = list(torch.unbind(w_acc, dim=0))
        w_chs = [ch.unsqueeze(0) if ch.dim() == 1 else ch for ch in w_chs]

        all_parts = parts + c_chs + w_chs  # list of [1,T] tensors, len=14
        x_cat = torch.cat(all_parts, dim=0)  # [C=14, T]
        xs.append(x_cat)
        ys.append(-1 if y is None else int(y))

    x_batch = torch.stack(xs, dim=0)  # [B, C, T]
    y_batch = torch.tensor(ys, dtype=torch.long)
    return x_batch, y_batch


# -----------------------
# train / eval loops
# -----------------------
def train_epoch(model, loader, optimizer, scaler, device, criterion, epoch, rank, world_size):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct_count = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mask = yb != -1

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Simple accuracy calculation
        preds = logits.argmax(dim=1)
        correct = preds.eq(yb).sum().item() if mask.any() else 0
        correct_count += correct

        batch_size = mask.sum().item() if mask.any() else 0
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # Aggregate across ranks
    if world_size > 1 and total_samples > 0:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        correct_count_tensor = torch.tensor(correct_count, device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_count_tensor, op=dist.ReduceOp.SUM)

        avg_loss = (total_loss_tensor / total_samples_tensor).item() if total_samples_tensor > 0 else 0.0
        avg_acc = (correct_count_tensor / total_samples_tensor).item() if total_samples_tensor > 0 else 0.0
    else:
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_acc = correct_count / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_acc


def eval_epoch(model, loader, device, criterion, epoch, rank, world_size):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mask = yb != -1

            logits = model(xb)
            loss = criterion(logits, yb)

            batch_size = mask.sum().item() if mask.any() else 0
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            if mask.any():
                all_preds.append(preds[mask].detach().cpu())
                all_labels.append(yb[mask].detach().cpu())

    if total_samples == 0:
        return 0.0, 0.0, {}

    # Aggregate across ranks
    if world_size > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / total_samples_tensor).item()
    else:
        avg_loss = total_loss / total_samples

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        metrics = compute_metrics(all_labels, all_preds, average='macro')
        acc = metrics['accuracy']
    else:
        metrics = {}
        acc = 0.0

    return avg_loss, acc, metrics


# -----------------------
# main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--window_sec', type=float, default=1.0)
    parser.add_argument('--stride_sec', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluate every N epochs')

    # DDP address arguments
    parser.add_argument('--master_addr', type=str, default=None,
                        help='Master node address (e.g., localhost, 192.168.1.100)')
    parser.add_argument('--master_port', type=int, default=None,
                        help='Master node port (e.g., 12355)')
    args = parser.parse_args()

    rank, world_size, local_rank = ddp_setup(args.master_addr, args.master_port)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # dataset & split
    dataset = MoveEDFWindowDataset(root=args.data_root, window_sec=args.window_sec, stride_sec=args.stride_sec,
                                   none_policy="extra_class")
    dataset = filter_labels(dataset, remove_labels=["sprint"])
    train_idx, val_idx = split_by_subject(dataset, val_ratio=0.2)

    # # compute class weights
    # num_classes = len(dataset.label_map)
    # train_labels = [dataset[i][1] for i in train_idx if dataset[i][1] is not None]
    # counts = torch.bincount(torch.tensor(train_labels), minlength=num_classes)
    # total = counts.sum().item()
    # weights = total / (num_classes * counts.float().clamp(min=1))
    # weights = [14.303363800048828, 3.314342498779297, 8.003421783447266, 27.840476989746094, 4.785348892211914, 0.17281104624271393, 3.78414249420166, 1.5267006158828735, 0.6440827250480652]
    weights = torch.tensor([40, 5, 20, 60, 20, 0.1, 4, 1.5, 0.7], dtype=torch.float)
    weights = weights.to(torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'))
    if is_main(rank):
        # print("Class counts:", counts.tolist())
        print("Class weights:", weights.tolist())

    # use Subset over flattened indices
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # distributed samplers
    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank,
                                       shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank,
                                     shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_windows,
        drop_last=False
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_windows,
        drop_last=False
    )

    num_classes = len(dataset.label_map)
    model = CNN1DResidual(num_classes=num_classes)
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training (no detailed metrics)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device, criterion, epoch, rank,
                                            world_size)

        # Evaluation every N epochs
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_loss, val_acc, val_metrics = eval_epoch(model, val_loader, device, criterion, epoch, rank, world_size)

            if is_main(rank):
                print(
                    f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f}")
                if val_metrics:
                    print("Validation metrics:")
                    print_metrics(val_metrics)
                print("-" * 50)
        else:
            if is_main(rank):
                print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.3f}")

        # Save checkpoint
        if is_main(rank):
            ckpt_path = os.path.join(args.save_dir, f"latest_ckp.pt")
            to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': to_save,
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_sched.state_dict(),
                'scaler': scaler.state_dict(),
                'label_map': dataset.label_map,
            }, ckpt_path)

        lr_sched.step()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()