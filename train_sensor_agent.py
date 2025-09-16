# train_agent_ddp.py
import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

# 你工程里的模块（请确认 import 路径）
from trainer.agent_trainer import SensorAgentTrainer
from models.former import build_former
from models.sensor_gating_agent import SensorGatingAgent
from dataset.ScientISST_MOVE_loader import ScientISSTMOVEDataset, split_by_subject, filter_labels


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_ddp(local_rank: int, backend: str = "nccl"):
    # If single-process (no ddp) we don't init the process-group.
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        world_size = int(os.environ['WORLD_SIZE'])
        # Use env:// if torchrun provides env vars
        dist.init_process_group(backend=backend, init_method='env://')
        torch.cuda.set_device(local_rank)
        return True
    else:
        # fallback: if user still passed local_rank and torchrun not used
        if local_rank is not None and torch.cuda.is_available() and local_rank >= 0:
            # not initializing group, but set device for single-process GPU usage
            torch.cuda.set_device(local_rank)
        return False

def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['siscientisst', 'mhealth'], default='mhealth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--force_data_prep', action='store_true')
    parser.add_argument('--window_sec', type=float, default=1.0)
    parser.add_argument('--stride_sec', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_energy', type=float, default=0.3)
    parser.add_argument('--lambda_trigger', type=float, default=0.1)
    parser.add_argument('--fn_cap', type=float, default=0.05)
    return parser.parse_args()

def main():
    args = parse_args()

    local_rank = args.local_rank
    use_cuda = torch.cuda.is_available()
    ddp_active = False
    try:
        ddp_active = setup_ddp(local_rank, backend=args.backend)
    except Exception as e:
        print("Warning: DDP setup failed or not used:", e)
        ddp_active = False

    device = torch.device(f'cuda:{local_rank}' if use_cuda else 'cpu')
    if is_main_process():
        print("Using device:", device)
        print("DDP active:", ddp_active)

    # ----------------------------
    # Prepare dataset and samplers
    # ----------------------------
    if args.dataset == 'siscientisst':
        dataset = ScientISSTMOVEDataset(root=args.data_root, window_sec=args.window_sec, stride_sec=args.stride_sec,
                                        none_policy="ignore", force_reprocess=args.force_data_prep)
        dataset = filter_labels(dataset, remove_labels=["sprint", "jumps"])
        train_idx, val_idx = split_by_subject(dataset, val_ratio=0.2)
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        num_modal = 14
        num_classes = len(dataset.label_map)
    # else:
    #     # 默认 mHealth
    #     train_subset = MHealthDataset(args.data_root, subjects=[i for i in range(1, 9)], time_steps=100, step=50,
    #                                   balance=False, majority_n=500)
    #     val_subset = MHealthDataset(args.data_root, subjects=[9, 10], time_steps=100, step=50, balance=False,
    #                                 majority_n=120)
    #     num_modal = 12
    #     num_classes = 12

    # Distributed Samplers when DDP active
    if ddp_active:
        train_sampler = DistributedSampler(train_subset, shuffle=True)
        val_sampler = DistributedSampler(val_subset, shuffle=False)
        per_proc_batch = max(1, args.batch_size // dist.get_world_size())
    else:
        train_sampler = None
        val_sampler = None
        per_proc_batch = args.batch_size

    train_loader = DataLoader(train_subset, batch_size=per_proc_batch, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=per_proc_batch, shuffle=False,
                            sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    if is_main_process():
        print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
        print("Batch size per process:", per_proc_batch)

    # ----------------------------
    # Build base model and agent
    # ----------------------------
    base_model = build_former(num_modal=14, num_classes=num_classes, model_dim=32)
    agent = SensorGatingAgent(num_modalities=14, feature_dim=256, hidden_dim=128)

    # Move to device
    base_model.to(device)
    agent.to(device)

    # Optionally wrap in DDP (wrap only if multi-process GPU)
    if ddp_active and use_cuda:
        base_model = torch.nn.parallel.DistributedDataParallel(base_model, device_ids=[local_rank], find_unused_parameters=True)
        agent = torch.nn.parallel.DistributedDataParallel(agent, device_ids=[local_rank], find_unused_parameters=True)

    # ----------------------------
    # Build trainer (ensure trainer uses device)
    # ----------------------------
    trainer = SensorAgentTrainer(
        agent=agent,
        base_model=base_model,
        lambda_energy=args.lambda_energy,
        lambda_trigger=args.lambda_trigger,
        fn_cap=args.fn_cap,
        device=device  # pass device so trainer can put tensors to right device
    )

    logs = trainer.train_episode(train_loader, num_episodes=args.epochs, lr=args.lr)  # adapt signature if needed

    if is_main_process():
        print(f"Epoch {epoch+1} finished. logs summary (example): {logs}")

    # optional evaluation every few epochs
    if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
        if is_main_process():
            print("Running evaluation on validation set...")
        metrics = trainer.evaluate(val_loader)
        if is_main_process():
            print("Validation metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

    # after training
    if is_main_process():
        print("Training completed.")
        # print learned thresholds (if trainer/agent expose them)
        try:
            # if wrapped by DDP, access .module
            ag = agent.module if isinstance(agent, torch.nn.parallel.DistributedDataParallel) else agent
            print("Threshold Up:", ag.threshold_up.detach().cpu().numpy())
            print("Threshold Down:", ag.threshold_down.detach().cpu().numpy())
            print("Conf Exit Threshold:", ag.conf_exit_threshold.detach().cpu().numpy())
        except Exception as e:
            print("Cannot print thresholds (attribute missing or DDP wrapped):", e)

    # cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()
