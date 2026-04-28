"""
trainer/sigma_delta_joint_trainer.py  —  Joint trainer (isolated, new)

Trains the sigma-delta AdaptiveSensingMultimodalTransformer and the
BeliefStatePOMDPAgent jointly.  Completely isolated from the old
agent_trainer.py; nothing is imported from it.

Architecture overview
---------------------
Window t-1  ──►  SigmaDeltaModel  ──► memory o_{t-1}  ──►  POMDPAgent
                                                               │
                                   gate a_t ◄─────────────────┘
                                       │
Window t   ──► gate(input) ──►  SigmaDeltaModel ──► memory o_t ──► logits_t
                │
            Lagrangian penalty  μ · active_ratio
            Policy gradient     −(R−V) · log π(a_t|b_t)
            CE task loss        CE(logits_t, y_t)

Training signal flow
--------------------
  L_total  =  λ_ce · CE
            + λ_pred · predictive_MSE      ← optional self-supervised auxiliary

  Agent is trained via STE: the CE gradient flows through p_soft (Gumbel-sigmoid)
  directly into GRU belief state and policy head — no REINFORCE, no value function.
  Patch-level sparsity is controlled by the learnable threshold in AdaptiveSensingModule.

BPTT
----
  Sequences are split into chunks of `bptt_steps` windows.
  Gradients are computed through the chunk; memory is detached at
  chunk boundaries (truncated BPTT).
  Belief state is carried across chunks *with detach* to prevent
  unlimited gradient accumulation.

Usage
-----
  python trainer/sigma_delta_joint_trainer.py \\
      --dataset mhealth --root /path/to/data \\
      --model_lr 1e-3 --agent_lr 1e-3 \\
      --bptt_steps 8 --num_epochs 50
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.amp import autocast as _autocast_ctx
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def autocast(enabled=True):
    return _autocast_ctx("cuda", enabled=enabled)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pomdp_sensor_agent import AgentOutput, BeliefStatePOMDPAgent
from models.pomdp_device_agent import BeliefStatePOMDPDeviceAgent
from utils.metrics import compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_ddp(rank: int, world_size: int, port: str = "12355"):
    """Initialise NCCL process group for DDP."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def _log(msg: str):
    """Log only from rank 0 (or non-distributed runs)."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        log.info(msg)


def _detach_mem(mem: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return mem.detach() if mem is not None else None


# ---------------------------------------------------------------------------
# SigmaDeltaJointTrainer
# ---------------------------------------------------------------------------

class SigmaDeltaJointTrainer:
    """
    Joint trainer for sigma-delta sensing model + POMDP gating agent.

    Key differences from agent_trainer.py
    --------------------------------------
    1. Uses BeliefStatePOMDPAgent (GRU belief state) not SensorGatingAgent (MLP)
    2. Sigma-delta thresholds are trained jointly via Lagrangian sparsity penalty
    3. Agent trained end-to-end via STE through the CE loss
    4. Value-function baseline reduces PG gradient variance
    5. Regret is tracked and logged each epoch
    6. Skip patterns (active_ratios per modality) are saved each epoch
    7. Temperature annealing schedule is respected by both model and agent
    """

    def __init__(
        self,
        model: nn.Module,
        agent: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        # Optimisers
        model_lr: float = 1e-3,
        agent_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # Training
        bptt_steps: int = 8,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        num_epochs: int = 50,
        # Loss weights
        ce_weight: float = 1.0,
        agent_sparsity_weight: float = 0.0,  # L1 penalty on p_soft.mean() — constrains agent_ratio
        predictive_weight: float = 0.0,      # optional self-supervised auxiliary
        # Misc
        save_dir: str = "./checkpoints_joint",
        log_interval: int = 10,
        device: str = "auto",
        use_memory: bool = True,        # carry memory across windows
        mem_cache_len: int = 10,        # rolling-mean cache length
        channels_per_device: Optional[List[int]] = None,  # device-wise gate expansion
        only_predictive: bool = False,  # run agent but don't apply gate mask to input
        # Ratio constraint (ablation-friendly)
        target_agent_ratio: Optional[float] = None,   # None = use L1 penalty instead
        # Predictive auxiliary loss
        predictive_lr: float = 1e-3,   # separate LR for predictive MLP
        # Ensemble diversity loss (only active when agent uses uncertainty_type='ensemble')
        agent_diversity_weight: float = 0.0,   # 0 = disabled; e.g. 0.01 prevents head collapse
        # DDP
        ddp_rank: Optional[int] = None,
        ddp_world_size: Optional[int] = None,
    ):
        # ---- DDP / device -----------------------------------------------------
        self.is_ddp    = ddp_rank is not None
        self.rank      = ddp_rank      if ddp_rank      is not None else 0
        self.world_size= ddp_world_size if ddp_world_size is not None else 1
        self.is_main   = self.rank == 0   # only rank-0 logs / saves

        if self.is_ddp:
            self.device = torch.device(f"cuda:{self.rank}")
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model  = model.to(self.device)
        self.agent  = agent.to(self.device)

        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
            self.agent = DDP(self.agent, device_ids=[self.rank], find_unused_parameters=True)

        # ---- Data loaders (rebuild with DistributedSampler for DDP) ----------
        if self.is_ddp:
            _cn  = train_loader.collate_fn
            _nw  = train_loader.num_workers
            _bs  = train_loader.batch_size
            self._train_sampler = DistributedSampler(
                train_loader.dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True,
            )
            self.train_loader = DataLoader(
                train_loader.dataset, batch_size=_bs,
                sampler=self._train_sampler, collate_fn=_cn,
                num_workers=_nw, pin_memory=True,
            )
            if val_loader is not None:
                _vcn = val_loader.collate_fn
                _vnw = val_loader.num_workers
                _vbs = val_loader.batch_size
                self._val_sampler = DistributedSampler(
                    val_loader.dataset, num_replicas=self.world_size,
                    rank=self.rank, shuffle=False,
                )
                self.val_loader = DataLoader(
                    val_loader.dataset, batch_size=_vbs,
                    sampler=self._val_sampler, collate_fn=_vcn,
                    num_workers=_vnw, pin_memory=True,
                )
            else:
                self.val_loader = None
                self._val_sampler = None
        else:
            self.train_loader    = train_loader
            self.val_loader      = val_loader
            self._train_sampler  = None
            self._val_sampler    = None

        # ---- Optimisers -------------------------------------------------------
        self.model_opt = torch.optim.AdamW(
            model.parameters(), lr=model_lr, weight_decay=weight_decay
        )
        self.agent_opt = torch.optim.AdamW(
            agent.parameters(), lr=agent_lr, weight_decay=weight_decay
        )

        # ---- Predictive auxiliary MLP  (belief → next obs) --------------------
        # Trained only when predictive_weight > 0.
        # Input : belief state  [B, agent_hidden]
        # Output: predicted next model memory, mean-pooled  [B, model_dim]
        # Gradient flows belief → GRU, training the recurrent state to be
        # predictive of future observations (decoupled from CE task loss).
        if predictive_weight > 0:
            obs_dim    = self._agent.obs_dim      # model_dim (memory dim)
            hidden_dim = self._agent.hidden_dim   # GRU hidden dim
            self.predictive_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, obs_dim),
            ).to(self.device)
            if self.is_ddp:
                self.predictive_mlp = DDP(self.predictive_mlp, device_ids=[self.rank], find_unused_parameters=True)
            self.predictive_opt = torch.optim.AdamW(
                self.predictive_mlp.parameters(),
                lr=predictive_lr, weight_decay=weight_decay,
            )
        else:
            self.predictive_mlp = None
            self.predictive_opt = None

        # ---- LR schedulers (cosine, one-per-epoch) ----------------------------
        self.model_sched = CosineAnnealingLR(self.model_opt, T_max=num_epochs, eta_min=1e-6)
        self.agent_sched = CosineAnnealingLR(self.agent_opt, T_max=num_epochs, eta_min=1e-6)
        if self.predictive_opt is not None:
            self.pred_sched = CosineAnnealingLR(self.predictive_opt, T_max=num_epochs, eta_min=1e-6)
        else:
            self.pred_sched = None

        # ---- AMP --------------------------------------------------------------
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler  = GradScaler("cuda", enabled=self.use_amp)

        # ---- Regret tracker ---------------------------------------------------
        # ---- Hyperparams -------------------------------------------------------
        self.bptt_steps       = bptt_steps
        self.grad_clip        = grad_clip
        self.num_epochs       = num_epochs
        self.ce_weight             = ce_weight
        self.agent_sparsity_weight = agent_sparsity_weight
        self.predictive_weight     = predictive_weight
        self.log_interval       = log_interval
        self.use_memory         = use_memory
        self.mem_cache_len      = mem_cache_len
        # Device-wise: list of channel counts per device (sorted by device_idx).
        # None means sensor-wise (uniform channels per sensor).
        self.channels_per_device   = channels_per_device
        self.only_predictive       = only_predictive
        self.target_agent_ratio    = target_agent_ratio
        self.agent_diversity_weight = agent_diversity_weight
        self.save_dir              = save_dir

        # ---- Criterion --------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # ---- State ------------------------------------------------------------
        self.epoch        = 0
        self.global_step  = 0
        self.best_val_acc = 0.0
        self._skip_history: List[Dict] = []   # per-epoch active_ratio per modality

        os.makedirs(save_dir, exist_ok=True)
        _log(f"SigmaDeltaJointTrainer on {self.device}")
        _log(f"Model params      : {sum(p.numel() for p in model.parameters()):,}")
        _log(f"Agent params      : {sum(p.numel() for p in agent.parameters()):,}")
        if self.predictive_mlp is not None:
            _log(f"Predictive params : {sum(p.numel() for p in self.predictive_mlp.parameters()):,}")

    # ==========================================================================
    # DDP-safe accessors
    # ==========================================================================

    @property
    def _model(self) -> nn.Module:
        """Unwrap DDP to access raw model attributes."""
        return self.model.module if self.is_ddp else self.model

    @property
    def _agent(self) -> nn.Module:
        """Unwrap DDP to access raw agent attributes."""
        return self.agent.module if self.is_ddp else self.agent

    @property
    def _pred_mlp(self) -> Optional[nn.Module]:
        """Unwrap DDP to access raw predictive MLP attributes."""
        if self.predictive_mlp is None:
            return None
        return self.predictive_mlp.module if self.is_ddp else self.predictive_mlp

    # ==========================================================================
    # Core training step  (one BPTT chunk of `bptt_steps` windows)
    # ==========================================================================

    def _train_chunk(
        self,
        sequences:    torch.Tensor,    # [B, max_seq_len, C, T]
        labels:       torch.Tensor,    # [B, max_seq_len]
        start:        int,
        end:          int,
        mem:          Optional[torch.Tensor],
        mem_cache:    List[torch.Tensor],
        belief:       torch.Tensor,    # [B, H]
        sensor_hist:  torch.Tensor,    # [B, T_h, M]
    ) -> Tuple[Dict[str, float], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Process one BPTT chunk.  Returns updated (mem, belief, sensor_hist).
        Agent is trained via STE: CE gradient flows through p_soft into agent params.

        BPTT design (aligned with agent_trainer.py):
        - Agent is NOT called on the first window of each chunk (no prev mem yet).
        - prev_mem_for_agent is NOT detached within the chunk so that CE gradient
          from window i+1 flows back through the agent into window i's model output.
        - chunk_loss accumulates all per-window losses; a single backward() is
          called at the end of the chunk (true truncated BPTT).
        """
        B, _, C, T = sequences.shape
        M = self._agent.num_sensors   # use _agent to safely unwrap DDP

        self.model_opt.zero_grad(set_to_none=True)
        self.agent_opt.zero_grad(set_to_none=True)
        if self.predictive_opt is not None:
            self.predictive_opt.zero_grad(set_to_none=True)

        metrics = defaultdict(float)
        n_valid  = 0
        chunk_loss: Optional[torch.Tensor] = None  # accumulated across windows

        # ---- Forward through chunk --------------------------------------------
        # mem is already detached at the chunk boundary by the caller.
        prev_mem_for_agent = mem   # memory from BEFORE this chunk starts

        for i in range(start, end):
            window = sequences[:, i, :, :]   # [B, C, T]
            label  = labels[:, i]             # [B]
            valid  = label != -100            # [B] bool mask

            # Reset cross-window state for independent-sample datasets
            if not self.use_memory:
                mem                = None
                prev_mem_for_agent = None

            # ---- Agent decision (uses memory from *previous* window) ----------
            # Skip the first window of the chunk: no meaningful prev mem yet.
            # This matches agent_trainer.py which gates only when i > start_idx.
            if prev_mem_for_agent is not None and i > start:
                with autocast(enabled=self.use_amp):
                    agent_out: AgentOutput = self.agent(
                        observation          = prev_mem_for_agent,
                        belief               = belief,
                        sensor_history       = sensor_hist,
                        use_straight_through = True,   # STE: gradient flows via p_soft
                    )

                belief = agent_out.belief   # NOT detached: gradient flows through GRU within chunk
                current_belief_for_pred = belief   # used for predictive auxiliary loss below

                # STE gate: forward = p_hard (discrete), backward through p_soft.
                # CE loss gradient flows directly into agent params — no REINFORCE.
                if self.channels_per_device is not None:
                    # Device-wise: each gate index covers a variable number of channels
                    gate_parts = []
                    for dev_i, n_ch in enumerate(self.channels_per_device):
                        g = agent_out.p_st[:, dev_i:dev_i+1]          # [B, 1]
                        gate_parts.append(g.unsqueeze(-1).expand(-1, n_ch, T))  # [B, n_ch, T]
                    gate_expanded = torch.cat(gate_parts, dim=1)       # [B, C, T]
                else:
                    # Sensor-wise: uniform channels per sensor
                    channels_per_sensor = C // M
                    gate_expanded = (
                        agent_out.p_st
                        .unsqueeze(-1).unsqueeze(-1)
                        .expand(-1, -1, channels_per_sensor, T)
                        .reshape(B, C, T)
                    )
                # only_predictive: agent runs (gradient flows) but mask is NOT applied.
                # Useful for pretraining the agent without hurting model performance.
                if self.only_predictive:
                    window_gated = window
                else:
                    window_gated = window * gate_expanded

                # Update sensor history (rolling, detached)
                sensor_hist = torch.cat(
                    [sensor_hist[:, 1:, :], agent_out.p_hard.unsqueeze(1).detach()], dim=1
                ).detach()

                agent_p_soft = agent_out.p_soft   # keep ref for sparsity penalty below
                metrics["sensor_usage"] += agent_p_soft.mean().item()

                # Log effective τ (adaptive when uncertainty is available)
                with torch.no_grad():
                    tau_base = self._agent.get_temperature()
                    if agent_out.uncertainty is not None:
                        tau_eff = tau_base * (1.0 + self._agent.uncertainty_scale * agent_out.uncertainty)
                        metrics["mean_tau"] += tau_eff.mean().item()
                        metrics["mean_uncertainty"] += agent_out.uncertainty.mean().item()
                    else:
                        metrics["mean_tau"] += tau_base.item()
            else:
                agent_p_soft = None
                current_belief_for_pred = None
                window_gated = window   # no gating yet (cold start / first window of chunk)

            # ---- Model forward -----------------------------------------------
            context = None
            if mem_cache:
                context = torch.stack(mem_cache).mean(0)

            with autocast(enabled=self.use_amp):
                out = self.model(window_gated, history=context)

            if isinstance(out, tuple):
                logits, mem_new = out[0], out[1]
                sinfo = out[2] if len(out) > 2 else None
            else:
                logits  = out
                mem_new = None
                sinfo   = None

            # Rolling memory cache — always detached (used only as history context)
            if mem_new is not None:
                mem_cache.append(mem_new.detach())
                if len(mem_cache) > self.mem_cache_len:
                    mem_cache.pop(0)

            # Do NOT detach prev_mem_for_agent within the chunk.
            # This lets the CE gradient from window i+1's agent call flow back
            # into window i's model output (BPTT through the agent observation).
            prev_mem_for_agent = mem_new

            # ---- Predictive auxiliary loss -----------------------------------
            # Train: belief_t → MLP → predicted mem_{t} (mean-pooled over heads).
            # Gradient flows belief → GRU, making the recurrent state predictive
            # of future model observations independently of the CE task loss.
            if (
                self.predictive_mlp is not None
                and current_belief_for_pred is not None
                and mem_new is not None
                and valid.any()
            ):
                with autocast(enabled=self.use_amp):
                    pred_target = mem_new.mean(dim=1).detach()   # [B, model_dim]
                    pred_out    = self.predictive_mlp(current_belief_for_pred)  # [B, model_dim]
                    pred_loss   = F.mse_loss(pred_out[valid], pred_target[valid])
                metrics["pred_loss"] += pred_loss.item()
            else:
                pred_loss = None

            # ---- CE loss -----------------------------------------------------
            if valid.any():
                with autocast(enabled=self.use_amp):
                    ce = self.criterion(logits[valid], label[valid])
                metrics["ce_loss"] += ce.item()
                n_valid += valid.sum().item()

                with torch.no_grad():
                    preds   = logits[valid].argmax(dim=-1)
                    correct = preds.eq(label[valid]).sum().item()
                metrics["correct"] += correct

                # Track sigma-delta active ratio
                # sigma_former_sensor uses "sparsity_losses"; sigma_former_device uses "active_ratios"
                _ar_list = (sinfo or {}).get("sparsity_losses") or (sinfo or {}).get("active_ratios")
                if _ar_list:
                    ar = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v)
                                      for v in _ar_list]).mean()
                    metrics["active_ratio"] += ar.item()

                # Sparsity penalty on agent gate
                window_loss = self.ce_weight * ce
                if agent_p_soft is not None:
                    cur_ratio = agent_p_soft.mean()
                    if self.target_agent_ratio is not None:
                        # Squared-error constraint: pushes ratio toward target from both sides.
                        # Gradient: 2·w·(ratio - target)/M  →  negative when ratio>target (pushes down),
                        #           positive when ratio<target (pushes up).
                        # More reliable than Lagrangian because it acts immediately every step.
                        ratio_penalty = self.agent_sparsity_weight * (cur_ratio - self.target_agent_ratio).pow(2)
                        window_loss = window_loss + ratio_penalty
                        metrics["sparsity_loss"] += ratio_penalty.item()
                    elif self.agent_sparsity_weight > 0:
                        # Original fixed L1 penalty (one-sided, only pushes ratio down)
                        window_loss = window_loss + self.agent_sparsity_weight * cur_ratio
                        metrics["sparsity_loss"] += cur_ratio.item()

                # Add predictive auxiliary loss to window_loss
                if pred_loss is not None:
                    window_loss = window_loss + self.predictive_weight * pred_loss

                # Ensemble diversity loss: only when agent was actually called this
                # window (agent_p_soft is not None).  Must NOT run when i == start
                # (agent skipped) because _last_logits_k would still hold the freed
                # graph from the previous chunk → "backward through freed graph" error.
                if self.agent_diversity_weight > 0 and agent_p_soft is not None:
                    div_loss = self._agent.ensemble_diversity_loss()
                    if div_loss is not None:
                        window_loss = window_loss + self.agent_diversity_weight * div_loss
                        # diversity metric: 1 - cosine_sim ∈ [0, 2]; higher = more diverse heads
                        metrics["ensemble_diversity"] += (1.0 - div_loss.item())

                # Accumulate into chunk loss — single backward at end of chunk
                chunk_loss = window_loss if chunk_loss is None else chunk_loss + window_loss

            mem = mem_new

        # ---- Single backward for entire chunk (BPTT through agent-model chain) --
        if chunk_loss is not None:
            self.scaler.scale(chunk_loss).backward()

            # ---- Gradient clipping + optimiser step ----------------------------
            # scaler.unscale_/step must only be called for optimisers whose params
            # actually received scaled gradients during backward.
            # We check dynamically after backward because:
            #   - bptt_steps=1  → agent never called (i > start always False)
            #   - only_predictive=True + agent_sparsity_weight=0 → agent called
            #     but gate not applied to loss, so agent params have no grads
            self.scaler.unscale_(self.model_opt)
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.model_opt)

            # Check whether any agent parameter actually received a gradient.
            agent_has_grad = any(
                p.grad is not None for p in self.agent.parameters()
            )
            if agent_has_grad:
                self.scaler.unscale_(self.agent_opt)
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
                self.scaler.step(self.agent_opt)

            if self.predictive_opt is not None:
                pred_has_grad = any(
                    p.grad is not None for p in self.predictive_mlp.parameters()
                )
                if pred_has_grad:
                    self.scaler.unscale_(self.predictive_opt)
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.predictive_mlp.parameters(), self.grad_clip)
                    self.scaler.step(self.predictive_opt)

            self.scaler.update()

        # ---- Normalise metrics -------------------------------------------------
        denom = max(end - start, 1)
        for k in ("ce_loss", "active_ratio", "sensor_usage", "mean_tau", "mean_uncertainty"):
            metrics[k] /= denom
        if n_valid > 0:
            metrics["accuracy"] = metrics["correct"] / n_valid

        return dict(metrics), mem, belief, sensor_hist

    # ==========================================================================
    # Training epoch  (sequential / BPTT mode)
    # ==========================================================================

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.agent.train()
        if self.predictive_mlp is not None:
            self.predictive_mlp.train()

        # DDP: set epoch on sampler so each rank gets a different shuffle
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self.epoch)

        epoch_metrics: Dict[str, List[float]] = defaultdict(list)
        batch_idx = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False,
                          disable=not self.is_main):
            # Support both SequentialBatch and plain (x, y) tuples
            if hasattr(batch, "sequences"):
                sequences  = batch.sequences.to(self.device)   # [B, S, C, T]
                labels     = batch.labels.to(self.device)       # [B, S]
                max_seq    = sequences.shape[1]
            else:
                # Plain batch: treat as single-window sequences
                xb, yb = batch
                xb     = xb.to(self.device)                     # [B, C, T]
                yb     = yb.to(self.device)                     # [B]
                sequences  = xb.unsqueeze(1)                    # [B, 1, C, T]
                labels     = yb.unsqueeze(1)                    # [B, 1]
                max_seq = 1

            # print(sequences.size())  #[1, 631, 12, 100]
            B = sequences.shape[0]
            M = self._agent.num_sensors

            # Initialise per-batch state
            mem         = None
            mem_cache:  List[torch.Tensor] = []
            belief      = self._agent.init_belief(B, self.device)
            sensor_hist = self._agent.init_sensor_history(B, self.device)

            # BPTT chunks
            num_chunks = max(1, (max_seq + self.bptt_steps - 1) // self.bptt_steps)

            for chunk in range(num_chunks):
                start = chunk * self.bptt_steps
                end   = min(start + self.bptt_steps, max_seq)
                if end <= start:
                    continue

                chunk_metrics, mem, belief, sensor_hist = self._train_chunk(
                    sequences, labels, start, end,
                    mem, mem_cache, belief, sensor_hist,
                )
                # Detach memory at chunk boundary
                mem    = _detach_mem(mem)
                belief = belief.detach()

                for k, v in chunk_metrics.items():
                    epoch_metrics[k].append(v)

                self.global_step += 1

            if self.is_main and batch_idx % self.log_interval == 0:
                _tau_val  = np.mean(epoch_metrics.get("mean_tau",        [self._agent.get_temperature().item()]))
                _unc_val  = np.mean(epoch_metrics.get("mean_uncertainty", [0.0]))
                _unc_str  = f"  unc={_unc_val:.4f}" if self._agent.uncertainty_type != "none" else ""
                _log(
                    f"  step {batch_idx:4d}  "
                    f"ce={np.mean(epoch_metrics.get('ce_loss', [0])):.4f}  "
                    f"acc={np.mean(epoch_metrics.get('accuracy', [0])):.3f}  "
                    f"sd_ratio={np.mean(epoch_metrics.get('active_ratio', [0])):.2%}  "
                    f"agent_ratio={np.mean(epoch_metrics.get('sensor_usage', [0])):.2%}  "
                    f"τ={_tau_val:.3f}{_unc_str}"
                )
            batch_idx += 1

        # LR schedulers
        self.model_sched.step()
        self.agent_sched.step()
        if self.pred_sched is not None:
            self.pred_sched.step()

        return {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

    # ==========================================================================
    # Validation
    # ==========================================================================

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.agent.eval()
        if self.predictive_mlp is not None:
            self.predictive_mlp.eval()

        all_preds, all_labels_list = [], []
        total_loss  = 0.0
        total_n     = 0
        total_ratio = 0.0
        ratio_steps = 0

        for batch in tqdm(self.val_loader, desc="Val", leave=False,
                          disable=not self.is_main):
            if hasattr(batch, "sequences"):
                sequences = batch.sequences.to(self.device)
                labels    = batch.labels.to(self.device)
                max_seq   = sequences.shape[1]
            else:
                xb, yb = batch
                sequences = xb.to(self.device).unsqueeze(1)
                labels    = yb.to(self.device).unsqueeze(1)
                max_seq   = 1

            B = sequences.shape[0]
            mem         = None
            mem_cache:  List[torch.Tensor] = []
            belief      = self._agent.init_belief(B, self.device)
            sensor_hist = self._agent.init_sensor_history(B, self.device)

            for i in range(max_seq):
                window = sequences[:, i, :, :]
                label  = labels[:, i]
                valid  = label != -100

                if not self.use_memory:
                    mem = None

                window_gated = window
                if mem is not None:
                    agent_out = self.agent(
                        mem, belief, sensor_hist, use_straight_through=False
                    )
                    belief = agent_out.belief
                    sensor_hist = torch.cat(
                        [sensor_hist[:, 1:, :], agent_out.p_hard.unsqueeze(1)], dim=1
                    )
                    T_w = window.shape[-1]
                    if self.channels_per_device is not None:
                        gate_parts = []
                        for dev_i, n_ch in enumerate(self.channels_per_device):
                            g = agent_out.p_soft[:, dev_i:dev_i+1]
                            gate_parts.append(g.unsqueeze(-1).expand(-1, n_ch, T_w))
                        gate = torch.cat(gate_parts, dim=1)
                    else:
                        C = window.shape[1]
                        M = self.agent.num_sensors
                        gate = (
                            agent_out.p_soft
                            .unsqueeze(-1).unsqueeze(-1)
                            .expand(-1, -1, C // M, T_w)
                            .reshape(B, C, T_w)
                        )
                    if self.only_predictive:
                        window_gated = window
                    else:
                        window_gated = window * gate
                    total_ratio += agent_out.p_soft.mean().item()
                    ratio_steps += 1

                context = torch.stack(mem_cache).mean(0) if mem_cache else None
                out = self.model(window_gated, history=context)

                if isinstance(out, tuple):
                    logits, mem_new = out[0], out[1]
                    sinfo = out[2] if len(out) > 2 else None
                else:
                    logits  = out
                    mem_new = None
                    sinfo   = None

                if mem_new is not None:
                    mem_cache.append(mem_new.detach())
                    if len(mem_cache) > self.mem_cache_len:
                        mem_cache.pop(0)
                mem = mem_new

                if valid.any():
                    loss = self.criterion(logits[valid], label[valid])
                    total_loss += loss.item() * valid.sum().item()
                    total_n    += valid.sum().item()
                    preds = logits[valid].argmax(dim=-1)
                    all_preds.append(preds.cpu())
                    all_labels_list.append(label[valid].cpu())

        # ---- DDP: synchronise scalars and predictions across ranks ----------
        if self.is_ddp:
            stats = torch.tensor(
                [total_loss, total_n, total_ratio, ratio_steps],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_n, total_ratio, ratio_steps = stats.tolist()

            # Gather variable-length predictions from all ranks
            local_preds  = torch.cat(all_preds).numpy()  if all_preds        else np.array([], dtype=np.int64)
            local_labels = torch.cat(all_labels_list).numpy() if all_labels_list else np.array([], dtype=np.int64)
            gathered_preds  = [None] * self.world_size
            gathered_labels = [None] * self.world_size
            dist.all_gather_object(gathered_preds,  local_preds)
            dist.all_gather_object(gathered_labels, local_labels)
            all_preds_t  = np.concatenate(gathered_preds)
            all_labels_t = np.concatenate(gathered_labels)
        else:
            all_preds_t  = torch.cat(all_preds).numpy()        if all_preds        else np.array([], dtype=np.int64)
            all_labels_t = torch.cat(all_labels_list).numpy()  if all_labels_list  else np.array([], dtype=np.int64)

        total_n = int(total_n)
        if total_n == 0:
            return {"val_loss": 0.0, "val_accuracy": 0.0}

        # Only compute / print metrics on rank 0
        if not self.is_main:
            return {}

        metrics = compute_metrics(all_labels_t, all_preds_t, use_weighted=True)
        print_metrics(metrics)

        return {
            "val_loss":         round(total_loss / total_n, 4),
            "val_accuracy":     round(metrics["accuracy"], 4),
            "val_sensor_usage": round(total_ratio / max(ratio_steps, 1), 4),
        }

    # ==========================================================================
    # Main training loop
    # ==========================================================================

    def train(self):
        _log(f"Starting joint training for {self.num_epochs} epochs  "
             f"(world_size={self.world_size}, rank={self.rank})")
        _log(f"BPTT steps      : {self.bptt_steps}")
        _log(f"save_dir        : {self.save_dir}")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            t0 = time.time()

            train_m = self.train_epoch()
            val_m   = self.validate()

            if self.is_main:
                elapsed = time.time() - t0
                _tau_e   = train_m.get("mean_tau", self._agent.get_temperature().item())
                _unc_e   = train_m.get("mean_uncertainty", 0.0)
                _unc_es  = f"  unc={_unc_e:.4f}" if self._agent.uncertainty_type != "none" else ""
                _log(
                    f"Epoch {epoch:3d}  "
                    f"loss={train_m.get('ce_loss', 0):.4f}  "
                    f"acc={train_m.get('accuracy', 0):.3f}  "
                    f"sd_ratio={train_m.get('active_ratio', 0):.2%}  "
                    f"agent_ratio={train_m.get('sensor_usage', 0):.2%}  "
                    f"τ={_tau_e:.3f}{_unc_es}  "
                    + (f"val_acc={val_m.get('val_accuracy', 0):.3f}  " if val_m else "")
                    + f"[{elapsed:.1f}s]"
                )

                # Save skip-pattern snapshot (only available for sigma-delta models)
                snap = self._model.get_sensing_stats() if hasattr(self._model, "get_sensing_stats") else {}
                snap["epoch"] = epoch
                snap["active_ratio"] = train_m.get("active_ratio", 0.0)
                self._skip_history.append(snap)

                # Save best checkpoint
                if val_m and val_m.get("val_accuracy", 0) > self.best_val_acc:
                    self.best_val_acc = val_m["val_accuracy"]
                    self.save_checkpoint("best_model.pt")
                    _log(f"  ↑ Best val_acc={self.best_val_acc:.4f}")

                # Save skip pattern history
                with open(os.path.join(self.save_dir, "skip_patterns.json"), "w") as f:
                    json.dump(self._skip_history, f, indent=2, default=str)

        _log("Training complete.")
        _log(f"Best val_accuracy : {self.best_val_acc:.4f}")

    # ==========================================================================
    # Checkpoint
    # ==========================================================================

    def save_checkpoint(self, name: str):
        if not self.is_main:
            return   # only rank 0 writes
        ckpt = {
            "epoch":        self.epoch,
            "global_step":  self.global_step,
            "best_val_acc": self.best_val_acc,
            "model":        self._model.state_dict(),
            "agent":        self._agent.state_dict(),
            "model_opt":    self.model_opt.state_dict(),
            "agent_opt":    self.agent_opt.state_dict(),
            "model_sched":  self.model_sched.state_dict(),
            "agent_sched":  self.agent_sched.state_dict(),
        }
        if self.predictive_mlp is not None:
            ckpt["predictive_mlp"] = self._pred_mlp.state_dict()
            ckpt["predictive_opt"] = self.predictive_opt.state_dict()
            ckpt["pred_sched"]     = self.pred_sched.state_dict()
        torch.save(ckpt, os.path.join(self.save_dir, name))

    def load_checkpoint(self, name: str):
        ckpt = torch.load(os.path.join(self.save_dir, name), map_location=self.device)
        self._model.load_state_dict(ckpt["model"])
        self._agent.load_state_dict(ckpt["agent"])
        self.model_opt.load_state_dict(ckpt["model_opt"])
        self.agent_opt.load_state_dict(ckpt["agent_opt"])
        self.model_sched.load_state_dict(ckpt["model_sched"])
        self.agent_sched.load_state_dict(ckpt["agent_sched"])
        if self.predictive_mlp is not None and "predictive_mlp" in ckpt:
            self._pred_mlp.load_state_dict(ckpt["predictive_mlp"])
            self.predictive_opt.load_state_dict(ckpt["predictive_opt"])
            self.pred_sched.load_state_dict(ckpt["pred_sched"])
        self.epoch        = ckpt["epoch"]
        self.global_step  = ckpt["global_step"]
        self.best_val_acc = ckpt["best_val_acc"]
        _log(f"Loaded checkpoint '{name}' (epoch {self.epoch})")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser("sigma_delta_joint_trainer")
    p.add_argument("--dataset", default="mhealth",
                   choices=["mhealth", "hmc", "wesad", "kinematics", "siscientisst", "seizeit2"])
    p.add_argument("--root",    required=True)
    p.add_argument("--save_dir",   default="./checkpoints_joint")
    p.add_argument("--port",       default="12355",
                   help="MASTER_PORT for DDP (torchrun sets this automatically).")
    # Optimiser
    p.add_argument("--model_lr",   type=float, default=1e-3)
    p.add_argument("--agent_lr",   type=float, default=1e-3)
    p.add_argument("--num_epochs", type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--bptt_steps", type=int,   default=8)
    # Loss weights
    p.add_argument("--ce_weight",             type=float, default=1.0)
    p.add_argument("--agent_sparsity_weight", type=float, default=0.0,
                   help="L1 penalty on agent p_soft.mean(). Increase to reduce agent_ratio. "
                        "Ignored when --target_agent_ratio is set.")
    p.add_argument("--target_agent_ratio",   type=float, default=None,
                   help="Target agent_ratio for squared-error constraint. "
                        "When set, loss += agent_sparsity_weight * (p_soft.mean() - target)^2. "
                        "Pushes ratio toward target from both directions. "
                        "Use agent_sparsity_weight to control how tightly the target is enforced.")
    p.add_argument("--predictive_weight",    type=float, default=0.0,
                   help="Weight for predictive auxiliary loss (belief → next model memory). "
                        "0 (default) = disabled. E.g. 0.1 trains GRU to be predictive of "
                        "future observations independently of the CE task loss.")
    p.add_argument("--predictive_lr",        type=float, default=1e-3,
                   help="Learning rate for the predictive MLP (default 1e-3).")

    # Sigma-delta
    p.add_argument("--init_threshold",     type=float, default=0.1)
    p.add_argument("--skip_steps",         type=int,   default=3)
    # Lagrangian

    # Agent
    p.add_argument("--agent_hidden",       type=int,   default=256)
    p.add_argument("--agent_modal_per",    type=int,   default=64,
                   help="Per-modality embedding dim fed directly to the policy head "
                        "and GRU. policy_head input = hidden + hidden/2 + M*modal_per.")
    p.add_argument("--agent_tau",          type=float, default=2.0,
                   help="Fixed Gumbel-sigmoid temperature (no annealing). "
                        "Higher = more exploration / softer gates. "
                        "Lower = sharper gates but smaller gradients.")
    p.add_argument("--history_length",     type=int,   default=10)
    # Model
    p.add_argument("--model_dim",          type=int,   default=512)
    p.add_argument("--modal_fusion",       default="concate",
                   choices=["concate", "cross_atten"])
    p.add_argument("--no_memory",          action="store_true")
    p.add_argument("--no_sigma_delta",     action="store_true",
                   help="Disable sigma-delta adaptive sensing (raw signal tokenisation).")
    p.add_argument("--only_predictive",    action="store_true",
                   help="Run agent (train belief/policy) but do NOT apply gate mask to input. "
                        "Useful for warming up the agent without degrading model performance.")
    p.add_argument("--use_device_wise_model", action="store_true",
                   help="Use former_device.py backbone with device-level POMDP gating instead of "
                        "sigma_former_sensor.py with sensor-level gating.")
    # Uncertainty-aware exploration (addresses Thompson Sampling approximation gap)
    p.add_argument("--agent_uncertainty", default="none",
                   choices=["none", "ensemble", "mc_dropout"],
                   help="Uncertainty estimation mode for the POMDP agent.\n"
                        "  none       — original fixed-variance Gumbel exploration (default)\n"
                        "  ensemble   — K independent policy heads; std of their logits = epistemic "
                                       "uncertainty; high-uncertainty sensors explore more (adaptive τ)\n"
                        "  mc_dropout — single head with AlwaysDropout; K stochastic MC passes; "
                                       "variance = uncertainty; works at both train and eval")
    p.add_argument("--agent_ensemble_k", type=int, default=5,
                   help="K = number of ensemble heads (ensemble) or MC passes (mc_dropout). "
                        "Higher K → better uncertainty estimate, more compute. Default 5.")
    p.add_argument("--agent_uncertainty_scale", type=float, default=1.0,
                   help="Scale for adaptive temperature widening: "
                        "τ_eff[m] = τ · (1 + scale · σ[m]). "
                        "0 = uncertainty computed but exploration not widened. Default 1.0.")
    p.add_argument("--agent_diversity_weight", type=float, default=0.0,
                   help="Weight for ensemble diversity regularisation loss "
                        "(only active when --agent_uncertainty=ensemble). "
                        "Prevents all K heads from collapsing to identical weights, "
                        "which would degenerate uncertainty estimates to ~0. "
                        "Suggested range: 0.001–0.05. Default 0 (disabled).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ---- DDP detection (torchrun sets LOCAL_RANK / WORLD_SIZE) --------------
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        setup_ddp(local_rank, world_size, port=args.port)

    # ---- Determine the 4-way (sigma_delta × device_wise) combination --------
    #
    #   no_sigma_delta=F, use_device_wise=F  →  sigma_former_sensor + pomdp_sensor_agent + DeltaDataset
    #   no_sigma_delta=T, use_device_wise=F  →  former_sensor        + pomdp_sensor_agent
    #   no_sigma_delta=F, use_device_wise=T  →  sigma_former_device  + pomdp_device_agent + DeltaDataset
    #   no_sigma_delta=T, use_device_wise=T  →  former_device        + pomdp_device_agent
    #
    use_sigma  = not args.no_sigma_delta
    use_device = args.use_device_wise_model
    apply_delta = use_sigma   # DeltaDataset only needed when sigma-delta is active

    logging.info(
        f"Mode: {'sigma_former_device' if use_sigma and use_device else 'former_device' if use_device else 'sigma_former_sensor' if use_sigma else 'former_sensor'}  "
        f"+ {'pomdp_device_agent' if use_device else 'pomdp_sensor_agent'}  "
        f"+ {'DeltaDataset' if apply_delta else 'raw signal'}"
    )

    # ---- Import dataset-specific loaders -----------------------------------
    from utils.modality_config import ModalityConfig

    if args.dataset == "mhealth":
        from dataset.mHEALTH_loader import MHealthDataset
        from dataset.delta_dataset import DeltaDataset
        from dataset.sequential_dataset import SequentialDataset, collate_sequential_batch

        train_s     = list(range(1, 8))
        val_s       = [8, 9, 10]
        num_classes = 12

        if use_device:
            # Device-wise: aligned with agent_trainer.py
            # Device 0 (left arm):  accel (3ch) + gyro (3ch) = 6 ch
            # Device 1 (right arm): accel (3ch) + gyro (3ch) = 6 ch
            ds = MHealthDataset(args.root, subjects=list(range(0, 11)),
                                time_steps=100, step=50, balance=False,
                                majority_n=500)
            modalities = [
                ModalityConfig('al', 3, 10, 0), ModalityConfig('gl', 3, 10, 0),
                ModalityConfig('ar', 3, 10, 1), ModalityConfig('gr', 3, 10, 1),
            ]
        else:
            # Sensor-wise: 13 individual single-channel modalities
            ds = MHealthDataset(args.root, subjects=list(range(1, 11)),
                                time_steps=50, step=25, balance=False)
            modalities = [ModalityConfig(f"m{i}", 1, 10) for i in range(13)]

        # Apply DeltaDataset when sigma-delta sensing is active
        if apply_delta:
            ds = DeltaDataset(ds, axis=-1)

        train_seq = SequentialDataset(ds, train_s)
        val_seq   = SequentialDataset(ds, val_s)

        train_loader = DataLoader(
            train_seq, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_sequential_batch, num_workers=2,
        )
        val_loader = DataLoader(
            val_seq, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_sequential_batch, num_workers=2,
        )

    elif args.dataset == "seizeit2":
        from dataset.seizeit2_loader import SeizeIT2Dataset
        from dataset.delta_dataset import DeltaDataset

        # SeizeIT2 (ds005873): wearable multimodal epilepsy seizure detection
        # Modalities: EEG (2 ch, device 0) + ECG (1 ch, device 1) +
        #             EMG (1 ch, device 2) + IMU acc+gyro (6 ch, device 3) = 10 ch total
        # Window: 500 samples (2 s @ 250 Hz), step: 250 (50 % overlap)
        # Classes: 0 = background, 1 = seizure
        #
        # ImageFolder-style loading — SequentialDataset is NOT used.
        # ~41 M windows / ~330 K per subject makes pre-loading a subject's full
        # sequence impossible.  Each __getitem__ reads exactly one 2-second window
        # on demand via readSignal(start=, n=).  train_epoch's `else` branch in
        # SigmaDeltaJointTrainer handles plain (x, y) batches as length-1 sequences.
        all_subjects = [f"sub-{i:03d}" for i in range(1, 126)]
        np.random.seed(42)
        rng_idx   = np.random.permutation(len(all_subjects))
        split_idx = int(len(all_subjects) * 0.8)
        train_s   = sorted([all_subjects[i] for i in rng_idx[:split_idx]])
        val_s     = sorted([all_subjects[i] for i in rng_idx[split_idx:]])
        num_classes = 2

        if use_device:
            modalities = [
                ModalityConfig('eeg', 2, 25, 0),
                ModalityConfig('ecg', 1, 25, 1),
                ModalityConfig('emg', 1, 25, 2),
                ModalityConfig('imu', 6, 25, 3),
            ]
        else:
            modalities = [ModalityConfig(f"m{i}", 1, 25) for i in range(10)]

        train_ds = SeizeIT2Dataset(
            args.root, train_s, time_steps=500, step=250,
            modalities=('eeg', 'ecg', 'emg', 'imu'), balance=False,
        )
        val_ds = SeizeIT2Dataset(
            args.root, val_s, time_steps=500, step=250,
            modalities=('eeg', 'ecg', 'emg', 'imu'), balance=False,
        )

        if apply_delta:
            train_ds = DeltaDataset(train_ds, axis=-1)
            val_ds   = DeltaDataset(val_ds,   axis=-1)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' not wired up yet in this CLI. "
                                  f"Add it following the mhealth pattern above.")

    # ---- Compute device groups (needed for device-wise gate expansion) -----
    from collections import defaultdict as _dd
    _sensor_mods = _dd(list)
    for m in modalities:
        _sensor_mods[m.device_idx].append(m)
    num_devices = len(_sensor_mods)
    channels_per_device = [
        sum(m.in_ch for m in _sensor_mods[d]) for d in sorted(_sensor_mods.keys())
    ]
    num_modal = sum(channels_per_device)   # total input channels
    # print("num_modal:", num_modal)

    # ---- Build model -------------------------------------------------------
    if use_sigma and use_device:
        # sigma_former_device: sigma-delta sensing, device-wise tokenisation
        from models.sigma_former_device import build_former_device as _build_sigma_device
        model = _build_sigma_device(
            num_classes       = num_classes,
            model_dim         = args.model_dim,
            return_mem        = True,
            return_sensing_info = True,
            init_threshold    = args.init_threshold,
            skip_steps        = args.skip_steps,
            learnable_skip    = False,
            modalities        = modalities,
            modal_fusion      = args.modal_fusion,
        )
        agent_cls               = BeliefStatePOMDPDeviceAgent
        # Use channel-level gating (aligned with agent_trainer.py: 12 gates not 2).
        # Device-level gating (num_devices=2) is too aggressive — one gate closure
        # kills 6/12 channels, causing instability.
        agent_num_sensors       = num_modal
        trainer_cpd             = None

    elif use_sigma and not use_device:
        # sigma_former_sensor: sigma-delta sensing, sensor-wise tokenisation
        from models.sigma_former_sensor import build_adaptive_sigma_former
        model = build_adaptive_sigma_former(
            num_classes       = num_classes,
            model_dim         = args.model_dim,
            return_mem        = True,
            return_sensing_info = True,
            init_threshold    = args.init_threshold,
            skip_steps        = args.skip_steps,
            learnable_skip    = False,
            modalities        = modalities,
            modal_fusion      = args.modal_fusion,
        )
        agent_cls               = BeliefStatePOMDPAgent
        agent_num_sensors       = len(modalities)   # one gate per sensor
        trainer_cpd             = None

    elif not use_sigma and use_device:
        # former_device: no sigma-delta, device-wise tokenisation
        from models.former_device import build_former_device as _build_plain_device
        model = _build_plain_device(
            num_classes  = num_classes,
            model_dim    = args.model_dim,
            return_mem   = True,
            modalities   = modalities,
            modal_fusion = args.modal_fusion,
        )
        agent_cls               = BeliefStatePOMDPDeviceAgent
        # Use channel-level gating (aligned with agent_trainer.py: 12 gates not 2).
        agent_num_sensors       = num_modal
        trainer_cpd             = None

    else:
        # former_sensor: no sigma-delta, sensor-wise tokenisation
        from models.former_sensor import build_former
        model = build_former(
            num_classes  = num_classes,
            model_dim    = args.model_dim,
            return_mem   = True,
            modalities   = modalities,
            modal_fusion = args.modal_fusion,
        )
        agent_cls               = BeliefStatePOMDPAgent
        agent_num_sensors       = len(modalities)
        trainer_cpd             = None

    logging.info(
        f"Model params : {sum(p.numel() for p in model.parameters()):,}  "
        f"num_devices={num_devices}  channels_per_device={channels_per_device}"
    )

    # ---- Build agent -------------------------------------------------------
    agent = agent_cls(
        num_sensors       = agent_num_sensors,
        obs_dim           = args.model_dim,
        hidden_dim        = args.agent_hidden,
        history_length    = args.history_length,
        tau               = args.agent_tau,
        modal_per         = args.agent_modal_per,
        uncertainty_type  = args.agent_uncertainty,
        ensemble_k        = args.agent_ensemble_k,
        uncertainty_scale = args.agent_uncertainty_scale,
    )

    if args.agent_uncertainty != "none":
        logging.info(
            f"Agent uncertainty: type={args.agent_uncertainty}  "
            f"K={args.agent_ensemble_k}  scale={args.agent_uncertainty_scale}  "
            f"diversity_weight={args.agent_diversity_weight}"
        )

    # ---- Build trainer and run ---------------------------------------------
    trainer = SigmaDeltaJointTrainer(
        model                  = model,
        agent                  = agent,
        train_loader           = train_loader,
        val_loader             = val_loader,
        model_lr               = args.model_lr,
        agent_lr               = args.agent_lr,
        bptt_steps             = args.bptt_steps,
        num_epochs             = args.num_epochs,
        ce_weight              = args.ce_weight,
        agent_sparsity_weight  = args.agent_sparsity_weight,
        predictive_weight      = args.predictive_weight,
        predictive_lr          = args.predictive_lr,
        agent_diversity_weight = args.agent_diversity_weight,
        save_dir               = args.save_dir,
        use_memory             = not args.no_memory,
        channels_per_device    = trainer_cpd,
        only_predictive        = args.only_predictive,
        target_agent_ratio     = args.target_agent_ratio,
        ddp_rank               = local_rank    if is_distributed else None,
        ddp_world_size         = world_size    if is_distributed else None,
    )
    trainer.train()

    if is_distributed:
        cleanup_ddp()
