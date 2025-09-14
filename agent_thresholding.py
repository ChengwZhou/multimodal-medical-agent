# agent_thresholding.py
import itertools
import math
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import deque

# ----- 假设：你已有的模型 -----
# from your_model_file import MultimodalActivityTransformer
# 这里我们假定 model(x) -> returns (logits, aux) where aux contains 'proj_tokens' shape [B, L, D]
# and tokenizers output scalar features per token if needed (we will use aux['proj_tokens'] or aux from model)

# -------------------------
# Helper: compute per-modality features
# -------------------------
def extract_per_modality_scalar_features_from_aux(aux: Dict[str, Any], strategy='l2'):
    """
    aux is expected to contain 'per_mod_tokens' or 'proj_tokens' = [B, L_total, D]
    We also expect knowledge of how tokens are arranged: currently tokens are concatenated
    per-modality as: [mod0_patch0..patchN, mod1_patch0..patchN, ...]
    This function returns per-modality per-patch scalar (B, M, P) where:
      - M = num_modalities
      - P = patches_per_modality (here 10)
    strategy: 'l2' or 'mean_abs' or 'logit_conf' etc.
    """
    # aux['proj_tokens'] shape: [B, L_total, D] or aux['concat_tokens'] maybe [B, L_total, 1]
    if 'proj_tokens' in aux:
        T = aux['proj_tokens']   # [B, L_total, D]
        # collapse D -> scalar per token
        if strategy == 'l2':
            token_scalar = T.norm(p=2, dim=-1)   # [B, L_total]
        elif strategy == 'mean_abs':
            token_scalar = T.abs().mean(dim=-1)
        else:
            token_scalar = T.norm(p=2, dim=-1)
    elif 'concat_tokens' in aux:
        token_scalar = aux['concat_tokens'].squeeze(-1)  # assume [B, L_total]
    else:
        raise ValueError("aux must contain 'proj_tokens' or 'concat_tokens'")

    return token_scalar  # [B, L_total]


# -------------------------
# Agent: per-modality thresholds (storage + grid-search)
# -------------------------
class ThresholdAgent:
    """
    管理每个 modality 的阈值与策略参数（T_U,T_r, CONF_EXIT, MIN_ON, MIN_OFF, COOLDOWN）
    - thresholds: base thresholds per-modality (list or ndarray)
    - search_space: values to grid-search for tuning (dict)
    """

    def __init__(self, num_modalities: int, patches_per_modal: int = 10,
                 init_thresholds: List[float] = None, device='cpu'):
        self.M = num_modalities
        self.P = patches_per_modal
        self.device = device

        if init_thresholds is None:
            self.thresholds = np.ones(self.M) * 0.5  # default
        else:
            assert len(init_thresholds) == self.M
            self.thresholds = np.array(init_thresholds, dtype=float)

    def decide_on_off_sequence(self,
                               token_scalar: np.ndarray,
                               thresholds: np.ndarray,
                               T_r: float = 0.0,
                               conf_exit: int = 1,
                               min_on: int = 1,
                               min_off: int = 1,
                               cooldown: int = 0) -> np.ndarray:
        """
        token_scalar: [B, L_total]  (L_total = M * P)
        thresholds: [M,] array of thresholds
        Return: binary on/off sequence per patch per modality: shape [B, M, P] (0/1)
        Rules implemented:
         - Compare each patch scalar > thresholds[mod]
         - Hysteresis via T_r (a relative relax amount) not fully continuous: we implement as offset
         - CONF_EXIT: number of consecutive patches below threshold to switch OFF
         - MIN_ON/MIN_OFF: minimum consecutive on/off patches length
         - COOLDOWN: after a turn-off, wait cooldown patches before next on allowed
        """
        B, Ltot = token_scalar.shape
        M = self.M
        P = self.P
        assert Ltot == M * P
        out = np.zeros((B, M, P), dtype=np.int32)

        # For each batch and modality handle independently
        for b in range(B):
            for m in range(M):
                th = thresholds[m]
                start = m * P
                end = start + P
                seq = token_scalar[b, start:end]  # shape [P,]
                # simple binary raw decisions
                raw = (seq > th).astype(np.int32)  # 1 candidate ON
                # apply hysteresis with T_r: we treat T_r as additive offset lower threshold
                low_th = th - T_r
                if low_th < 0: low_th = 0.0
                # state machine
                state = 0  # 0 off, 1 on
                on_counter = 0
                off_counter = 0
                cooldown_counter = 0
                conf_counter = 0
                for i in range(P):
                    val = seq[i]
                    if cooldown_counter > 0:
                        # enforced off in cooldown
                        out[b, m, i] = 0
                        cooldown_counter -= 1
                        off_counter += 1
                        on_counter = 0
                        continue

                    if state == 0:
                        # consider turning on if val > th
                        if val > th:
                            on_counter += 1
                            off_counter = 0
                            # enforce min_on: only set to on when reached 1 patch (or min_on)
                            if on_counter >= max(1, 1):  # immediate turn-on; keep min_on enforced after turning on
                                state = 1
                                out[b, m, i] = 1
                                on_counter = 1
                                off_counter = 0
                            else:
                                out[b, m, i] = 0
                        else:
                            out[b, m, i] = 0
                            off_counter += 1
                    else:
                        # currently ON
                        if val < low_th:
                            conf_counter += 1
                        else:
                            conf_counter = 0
                        # only exit when conf_counter >= conf_exit and we've been on at least min_on
                        if (conf_counter >= conf_exit and on_counter >= min_on) or (off_counter >= min_off):
                            # go off
                            state = 0
                            out[b, m, i] = 0
                            off_counter = 1
                            on_counter = 0
                            cooldown_counter = cooldown
                            conf_counter = 0
                        else:
                            out[b, m, i] = 1
                            on_counter += 1
                            off_counter = 0
                # end per-patch
        return out  # [B, M, P], 0/1

    def compute_energy_and_triggers(self,
                                    on_off: np.ndarray,
                                    power_table: Dict[int, float],
                                    sample_rate_hz: float,
                                    patch_duration_sec: float) -> Tuple[float, float]:
        """
        on_off: [B, M, P] binary
        power_table: dict mapping modality idx -> mW when ON (idle/off cost ignored)
        sample_rate_hz: original sensor sampling rate (for conversion)
        patch_duration_sec: seconds per patch (e.g., patch_size / fs)
        Returns average_energy_mW (averaged across batch & time), triggers_per_hour
        """
        B, M, P = on_off.shape
        # duty cycle per modality
        duty = on_off.mean(axis=(0, 2))  # [M,]
        # energy per modality (mW) weighted by duty
        energies = np.array([power_table.get(m, 0.0) for m in range(M)])  # mW when ON
        avg_mw = float((duty * energies).sum())  # average mW across modalities
        # triggers = count OFF->ON transitions per hour (we have patches -> convert to hours)
        # number of patches per second = 1/patch_duration_sec; per hour = 3600/patch_duration_sec
        transitions = 0
        for b in range(B):
            for m in range(M):
                seq = on_off[b, m, :]
                trans = np.sum((seq[1:] > seq[:-1]) & (seq[1:] == 1))
                transitions += trans
        transitions_per_sample = transitions / B
        patches_per_hour = 3600.0 / patch_duration_sec
        # scale transitions observed per sample window (P patches) to per hour:
        # transitions_per_sample / P * patches_per_hour
        triggers_per_hour = float(transitions_per_sample / P * patches_per_hour)
        return avg_mw, triggers_per_hour

#
