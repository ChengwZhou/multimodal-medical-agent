# Training Pipeline README

This project provides three main training pipelines for sequential biosignal modeling using PyTorch. Each trainer is designed for different levels of model complexity and use cases. Below is an overview and usage guide for each trainer.

---
## Model Descriptions

### 1. Sensor-wise Transformer (`models/former_sensor.py`)
- **Architecture:**  
  Multimodal transformer with per-sensor tokenization. Each sensor/modality is tokenized using a 1D convolution, then all tokens are concatenated or fused via cross-modal attention. The model supports positional encoding, fusion transformer layers, and a classification head.
- **Use Case:**  
  Standard sequential modeling of multimodal biosignals, where each sensor is treated independently.
- **Key Options:**  
  - `modal_fusion`: Choose between simple concatenation or cross-modal attention for token fusion.
  - `model_dim`, `nhead`, `fusion_depth`: Control transformer size and depth.

### 2. Device-wise Transformer (`models/former_device.py`)
- **Architecture:**  
  Similar to the sensor-wise transformer, but modalities are grouped by device. Each device's signals are tokenized and fused, enabling device-level attention and gating. Supports device-aware gating agents for adaptive sensing.
- **Use Case:**  
  Scenarios where sensors are grouped by device, and device-level adaptation or gating is desired.
- **Key Options:**  
  - `modalities`: List of `ModalityConfig` objects specifying device grouping.
  - `modal_fusion`: Supports both concatenation and cross-modal attention.
  - Compatible with `DeviceGatingAgent` for device-level adaptive sensing.

### 3. Sigma-Delta Adaptive Masking Transformer (`models/sigma_former_sensor.py`, `models/sigma_former_device.py`)
- **Architecture:**  
  Transformer backbone with an adaptive sigma-delta masking module. Each modality or device has a learnable threshold and skip logic, enabling dynamic masking of input patches based on activity. Custom gradients allow for efficient training with masking.
- **Use Case:**  
  Research on adaptive sensing, energy-efficient multimodal learning, and dynamic input selection.
- **Key Options:**  
  - `init_threshold`, `skip_steps`: Control initial masking sensitivity and skip logic.
  - `learnable_skip`: Optionally learn the skip interval.
  - `return_sensing_info`: Output masking statistics for analysis.

---

## Agent Module Descriptions

### 1. SensorGatingAgent (`models/agent_sensor_masking.py`)
- **Mechanism:**  
  Learns to adaptively gate (on/off) each sensor/modality at each time step using a Gumbel-Sigmoid estimator and straight-through (ST) trick for hard decisions. Receives per-modality features and sensor state history, outputs soft gating probabilities and hard gating actions.
- **Use Case:**  
  Sensor-wise adaptive sensing for energy-efficient or context-aware biosignal modeling.
- **Key Options:**  
  - `num_modalities`, `feature_dim`: Number of modalities and feature dimension.
  - `history_length`: Number of past steps for state tracking.
  - `gumbel_tau`, `thresh_init`, `thresh_range`: Gumbel temperature and gating threshold settings.
  - `compute_aux_loss`: Joint loss combining task performance, energy cost, and trigger penalty.

### 2. DeviceGatingAgent (`models/agent_device_masking.py`)
- **Mechanism:**  
  Similar to `SensorGatingAgent`, but operates at the device level. Aggregates features per device, applies Gumbel-Sigmoid gating, and outputs device-wise gating actions. Supports device grouping via `ModalityConfig`.
- **Use Case:**  
  Device-wise adaptive sensing, where multiple sensors are grouped and controlled together.
- **Key Options:**  
  - `num_modalities`, `modalities`, `feature_dim`: Device grouping and feature config.
  - `history_length`, `gumbel_tau`, `thresh_init`, `thresh_range`: As above, but for devices.
  - `compute_aux_loss`: Device-level loss with task, energy, and trigger terms.

---

**See the respective files in `models/` for implementation details and further configuration.**

---

## 1. `sequential_trainer.py`

**Purpose:**  
Standard sequential trainer for real-time prediction tasks. Supports DDP (Distributed Data Parallel), mixed precision, and gradient accumulation.

**Key Features:**
- Sequential window-based training
- DDP support for multi-GPU training
- Mixed precision (AMP) for faster training
- Gradient accumulation for large effective batch sizes
- Checkpointing and validation

**Usage:**
```bash
python trainer/sequential_trainer.py \
  --batch_size 12 \
  --gradient_accumulation_steps 8 \
  --num_epochs 10 \
  --root /path/to/data
```

**Main Arguments:**
- `--batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients
- `--num_epochs`: Number of training epochs
- `--root`: Path to dataset root

---

## 2. `agent_trainer.py`

**Purpose:**  
Trainer for models with a sensor gating agent. Supports BPTT (Backpropagation Through Time), sensor gating, contrastive alignment, and predictive coding losses.

**Key Features:**
- Joint training of model and agent
- BPTT for long sequence modeling
- Sensor gating for adaptive sensing
- Optional contrastive and predictive coding losses
- DDP and mixed precision support

**Usage:**
```bash
python torchrun --nproc_per_node=1 --master_port=12355 trainer/agent_trainer.py --batch_size 12 --bptt_steps 10 --num_epochs 10 --dataset mhealth --root /mnt/vstor/CSE_ECSE_GXD234/data/MHEALTHDATASET --model_lr 1e-4 --agent_lr 1e-3 --num_epochs 100 --gating_weight 0.05
```

**Main Arguments:**
- `--dataset`: Dataset name (`siscientisst`, `mhealth`, `hmc`, `wesad`)
- `--root`: Path to dataset root
- `--bptt_steps`: Number of windows for BPTT
- `--use_contrastive_loss`: Enable contrastive loss
- `--use_predictive_loss`: Enable predictive coding loss
- `--modal_fusion`: Fusion method (`concate` or `CrossModalAttention`)
- `--use_device_wise_model`: Use device-wise model and agent


---

## 3. `sigma_delta_modality_masking_agent_trainer.py`

**Purpose:**  
Advanced trainer for models with adaptive sigma-delta modality masking and sensor gating. Designed for research on adaptive sensing and efficient multimodal learning.

**Key Features:**
- Sigma-delta adaptive modality masking
- Sensor gating agent
- BPTT, contrastive, and predictive coding losses
- Fine-grained optimizer parameter groups (e.g., per-threshold learning rates)
- DDP and mixed precision support

**Usage:**
```bash
torchrun --nproc_per_node=1 --master_port=12355 trainer/sigma_delta_modality_masking_agent_trainer.py --batch_size 12 --bptt_steps 10 --num_epochs 100 --dataset mhealth --root /mnt/vstor/CSE_ECSE_GXD234/data/MHEALTHDATASET --model_lr 1e-4 --agent_lr 1e-3 --num_epochs 100 --gating_weight 0.05
```

**Main Arguments:**
- All arguments from `agent_trainer.py`
- `--SD_active_weight`: Weight for sigma-delta activation loss
- `--init_threshold`: Initial threshold for adaptive masking
- `--skip_step`: Skip steps for masking


---

## General Notes

- All trainers support checkpointing and can resume from saved checkpoints.
- For DDP, ensure the correct environment variables are set or use `torchrun`.
- For custom datasets or models, modify the dataset/model construction sections in each script.
- For more details on arguments, run:
  ```bash
  python trainer/<trainer_script>.py --help
  ```

---

## File Overview

- `trainer/sequential_trainer.py`: Standard sequential trainer.
- `trainer/agent_trainer.py`: Trainer with sensor gating agent and advanced losses.
- `trainer/sigma_delta_modality_masking_agent_trainer.py`: Trainer with adaptive sigma-delta masking and agent.

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- tqdm, numpy, and other dependencies (see `requirements.txt`)

---

## Citation

If you use this codebase in your research, please cite the original paper or repository as appropriate.

---