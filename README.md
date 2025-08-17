# Multi-Modal Transformer Baseline for ScientISST-MOVE Dataset

This repository provides a baseline implementation for **multi-modal activity recognition** using the [ScientISST-MOVE Biosignals Dataset](https://physionet.org/content/scientisst-move-biosignals/1.0.1/).  
The baseline leverages **Transformer-based modality sub-networks + fusion**, with support for **historical context** as additional input.

---

## 📂 Project Structure

├── loader.py # Data loading, EDF parsing, temporal slicing, multi-modal alignment
├── baseline_model.py # Multi-modal baseline model with modality subnets + fusion
├── transformer_utils.py 
├── train.py # Training script (supports single GPU and DDP)
├── utils.py # Helper functions (logging, metrics, etc.)
└── README.md # Project documentation


---

## 📊 Dataset

The dataset is hosted on [PhysioNet](https://physionet.org/content/scientisst-move-biosignals/1.0.1/).  
It contains synchronized **multi-modal biosignals** recorded during human activity:

- **ECG** (Electrocardiogram)
- **EDA** (Electrodermal Activity)
- **EMG** (Electromyography)
- **RESP** (Respiration)
- **ACC** (Accelerometer, 3-axis)
- **EDA** (Electrodermal Activity)
- **Temperature**
- **Labels**: Activities performed (e.g., rest, walking, running, etc.)

We design the task as **Activity Classification**.

---

## ⚙️ Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.13
- Torchvision
- NumPy
- Pandas
- MNE (for EDF file reading)
- SciPy
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📥 Data Preparation

Download the dataset:
```bash
wget -r -N -c -np https://physionet.org/files/scientisst-move-biosignals/1.0.1/
```

Place the EDF files under:
```bash
./data/edf/
```

The ```loader.py``` will:

-Parse EDF files

-Slice data into temporal windows (e.g., 3s per sample)

-Ensure all modalities are time-synchronized

-Return (multi_modal_tensor, label) for each sample

## 🏗️ Model

The baseline model consists of:

1.Modality-specific sub-networks (each maps raw signal to embeddings)

2.Transformer Fusion Module (learns cross-modal interactions)

3.Historical Context Encoding (previous window embeddings concatenated to current input)

4.Classifier Head (predicts activity class)


## 🚀 Training
Single GPU
```bash
python train.py --epochs 50 --batch_size 64 --lr 1e-4
```
Multi-GPU (DDP)
```bash
torchrun --nproc_per_node=4 train.py --epochs 50 --batch_size 64 --lr 1e-4 --ddp
```
Arguments
```bash
--epochs : number of training epochs

--batch_size : mini-batch size

--lr : learning rate

--window_size : time window length in seconds (default 3s)

--ddp : enable DistributedDataParallel
```
## 📈 Evaluation

During training, metrics are logged:

Accuracy

F1-score

Confusion Matrix

Evaluation is automatically run on the validation set after each epoch.


## ✨ Acknowledgements

PhysioNet for providing the dataset

ScientISST Foundation for hardware and data collection