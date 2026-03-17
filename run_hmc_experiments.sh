#!/bin/bash
# 文件名: run_hmc_experiments.sh
# 用法: chmod +x run_hmc_experiments.sh && ./run_hmc_experiments.sh

# 设置实验根目录（可自行修改）
EXP_ROOT="./save"
mkdir -p $EXP_ROOT

# 当前时间戳，用于日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 公共参数（避免重复写）
COMMON_ARGS="--batch_size 30 \
             --bptt_steps 10 \
             --num_epochs 100 \
             --dataset hmc \
             --root /usr/homes/cxz760/data/hmc-sleep-staging/physionet.org/files/hmc-sleep-staging/1.1/recordings \
             --modal_fusion cross_atten \
             --model_lr 1e-4 \
             --agent_lr 1e-5"


LOG1="${EXP_ROOT}/hmc_only_predictive_SD_T0.1_step1.log"
echo "=========================================="
echo "Starting Experiment 1: only_predictive"
echo "Log file: $LOG1"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=12355 \
    trainer/sigma_delta_modality_masking_agent_trainer.py \
    $COMMON_ARGS \
    --gating_weight 0.05 \
    --only_predictive \
    2>&1 | tee $LOG1


if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Experiment 1 FAILED! Stop here."
    exit 1
fi

echo -e "\nExperiment 1 finished.\n\n"


LOG2="${EXP_ROOT}/hmc_gating_weight_0.01_SD_T0.1_step1.log"
echo "=========================================="
echo "Starting Experiment 2: gating_weight=0.01"
echo "Log file: $LOG2"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=12355 \
    trainer/sigma_delta_modality_masking_agent_trainer.py \
    $COMMON_ARGS \
    --gating_weight 0.01 \
    2>&1 | tee $LOG2

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Experiment 2 FAILED!"
    exit 1
fi

LOG3="${EXP_ROOT}/hmc_gating_weight_0.001_SD_T0.1_step1.log"
echo "=========================================="
echo "Starting Experiment 3: gating_weight=0.01"
echo "Log file: $LOG2"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=12355 \
    trainer/sigma_delta_modality_masking_agent_trainer.py \
    $COMMON_ARGS \
    --gating_weight 0.001 \
    2>&1 | tee $LOG3

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Experiment 3 FAILED!"
    exit 1
fi


echo -e "\nAll experiments completed successfully!"
echo "Logs saved in $EXP_ROOT:"
echo "  - $(basename $LOG1)"
echo "  - $(basename $LOG2)"
echo "  - $(basename $LOG3)"