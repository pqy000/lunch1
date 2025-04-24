#!/usr/bin/env bash

mkdir -p logs

DATASETS=("cifar10" "cifar100")
METHODS=("ER" "DER" "RAR")
MODELS=("resnet50" "vittiny")

SEEDS=(42)
EPOCHS=3
MEM_SIZE=2000
BATCH=32
LR=0.001

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
GPUS=(0 1)

IDX=0

for dset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for net in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        gpu_index=$((IDX % ${#GPUS[@]}))
        GPU_ID=${GPUS[$gpu_index]}

        # 如果 GPU_ID >= NUM_GPUS，则说明机器上并没有这块卡
        # 可回退到0或CPU
        if [ "$GPU_ID" -ge "$NUM_GPUS" ]; then
          echo "Warning: GPU $GPU_ID not available on this machine, fallback to CPU"
          DEVICE_ARG="cpu"
        else
          DEVICE_ARG="cuda:${GPU_ID}"
        fi

        EXP_NAME="lunch_${dset}_${method}_${net}_seed${seed}"
        LOGF="logs/${EXP_NAME}.log"
        echo "=== Running $EXP_NAME on $DEVICE_ARG ==="

        nohup python main.py \
          --dataset "$dset" \
          --method "$method" \
          --model_type "$net" \
          --memory_size "$MEM_SIZE" \
          --epochs_per_task "$EPOCHS" \
          --batch_size "$BATCH" \
          --lr "$LR" \
          --device "$DEVICE" \
          > "$LOGF" 2>&1 &

        IDX=$((IDX+1))
      done
    done
  done
done

echo "All lunch+ema experiments done."