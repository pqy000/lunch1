#!/usr/bin/env bash

mkdir -p logs

DATASETS=("cifar10" "cifar100")
METHODS=("DER" "RAR")
MODELS=("resnet50")
SEEDS=(1996)

EPOCHS=5
MEM_SIZE=2000
BATCH=32
LR=0.001

# 如果您有多个GPU, 可以在这里将其罗列
GPUS=(0)

# 用于给不同实验分配GPU
gpu_index=0

for dset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for net in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do

        # 取下一个GPU编号
        gpu_id="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
        ((gpu_index++))

        # 组合实验名称
        EXP_NAME="no_lunch_${dset}_${method}_${net}_seed${seed}_gpu${gpu_id}"
        LOGF="logs/${EXP_NAME}.log"

        echo "=== Running $EXP_NAME on GPU=$gpu_id ==="

        nohup python main.py \
          --dataset "$dset" \
          --method "$method" \
          --model_type "$net" \
          --memory_size "$MEM_SIZE" \
          --epochs_per_task "$EPOCHS" \
          --batch_size "$BATCH" \
          --lr "$LR" \
          --gpu "$gpu_id" \
          --seed "$seed" \
          > "$LOGF" 2>&1 &

        echo "Launched => $LOGF (GPU=$gpu_id)"
        echo
      done
    done
  done
done

echo "All no_lunch experiments dispatched."