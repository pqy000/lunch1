#!/usr/bin/env bash

# run_exps_lunch.sh
# 批量运行 main_lunch.py (启用 LUNCH + EMA)，使用 --gpu 指定显卡，并通过 --use_wandb 记录到wandb。
# 并行使用多卡：依次为每个实验分配 GPU。

mkdir -p logs

DATASETS=("cifar10" "cifar100")
METHODS=("ER" "DER" "RAR")
MODELS=("resnet")
SEEDS=(42 2023 314)

EPOCHS=5
MEM_SIZE=2000
BATCH=32
LR=0.001

GPUS=(0 1)
# 用来循环分配 GPU 的索引
gpu_index=0
#######################################################

for dset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for net in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do

        gpu_id="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
        ((gpu_index++))

        EXP_NAME="lunch_${dset}_${method}_${net}_seed${seed}_gpu${gpu_id}"
        LOGF="logs/${EXP_NAME}.log"

        echo "=== Running $EXP_NAME (LUNCH+EMA) on GPU=$gpu_id ==="

        # 通过 --gpu $gpu_id 指定显卡，并启用 --use_lunch, --use_ema, --use_wandb
        nohup python main_lunch.py \
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

        echo "Launched => $LOGF"
        echo
      done
    done
  done
done

echo "All lunch+ema experiments dispatched."