#!/usr/bin/env python3

import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

import wandb

from data_utils import build_cifar_datasets
from tasks import get_cifar10_tasks, get_cifar100_tasks, TaskSubset
from memory import MemoryBuffer
from model import IncrementalResNet50, IncrementalViT, IncrementalViTTiny
from metrics import compute_faa_aaa_wca
from trainers_lunch import train_ER_LUNCH, train_DER_LUNCH, train_RAR_LUNCH
from lunch_utils import KendallUncertaintyWeighting, ExponentialMovingAverage


def set_global_seed(seed: int):
    """ 设置全局随机种子, 确保结果可复现 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_task(model, loader, device="cuda:0"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--method", type=str, default="ER", choices=["ER", "DER", "RAR"])
    parser.add_argument("--model_type", type=str, default="vittiny", choices=["resnet50", "vit", "vittiny"],
                        help="Choose backbone: ResNet50 or ViT.")
    parser.add_argument("--use_lunch", type=bool, default=True)
    parser.add_argument("--use_ema", type=bool, default=True, help="Enable EMA.")
    parser.add_argument("--memory_size", type=int, default=2000)
    parser.add_argument("--epochs_per_task", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--gpu", type=int, default=1, help="GPU index to use")
    parser.add_argument("--use_wandb", type=bool, default=True)

    # 新增: 随机种子
    parser.add_argument("--seed", type=int, default=2023, help="Random seed for reproducibility")

    args = parser.parse_args()

    # 设置随机种子
    set_global_seed(args.seed)

    device = f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu"
    print(f"Using device: {device}")

    # 若使用 wandb，则初始化
    if args.use_wandb:
        wandb.init(project=f"my_lunch_project_{args.dataset}", config=vars(args))
        wandb.run.name = f"{args.method}_{args.model_type}"
        wandb.run.save()

    # 2) 数据集
    trainset, testset, num_classes = build_cifar_datasets(args.dataset)
    if args.dataset == "cifar10":
        task_splits = get_cifar10_tasks()
    else:
        task_splits = get_cifar100_tasks()

    # 3) 模型
    if args.model_type == "resnet50":
        model = IncrementalResNet50(max_classes=num_classes).to(device)
    elif args.model_type == "vittiny":
        model = IncrementalViTTiny(
            max_classes=10,
            ckpt_path="/home/panqy/.cache/torch/hub/checkpoints/vit_base_patch16_224.pth",
            freeze_backbone=False,
            use_prompt=False
        ).to(device)
    else:
        model = IncrementalViT(max_classes=num_classes).to(device)

    # 4) 优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # 5) 回放内存
    store_logits = (args.method == "DER")
    replay_mem = MemoryBuffer(max_size=args.memory_size, store_logits=store_logits)

    # 6) LUNCH
    if args.method == "DER":
        loss_keys = ["new", "oldCE", "mse"]
    else:
        loss_keys = ["new", "old"]
    lw_model = None
    if args.use_lunch:
        lw_model = KendallUncertaintyWeighting(loss_keys).to(device)

    # 7) EMA
    ema = None
    if args.use_ema:
        ema = ExponentialMovingAverage(model, decay=0.999)

    # 8) 训练任务循环
    T = len(task_splits)
    acc_matrix = [[0.0] * T for _ in range(T)]
    acc_matrix_ema = [[0.0] * T for _ in range(T)] if args.use_ema else None

    from trainers import train_ER, train_DER, train_RAR

    def train_one_epoch(loader):
        """单个 epoch 的训练流程，根据是否启用 LUNCH 调用不同的训练函数"""
        if not args.use_lunch:
            # 原版
            if args.method == "ER":
                train_ER(model, nn.CrossEntropyLoss(), optimizer, loader, replay_mem, device)
            elif args.method == "DER":
                train_DER(model, nn.CrossEntropyLoss(), optimizer, loader, replay_mem, device)
            else:
                train_RAR(model, nn.CrossEntropyLoss(), optimizer, loader, replay_mem, device)
        else:
            # LUNCH 版
            if args.method == "ER":
                train_ER_LUNCH(model, lw_model, optimizer, loader, replay_mem, device, ema=ema)
            elif args.method == "DER":
                train_DER_LUNCH(model, lw_model, optimizer, loader, replay_mem, device, ema=ema)
            else:
                train_RAR_LUNCH(model, lw_model, optimizer, loader, replay_mem, device, ema=ema)

    # 依次学习每个 Task
    for t_idx, cids in enumerate(task_splits):
        train_ds_t = TaskSubset(trainset, cids)
        test_ds_t = TaskSubset(testset, cids)
        train_loader_t = DataLoader(train_ds_t, batch_size=args.batch_size, shuffle=True)

        # 对当前任务进行 epochs_per_task 轮训练
        for ep in range(args.epochs_per_task):
            train_one_epoch(train_loader_t)

        # Normal 模型评测
        for old_i in range(t_idx + 1):
            test_ds_old = TaskSubset(testset, task_splits[old_i])
            test_loader_old = DataLoader(test_ds_old, batch_size=64, shuffle=False)
            acc_val = evaluate_task(model, test_loader_old, device)
            acc_matrix[t_idx][old_i] = acc_val
            print(f"[LUNCH] T{t_idx}, {args.method}+{args.model_type}, normal => test T{old_i} = {acc_val * 100:.2f}%")

        if args.use_wandb:
            wandb.log({"acc_normal": acc_val * 100.0})

        # 如果启用了 EMA，则评测 EMA 状态
        if args.use_ema and ema is not None and acc_matrix_ema is not None:
            orig_sd = model.state_dict()
            ema_sd = ema.get_state_dict(model)
            model.load_state_dict(ema_sd)
            for old_i in range(t_idx + 1):
                test_ds_old = TaskSubset(testset, task_splits[old_i])
                test_loader_old = DataLoader(test_ds_old, batch_size=64, shuffle=False)
                acc_ema_val = evaluate_task(model, test_loader_old, device)
                acc_matrix_ema[t_idx][old_i] = acc_ema_val
                print(
                    f"[LUNCH] T{t_idx}, {args.method}+{args.model_type}, EMA => test T{old_i} = {acc_ema_val * 100:.2f}%")

            model.load_state_dict(orig_sd)

            if args.use_wandb:
                wandb.log({"acc_ema": acc_ema_val * 100.0})

    # 最终结果 (normal)
    FAA, AAA, WCA = compute_faa_aaa_wca(acc_matrix)
    print(f"\n[LUNCH final (normal)] model={args.model_type}, ds={args.dataset}, method={args.method}")
    print(f"FAA={FAA * 100:.2f}%, AAA={AAA * 100:.2f}%, WCA={WCA * 100:.2f}%")

    if args.use_wandb:
        wandb.log({
            "Final/FAA": FAA,
            "Final/AAA": AAA,
            "Final/WCA": WCA
        })

    # EMA 模型
    if args.use_ema and acc_matrix_ema is not None:
        FAA_ema, AAA_ema, WCA_ema = compute_faa_aaa_wca(acc_matrix_ema)
        print(f"\n[LUNCH final (EMA)] model={args.model_type}, ds={args.dataset}, method={args.method}")
        print(f"FAA={FAA_ema * 100:.2f}%, AAA={AAA_ema * 100:.2f}%, WCA={WCA_ema * 100:.2f}%")

        if args.use_wandb:
            wandb.log({
                "Final_EMA/FAA": FAA_ema,
                "Final_EMA/AAA": AAA_ema,
                "Final_EMA/WCA": WCA_ema
            })

        overhead_bytes = ema.get_memory_overhead()
        overhead_mb = overhead_bytes / (1024 ** 2)
        print(f"EMA overhead: {overhead_bytes} bytes (~{overhead_mb:.2f} MB)")
        wandb.log({"EMA_memory_overhead_MB": overhead_mb})

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()