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
from trainers import train_ER, train_DER, train_RAR


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
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--method", type=str, default="ER", choices=["ER", "DER", "RAR"])
    parser.add_argument("--model_type", type=str, default="vittiny",
                        choices=["resnet50", "vit", "vittiny"],
                        help="Choose backbone: ResNet50 or ViT.")
    parser.add_argument("--memory_size", type=int, default=2000)
    parser.add_argument("--epochs_per_task", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--gpu", type=int, default=1, help="GPU index to use")
    parser.add_argument("--use_wandb", type=bool, default=True)

    # 新增: 随机种子
    parser.add_argument("--seed", type=int, default=2023, help="Random seed for reproducibility")

    args = parser.parse_args()

    # 设置全局随机种子
    set_global_seed(args.seed)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # wandb 初始化
    if args.use_wandb:
        wandb.init(project=f"my_cl_project_{args.dataset}", config=vars(args))
        wandb.run.name = f"{args.method}_{args.model_type}"
        wandb.run.save()

    # 1) 数据集
    trainset, testset, num_classes = build_cifar_datasets(args.dataset)
    if args.dataset == "cifar10":
        task_splits = get_cifar10_tasks()
    else:
        task_splits = get_cifar100_tasks()

    # 2) 模型
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

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def count_params(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    print(f'Parameters: {count_params(model):.2f}M')

    store_logits = (args.method == "DER")
    replay_mem = MemoryBuffer(max_size=args.memory_size, store_logits=store_logits)

    # 5) 训练
    T = len(task_splits)
    acc_matrix = [[0.0] * T for _ in range(T)]

    for t_idx, cids in enumerate(task_splits):
        train_ds_t = TaskSubset(trainset, cids)
        test_ds_t = TaskSubset(testset, cids)
        train_loader_t = DataLoader(train_ds_t, batch_size=args.batch_size, shuffle=True)

        # 针对当前任务进行若干个 epoch 的训练
        for ep in range(args.epochs_per_task):
            if args.method == "ER":
                train_ER(model, criterion, optimizer, train_loader_t, replay_mem, device)
            elif args.method == "DER":
                train_DER(model, criterion, optimizer, train_loader_t, replay_mem, device)
            else:  # RAR
                train_RAR(model, criterion, optimizer, train_loader_t, replay_mem, device)

        # 训练完 task t_idx 后, 对 0..t_idx 任务进行评测
        for old_i in range(t_idx + 1):
            test_ds_old = TaskSubset(testset, task_splits[old_i])
            test_loader_old = DataLoader(test_ds_old, batch_size=64, shuffle=False)
            acc_val = evaluate_task(model, test_loader_old, device=device)
            acc_matrix[t_idx][old_i] = acc_val
            print(f"[No-Lunch] T{t_idx}, test T{old_i}, "
                  f"model={args.model_type}, method={args.method}, ACC={acc_val * 100:.2f}%")
            if args.use_wandb:
                wandb.log({"acc": acc_val * 100.0})

    FAA, AAA, WCA = compute_faa_aaa_wca(acc_matrix)
    print(f"\n[No-Lunch final] model={args.model_type}, dataset={args.dataset}, method={args.method}")
    print(f"FAA={FAA * 100:.2f}%, AAA={AAA * 100:.2f}%, WCA={WCA * 100:.2f}%")

    if args.use_wandb:
        wandb.log({
            "Final/FAA": FAA,
            "Final/AAA": AAA,
            "Final/WCA": WCA,
        })
        wandb.finish()


if __name__ == "__main__":
    main()