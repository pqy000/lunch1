#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings('ignore')

def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

@torch.no_grad()
def eval_one_epoch(model, criterion, dataloader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for demonstration")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to your vit_tiny_patch16_224 checkpoint .pth file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: cuda or cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 准备 CIFAR-10 数据
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader  = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = 10  # CIFAR-10

    # 2. 初始化模型, 先设定与原权重匹配的 num_classes=1000, 再做 partial load
    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=1000,
        in_chans=3
    )

    # 3. 加载 checkpoint, 但 strict=False 忽略最后一层
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print("==> Missing keys:", missing_keys)
    print("==> Unexpected keys:", unexpected_keys)

    # 4. 手动替换 final head 以适应 CIFAR-10
    #    vit_tiny_patch16_224 默认 head 输入维度是 192，可用 print(model) 检查
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    nn.init.normal_(model.head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(model.head.bias)

    # 如果需要适配小尺寸输入，如 CIFAR-10 的 32x32，可调用 set_input_size
    # (注：部分 timm 版本支持)
    try:
        model.set_input_size(img_size=32, patch_size=4)
        print("Set input size to 32, patch_size=4 for CIFAR-10.")
    except AttributeError:
        print("This timm version may not support `set_input_size`. Ignored.")

    model.to(device)

    # 5. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # 6. 训练 & 测试
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, trainloader, device)
        test_loss, test_acc   = eval_one_epoch(model, criterion, testloader, device)

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()