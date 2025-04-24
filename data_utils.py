# data_utils.py

import torchvision
import torchvision.transforms as T

def build_cifar_datasets(dataset_name="cifar10"):
    """
    返回: (trainset, testset, num_classes).
    注意: 这里统一对图像做 Resize(224) + ImageNet统计的 Normalize,
         以兼容预训练 ResNet50 和 ViT.
    """
    # 常见ImageNet预训练模型的mean/std:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    transform = T.Compose([
        # T.Resize(224),   # 关键: 将CIFAR(32x32)放大到224x224
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    if dataset_name.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        testset  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError("Unknown dataset name: %s" % dataset_name)

    return trainset, testset, num_classes