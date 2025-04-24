# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class IncrementalResNet50(nn.Module):
    def __init__(self, max_classes=100):
        super().__init__()
        base_net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base_net.children())[:-1])  # remove fc
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, max_classes)

    def forward(self, x):
        feats = self.features(x)
        feats = self.pool(feats)
        feats = feats.view(feats.size(0), -1)
        logits = self.fc(feats)
        return logits