# model.py

import torch
import torch.nn as nn
import timm
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights
)

class IncrementalResNet50(nn.Module):
    """
    与原先相同，用于对比参照
    """
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


class IncrementalViTTiny(nn.Module):
    """
    使用 timm 'vit_tiny_patch16_224' 架构进行增量学习：
    - 可指定是否冻结除最后层外的权重；
    - 可通过传入 ckpt_path 对模型权重进行部分加载；
    - 最后一层替换为输出 max_classes 大小。
    """
    def __init__(self, max_classes=100,
                 ckpt_path=None,
                 freeze_backbone=True,
                 use_prompt=False,
                 num_tasks=10,
                 prompt_len=8):
        super().__init__()

        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=1000,
            in_chans=3
        )

        # 若用户提供了 ckpt_path, 则部分加载外部权重
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            missing_keys, unexpected_keys = self.vit.load_state_dict(state_dict, strict=False)
            print("==> [IncrementalViTTiny] Load ckpt partial done.")
            print("    Missing keys:   ", missing_keys)
            print("    Unexpected keys:", unexpected_keys)

        # 2) 替换分类 head => [batch, max_classes]
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, max_classes)
        # 手动初始化最后一层
        nn.init.normal_(self.vit.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.vit.head.bias)

        # 3) 根据需求, 冻结除最后层以外的参数
        if freeze_backbone:
            for name, p in self.vit.named_parameters():
                # 跳过刚替换的最后一层
                if "head" not in name:
                    p.requires_grad = False

        # 4) 若想启用 prompt 学习, 增加可学习 Parameter
        self.use_prompt = use_prompt
        if use_prompt:
            # prompt形状: (num_tasks, prompt_len, embed_dim)
            embed_dim = in_features
            prompt_shape = (num_tasks, prompt_len, embed_dim)
            self.prompt_embeddings = nn.Parameter(torch.randn(*prompt_shape))

        # 也可尝试设置输入大小, 若 timm 版本支持 vit_tiny 的 set_input_size
        try:
            self.vit.set_input_size(img_size=32, patch_size=4)
            print("[IncrementalViTTiny] set_input_size(32,4) done.")
        except AttributeError:
            pass

    def forward(self, x, task_id=0):
        """
        x: [B,3,H,W], e.g. CIFAR(32x32).
        task_id: 若use_prompt=True, 指示当前任务id(0..num_tasks-1),
                 用以注入对应Prompt.
        """
        if not self.use_prompt:
            return self.vit(x)
        else:
            # 需要在 forward_features 注入prompt
            old_forward = self.vit.forward_features

            def new_forward_features(_model, _x):
                out = _model.patch_embed(_x)         # => shape [B,#tokens,embed_dim]
                cls_token = _model.cls_token.expand(out.shape[0], -1, -1)
                out = torch.cat((cls_token, out), dim=1)  # [B,1+N,d]

                # 注入prompt
                prompt_vec = self.prompt_embeddings[task_id]  # [prompt_len,embed_dim]
                prompt_batch = prompt_vec.unsqueeze(0).expand(out.size(0), -1, -1)
                out = torch.cat((out, prompt_batch), dim=1)   # [B,1+N+prompt_len,d]

                # 位置编码 + dropout
                out = _model._pos_embed(out)
                out = _model.drop_rate(out)
                out = _model.blocks(out)
                out = _model.norm(out)
                return out[:,0]

            self.vit.forward_features = new_forward_features.__get__(self.vit, type(self.vit))
            logits = self.vit(x)
            self.vit.forward_features = old_forward
            return logits


class IncrementalViT(nn.Module):
    """
    Vision Transformer (vit_b_16), ImageNet预训练 => heads.head -> max_classes
    注意: 需给输入图像Resize到224, 否则会报 shape错误.
    """
    def __init__(self, max_classes=100):
        super().__init__()
        base_vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = base_vit.heads.head.in_features
        base_vit.heads.head = nn.Linear(in_features, max_classes)
        self.vit = base_vit

    def forward(self, x):
        return self.vit(x)