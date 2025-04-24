# lunch_utils.py

import torch
import torch.nn as nn
import math

class KendallUncertaintyWeighting(nn.Module):
    """
    针对多项损失 L1, L2, ... 引入多个 log_sigma_i,
    total_loss = sum_i [ 1/(2*sigma_i^2) * Li + log_sigma_i ]
    这里为了灵活，可在 forward(loss_dict) 里指定要加权的多个loss key
    """
    def __init__(self, loss_keys):
        """
        :param loss_keys: list of string, e.g. ["new", "old", "mse"]
        """
        super().__init__()
        self.loss_keys = loss_keys
        # 为每个loss key 引入 log_sigma
        self.log_sigmas = nn.ParameterDict()
        for k in loss_keys:
            # 初始化 log_sigma=0 => sigma=1
            self.log_sigmas[k] = nn.Parameter(torch.zeros(1))

    def forward(self, loss_dict):
        """
        loss_dict: { "new": tensor, "old": tensor, ... }
        返回 total_loss
        """
        total_loss = 0.0
        for k in self.loss_keys:
            Lk = loss_dict.get(k, None)
            if Lk is None:
                continue
            # 1/(2*sigma^2)*Lk + log_sigma
            sigma = torch.exp(self.log_sigmas[k])
            # or sigma^2 => torch.exp(2*log_sigma)
            weighted = (0.5 / (sigma*sigma)) * Lk + self.log_sigmas[k]
            total_loss += weighted
        return total_loss


class ExponentialMovingAverage:
    """
    指数滑动平均, 存储在 CPU, 以避免GPU额外显存
    """
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.ema_state = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.ema_state[name] = param.detach().cpu().clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.ema_state:
                    # 新出现参数(如动态添加), 这里不考虑, 或同样init
                    self.ema_state[name] = param.detach().cpu().clone()
                else:
                    param_cpu = param.detach().cpu()
                    self.ema_state[name].mul_(self.decay).add_((1.0-self.decay)*param_cpu)

    def get_state_dict(self, model: nn.Module):
        """
        把 CPU 的 ema参数拷到 model对应device
        """
        sd = model.state_dict()
        new_sd = {}
        for k,v in sd.items():
            if k in self.ema_state:
                new_sd[k] = self.ema_state[k].to(v.device)
            else:
                new_sd[k] = v
        return new_sd

    def get_memory_overhead(self):
        # 计算self.ema_state中元素总数
        total_numel=0
        for t_cpu in self.ema_state.values():
            total_numel += t_cpu.numel()
        overhead_bytes = total_numel*4  # float32 => 4 bytes
        return overhead_bytes