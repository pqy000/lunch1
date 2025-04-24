# memory.py

import torch
import random

class MemoryBuffer:
    """
    存储 (img, label, [logits]) 用于回放.
    """
    def __init__(self, max_size=2000, store_logits=False):
        self.max_size = max_size
        self.store_logits = store_logits
        self.samples = []

    def add_samples(self, x, y, logits=None):
        bsz = x.size(0)
        for i in range(bsz):
            if self.store_logits and logits is not None:
                self.samples.append((x[i].cpu(), y[i].item(), logits[i].cpu()))
            else:
                self.samples.append((x[i].cpu(), y[i].item()))
        # 裁剪
        if len(self.samples) > self.max_size:
            self.samples = random.sample(self.samples, self.max_size)

    def get_batch(self, batch_size=32):
        if len(self.samples) == 0:
            return None
        mb = random.sample(self.samples, min(batch_size, len(self.samples)))
        if self.store_logits and len(mb[0])==3:
            x_mb = [m[0] for m in mb]
            y_mb = [m[1] for m in mb]
            l_mb = [m[2] for m in mb]
            x_mb = torch.stack(x_mb, dim=0)
            y_mb = torch.tensor(y_mb, dtype=torch.long)
            l_mb = torch.stack(l_mb, dim=0)
            return x_mb, y_mb, l_mb
        else:
            x_mb = [m[0] for m in mb]
            y_mb = [m[1] for m in mb]
            x_mb = torch.stack(x_mb, dim=0)
            y_mb = torch.tensor(y_mb, dtype=torch.long)
            return x_mb, y_mb, None