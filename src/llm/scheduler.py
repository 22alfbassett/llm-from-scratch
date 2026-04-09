"""Learning-rate schedulers."""

import math


class CosineWithWarmup:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
