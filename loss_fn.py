from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    支持软标签的交叉熵损失。
    可传入 hard label（整数）+ weight（lam），内部自动转换为软标签。
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            x:      (B, C) - logits
            target: (B,) int 或 (B, C) soft label
            weight: float - 用于 lam 加权（可选）
        """
        if target.shape != x.shape:
            num_classes = x.shape[-1]
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            t = target.long().view(-1, 1)
            target = torch.full(
                (t.size(0), num_classes), off_value,
                device=x.device, dtype=x.dtype
            ).scatter_(1, t, on_value)

        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        if weight is not None:
            loss = loss * weight
        return loss.mean()
