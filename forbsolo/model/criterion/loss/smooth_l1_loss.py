import torch
import torch.nn as nn

from ...registry import LOSSES


@LOSSES.register_module
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        assert reduction in ['mean', 'sum']
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None):

        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                           diff - 0.5 * self.beta)
        loss *= self.loss_weight

        if weight is not None:
            loss *= weight

        if self.reduction == 'mean' and avg_factor is not None:
            return loss.sum() / avg_factor
        else:
            return loss.sum()
