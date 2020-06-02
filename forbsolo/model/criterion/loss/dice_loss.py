import torch
import torch.nn as nn

from ...registry import LOSSES


@LOSSES.register_module
class DiceLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, epsilon=1e-3):
        super(DiceLoss, self).__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.epsilon = epsilon

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None):

        pred = pred.sigmoid()
        target = target.type_as(pred)

        dice = (2 * torch.sum(pred * target, 1) /
                ((torch.sum(pred ** 2, 1) + torch.sum(target ** 2, 1)) + 2 * self.epsilon))

        loss = (1 - dice) * self.loss_weight

        if weight is not None:
            loss *= weight

        if self.reduction == 'mean' and avg_factor is not None:
            return loss.sum() / avg_factor
        else:
            return loss.sum()
