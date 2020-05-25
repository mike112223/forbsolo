import torch.nn as nn
import torch.nn.functional as F

from ...registry import LOSSES


@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert reduction in ['mean', 'sum']
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None):

        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight

        loss *= self.loss_weight

        if weight is not None:
            loss *= weight

        if self.reduction == 'mean' and avg_factor is not None:
            return loss.sum() / avg_factor
        else:
            return loss.sum()
