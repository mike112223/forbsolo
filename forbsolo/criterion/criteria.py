import torch.nn as nn
import torch.nn.functional as F

from .builder import build_loss
from .registry import CRITERIA

from forbsolo.model.utils import multi_apply


@CRITERIA.register_module
class Criteria(nn.Module):
    def __init__(self, cls_loss, seg_loss, num_classes):
        super(Criteria, self).__init__()
        self.cls_loss = build_loss(cls_loss)
        self.seg_loss = build_loss(seg_loss)
        self.num_classes = num_classes

    def forward(self, pred_results, target_results):

        # list of level
        cate_labels, ins_masks, num_total_pos, num_total_neg = target_results

        cls_scores, pred_masks = pred_results

        seg_losses, cls_losses = multi_apply(
            self.forward_single,
            cls_scores,
            pred_masks,
            cate_labels,
            ins_masks,
            num_total_pos=num_total_pos,
        )

        return seg_losses, cls_losses

    def forward_single(self,
                       cls_score,
                       pred_mask,
                       cate_label,
                       ins_mask,
                       num_total_pos):
        # seg loss
        seg_loss = self.seg_loss(pred_mask, ins_mask, avg_factor=num_total_pos)

        # cls loss
        cate_label = cate_label.reshape(-1)
        cate_label = F.one_hot(cate_label, self.num_classes + 1)[:, 1:]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        cls_loss = self.cls_loss(cls_score, cate_label, avg_factor=num_total_pos)

        return seg_loss, cls_loss