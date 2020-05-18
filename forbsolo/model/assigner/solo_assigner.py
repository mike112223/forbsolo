import torch
import numpy as np

from ..registry import ASSIGNERS
from ..utils import multi_apply, get_center_regions, get_grids


@ASSIGNERS.register_module
class SOLOAssigner(object):
    def __init__(self,
                 grid_numbers=[40, 36, 24, 16, 12],
                 scales=[[0, 96], [48, 192], [96, 384], [192, 768], [384, -1]],
                 inner_thres=0.2):
        super(SOLOAssigner, self).__init__()

        self.grid_numbers = grid_numbers
        self.scales = scales
        self.inner_thres = inner_thres

    @staticmethod
    def cls_assign(center_regions, grids, valid_scale_flag):
        '''
            shape: center_regions (num_gt_box, 4)
                   grids (grid_number^2, 2)
        '''
        grids = grids.type_as(center_regions)

        gx1s = grids[:, 0].unsqueeze(dim=1)
        gy1s = grids[:, 1].unsqueeze(dim=1)
        gx2s = grids[:, 2].unsqueeze(dim=1)
        gy2s = grids[:, 3].unsqueeze(dim=1)

        rx1s = center_regions[:, 0]
        ry1s = center_regions[:, 1]
        rx2s = center_regions[:, 2]
        ry2s = center_regions[:, 3]

        x1_judge = gx1s - rx1s
        y1_judge = gy1s - ry1s
        x2_judge = gx2s - rx2s
        y2_judge = gy2s - ry2s

        assigner = (x1_judge > 0) * (y1_judge > 0) * (x2_judge < 0) * (y2_judge < 0)
        pos_ind = (assigner * valid_scale_flag).nonzero()
        neg_ind = (assigner.sum(dim=1) == 0).nonzero()

        return pos_ind, neg_ind

    @staticmethod
    def images_to_levels(target, num_level_anchors):
        """Convert targets by image to targets by feature level.

        [target_img0, target_img1] -> [target_level0, target_level1, ...]
        """
        target = torch.stack(target, 0)
        level_targets = []
        start = 0
        for n in num_level_anchors:
            end = start + n
            level_targets.append(target[:, start:end].squeeze(0))
            start = end
        return level_targets

    def get_target_single(self,
                          gt_bboxes,
                          gt_masks,
                          gt_labels,
                          img_meta,
                          gt_bboxes_ignore):
        '''
        get target for each img
        '''
        # shape: (num_gt_box, 4)
        center_regions = get_center_regions(gt_bboxes, self.inner_thres)

        # process each level
        for i in range(len(self.grid_numbers)):

            grid_number = self.grid_numbers[i]
            scale = self.scales[i]

            ws = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            hs = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            areas = torch.sqrt(ws * hs)

            valid_scale_flag = (areas >= scale[0]) * (areas < scale[1])

            # shape: (grid_number^2, 2)
            grids = get_grids(img_meta['pad_shape'][:2], grid_number)

            grid2label = gt_labels.new_tensor([0] * grids.shape[0], dtype=torch.long)
            grid2gt = gt_labels.new_tensor([-1] * grids.shape[0], dtype=torch.long)

            pos_inds, neg_inds = self.cls_assign(center_regions, grids, valid_scale_flag)

            if len(pos_inds):
                grid2label[pos_inds[:, 0]] = gt_labels[pos_inds[:, 1]]
                grid2gt[pos_inds[:, 0]] = pos_inds[:, 1]




    def get_target(self,
                   gt_bboxes_list,
                   gt_masks_list,
                   gt_labels_list,
                   img_metas,
                   gt_bboxes_ignore_list=None):

        num_imgs = len(img_metas)

        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        # apply img
        target_results = multi_apply(
            self.get_targets_single,
            gt_bboxes_list,
            gt_masks_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list
        )
