import torch
import torch.nn.functional as F

from ..registry import GRIDS
from ..utils import multi_apply, get_center_regions, get_grids


@GRIDS.register_module
class SOLOGrid(object):
    def __init__(self,
                 grid_numbers=[40, 36, 24, 16, 12],
                 strides=[4, 8, 16, 32, 64],
                 scales=[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
                 inner_thres=0.2):
        super(SOLOGrid, self).__init__()

        self.grid_numbers = grid_numbers
        self.strides = strides
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

        x1_judge = gx2s - rx1s
        y1_judge = gy2s - ry1s
        x2_judge = gx1s - rx2s
        y2_judge = gy1s - ry2s

        assigner = (x1_judge > 0) * (y1_judge > 0) * \
            (x2_judge < 0) * (y2_judge < 0) * valid_scale_flag

        pos_ind = assigner.nonzero()
        neg_ind = (assigner.sum(dim=1) == 0).nonzero()

        return pos_ind, neg_ind

    @staticmethod
    def images_to_levels(target):
        """Convert targets by image to targets by feature level.
        [[target_img0_level0, ...], [target_img1_level0, ...]] -> [target_level0, target_level1, ...]
        """
        num_img = len(target)
        num_level = len(target[0])
        level_targets = []
        for l in range(num_level):
            img_targets = []
            for i in range(num_img):
                img_targets.append(target[i][l])
            level_targets.append(torch.stack(img_targets, 0))

        return level_targets

    def get_target_single(self,
                          gt_bboxes,
                          gt_masks,
                          img_meta,
                          gt_bboxes_ignore,
                          gt_labels,
                          featmap_sizes):
        '''
        get target for each img
        '''
        cate_labels = []
        ins_masks = []
        num_pos = 0
        num_neg = 0

        # shape: (num_gt_box, 4)
        center_regions = get_center_regions(gt_masks, gt_bboxes, self.inner_thres)

        ws = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        hs = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        areas = torch.sqrt(ws * hs)

        # process each level
        for i in range(len(self.grid_numbers)):

            rescaled_gt_masks = F.interpolate(
                gt_masks.float().unsqueeze(0),
                scale_factor=1. / (self.strides[i] / 2),
                mode='nearest').byte().squeeze(0)

            grid_number = self.grid_numbers[i]
            scale = self.scales[i]
            featmap_size = featmap_sizes[i]

            ins_mask = rescaled_gt_masks.new_zeros(
                [grid_number ** 2, featmap_size[0], featmap_size[1]],
                dtype=torch.uint8)

            grid2label = gt_labels.new_tensor([0] * grid_number ** 2, dtype=torch.long)

            valid_scale_flag = (areas >= scale[0]) * (areas <= scale[1])

            # shape: (grid_number^2, 2 (4))
            grids = get_grids(img_meta['pad_shape'][:2], grid_number, center=False)

            # pos_inds [grid_idx, gt_idx]
            pos_inds, neg_inds = self.cls_assign(center_regions, grids, valid_scale_flag)

            if len(pos_inds):
                grid2label[pos_inds[:, 0]] = gt_labels[pos_inds[:, 1]]
                ins_mask[pos_inds[:, 0]] = rescaled_gt_masks[pos_inds[:, 1]]

            cate_labels.append(grid2label.reshape(grid_number, grid_number))
            ins_masks.append(ins_mask)
            num_pos += len((grid2label).nonzero())
            num_neg += len(neg_inds)

        return cate_labels, ins_masks, num_pos, num_neg

    def get_target(self,
                   gt_bboxes_list,
                   gt_masks_list,
                   img_metas,
                   gt_bboxes_ignore_list,
                   gt_labels_list,
                   featmap_sizes):

        num_imgs = len(img_metas)

        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        # apply img
        all_cate_labels, all_ins_masks, num_pos, num_neg = multi_apply(
            self.get_target_single,
            gt_bboxes_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
            gt_labels_list,
            featmap_sizes=featmap_sizes
        )

        num_total_pos = sum(num_pos)
        num_total_neg = sum(num_neg)

        cate_labels_list = self.images_to_levels(all_cate_labels)
        ins_masks_list = self.images_to_levels(all_ins_masks)

        return cate_labels_list, ins_masks_list, num_total_pos, num_total_neg
