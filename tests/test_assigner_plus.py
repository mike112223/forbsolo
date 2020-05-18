
import numpy as np
import cv2
import torch

from forbsolo.model.utils import get_grids, get_center_regions

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


grid_numbers = [40, 36, 24, 16, 12]
scales = [[0, 96], [48, 192], [96, 384], [192, 768], [384, -1]]
inner_thres = 0.2

gt_bboxes_list = batch['gt_bboxes']
gt_labels_list = batch['gt_labels']
img_meta_list = batch['img_meta']
img_list = batch['img']

for j in range(len(gt_bboxes_list)):

    gt_bboxes = gt_bboxes_list[j]
    gt_labels = gt_labels_list[j]
    img_meta = img_meta_list[j]
    img = img_list[j].numpy().transpose((1, 2, 0))
    print(img.shape)

    # shape: (num_gt_box, 4)
    center_regions = get_center_regions(gt_bboxes, inner_thres)

    for k in range(len(gt_bboxes)):
        x1, y1, x2, y2 = gt_bboxes.numpy()[k]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, x2, y2 = center_regions.numpy()[k]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('b', img)
    cv2.waitKey(0)

    # process each level
    for i in range(len(grid_numbers)):
        cimg = img.get()

        grid_number = grid_numbers[i]
        scale = scales[i]

        ws = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        hs = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        areas = torch.sqrt(ws * hs)

        valid_scale_flag = (areas >= scale[0]) * (areas < scale[1])

        # shape: (grid_number^2, 2)
        grids = get_grids(img_meta['pad_shape'][:2], grid_number)

        grid2label = gt_labels.new_tensor([0] * grids.shape[0], dtype=torch.long)
        grid2gt = gt_labels.new_tensor([-1] * grids.shape[0], dtype=torch.long)

        pos_inds, neg_inds = cls_assign(center_regions, grids)

        if len(pos_inds):
            grid2label[pos_inds[:, 0]] = gt_labels[pos_inds[:, 1]]
            grid2gt[pos_inds[:, 0]] = pos_inds[:, 1]

        for i, point in enumerate(grids.numpy().astype(np.int32)):
            c = (0, 0, 255) if i in pos_inds[:, 0] else (255, 255, 0)
            x1, y1, x2, y2 = point
            cimg = cv2.rectangle(cimg, (x1, y1), (x2, y2), c, 1)
            # cv2.circle(cimg, tuple(point), 1, (0, 0, 255), 2)

        cv2.imshow('a', cimg)
        cv2.waitKey(0)
