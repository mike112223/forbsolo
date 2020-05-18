import torch
import numpy as np


def get_center_regions(bboxes, inner_thres):

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    w = inner_thres * (x2 - x1 + 1)
    h = inner_thres * (y2 - y1 + 1)
    x, y = (x1 + x2) / 2, (y1 + y2) / 2

    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2

    center_regions = torch.stack([x1, y1, x2, y2], dim=-1)

    return center_regions


def get_grids(shape, grid_number, device='cuda'):

    stride_x = shape[1] / grid_number
    stride_y = shape[0] / grid_number
    shift_x = np.arange(0, grid_number) * stride_x + stride_x / 2
    shift_y = np.arange(0, grid_number) * stride_y + stride_y / 2
    pairs = np.meshgrid(shift_x, shift_y)

    # centers = np.column_stack([pairs[0].reshape(-1, 1), pairs[1].reshape(-1, 1)])
    x1s = (pairs[0] - stride_x / 2).reshape(-1, 1)
    x2s = (pairs[0] + stride_x / 2).reshape(-1, 1)
    y1s = (pairs[1] - stride_y / 2).reshape(-1, 1)
    y2s = (pairs[1] + stride_y / 2).reshape(-1, 1)

    grids = np.column_stack([x1s, y1s, x2s, y2s])

    if device == 'cuda':
        grids = torch.from_numpy(grids)
        # centers = torch.from_numpy(centers)

    return grids


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)

    x1s = torch.max(bboxes1[:, None, 0], bboxes2[:, 0])
    y1s = torch.max(bboxes1[:, None, 1], bboxes2[:, 1])
    x2s = torch.min(bboxes1[:, None, 2], bboxes2[:, 2])
    y2s = torch.min(bboxes1[:, None, 3], bboxes2[:, 3])

    ws = torch.max(torch.tensor(0.0), x2s - x1s + 1)
    hs = torch.max(torch.tensor(0.0), y2s - y1s + 1)

    intersections = ws * hs

    if mode == 'iou':
        ious = intersections / (areas1[:, None] + areas2 - intersections)
    else:
        ious = intersections / (areas1[:, None])

    return ious

def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):

    pxs = (proposals[:, 0] + proposals[:, 2]) / 2
    pys = (proposals[:, 1] + proposals[:, 3]) / 2
    pws = (proposals[:, 2] - proposals[:, 0]) + 1
    phs = (proposals[:, 3] - proposals[:, 1]) + 1

    gxs = (gt[:, 0] + gt[:, 2]) / 2
    gys = (gt[:, 1] + gt[:, 3]) / 2
    gws = (gt[:, 2] - gt[:, 0]) + 1
    ghs = (gt[:, 3] - gt[:, 1]) + 1

    txs = (gxs - pxs) / pws
    tys = (gys - pys) / phs
    tws = torch.log(gws / pws)
    ths = torch.log(ghs / phs)

    deltas = torch.stack([txs, tys, tws, ths], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)

    deltas = (deltas - means) / stds

    return deltas
