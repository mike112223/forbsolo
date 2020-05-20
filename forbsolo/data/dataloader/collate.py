# modify from mmcv collate

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch._six import container_abcs


def collate(batch, collate_key='img'):

    if isinstance(batch[0], container_abcs.Mapping):
        elem = batch[0]
        tmp = {}
        for key in elem:
            if key in ('img', 'gt_masks'):
                tmp[key] = collate([d[key] for d in batch], collate_key=key)
            else:
                tmp[key] = [d[key] for d in batch]

        return tmp
    elif isinstance(batch[0], torch.Tensor):
        # only work for img (C, H, W)
        if len(batch) == 1:
            return default_collate(batch)
        else:
            padded_samples = []
            for i in range(len(batch)):
                ndim = batch[i].dim()
                assert ndim == 3
                # (W, H)
                max_shape = [0, 0]
                for dim in range(1, 3):
                    max_shape[dim - 1] = batch[i].size(-dim)
                for sample in batch:
                    for dim in range(1, 3):
                        max_shape[dim - 1] = max(max_shape[dim - 1],
                                                 sample.size(-dim))

            for sample in batch:
                # 2 sides * 2 coors
                pad = [0 for _ in range(2 * 2)]
                for dim in range(1, 3):
                    pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                padded_samples.append(
                    F.pad(sample.data, pad, value=0))

            if collate_key == 'img':
                return default_collate(padded_samples)
            elif collate_key == 'gt_masks':
                return padded_samples
            else:
                raise ValueError('Not support this collate key!')

    else:
        return default_collate(batch)
