import os
import sys

sys.path.insert(0, os.path.abspath('../forbsolo'))

import torch
import torch.nn as nn

from forbsolo.utils import Config
from forbsolo.data import build_dataset, build_transform, build_dataloader
from forbsolo.model import build_model


def main():
    cfg_fp = os.path.join(os.path.abspath('config'), 'solo_r50_fpn_multi_gpu.py')
    cfg = Config.fromfile(cfg_fp)

    val_tf = build_transform(cfg['data']['val']['transforms'])
    val_dataset = build_dataset(cfg['data']['val']['dataset'], dict(transforms=val_tf))

    val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset))

    device_ids = range(torch.cuda.device_count())

    model = build_model(cfg['model']).cuda()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model.cuda()

    for i, batch in enumerate(val_loader):
        break

    for k in batch.keys():
        print(type(batch[k]))
        if isinstance(batch[k], (list, tuple)):
            for i in batch[k]:
                print('inner_lt', type(i))
        elif isinstance(batch[k], dict):
            for kk in batch[k].keys():
                print('inner_dict', type(batch[k][kk]))

    inputs = nn.parallel.scatter(batch, device_ids)

if __name__ == '__main__':
    main()
