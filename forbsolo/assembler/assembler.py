import os

import torch
from torch import nn

from forbsolo.utils import Config
from forbsolo.data import build_dataset, build_transform, build_dataloader, collate
from forbsolo.model import build_model
from forbsolo.criterion import build_criterion
from forbsolo.optimizer import build_optimizer, build_lr_scheduler
from forbsolo.runner import build_runner


def assemble(cfg_fp, checkpoint='', test_mode=False):

    # 1.logging
    print('build config!')
    cfg = Config.fromfile(cfg_fp)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    # 2. data
    # 2.1 dataset
    print('build data!')
    train_tf = build_transform(cfg['data']['train']['transforms'])
    train_dataset = build_dataset(cfg['data']['train']['dataset'], dict(transforms=train_tf))

    if cfg['data'].get('val'):
        val_tf = build_transform(cfg['data']['val']['transforms'])
        val_dataset = build_dataset(cfg['data']['val']['dataset'], dict(transforms=val_tf))

    # 2.2 dataloader
    train_loader = build_dataloader(cfg['data']['train']['loader'], dict(dataset=train_dataset, collate_fn=collate))
    loader = {'train': train_loader}
    if cfg['data'].get('val'):
        val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset, collate_fn=collate))
        loader['val'] = val_loader

    # 3. model
    print('build model!')
    model = build_model(cfg['model'])
    if torch.cuda.is_available():
        gpu = True
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
    else:
        gpu = False

    # 4. criterion
    print('build loss!')
    criterion = build_criterion(cfg['criterion'])

    # 5. optim
    print('build optimizer!')
    optimizer = build_optimizer(
        cfg['optim']['optimizer'],
        dict(params=model.parameters())
    )
    lr_scheduler = build_lr_scheduler(
        cfg['optim']['lr_scheduler'],
        dict(optimizer=optimizer, niter_per_epoch=len(train_loader))
    )

    # 6. runner
    print('build runner!')
    runner = build_runner(
        cfg['runner'],
        dict(
            loader=loader,
            model=model,
            criterion=criterion,
            optim=optimizer,
            lr_scheduler=lr_scheduler,
            workdir=cfg['workdir'],
            gpu=gpu,
        )
    )

    return runner
