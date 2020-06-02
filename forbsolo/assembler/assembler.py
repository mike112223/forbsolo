import os
import random

import torch
import numpy as np

from forbsolo.utils import Config
from forbsolo.data import build_dataset, build_transform, build_dataloader
from forbsolo.model import build_model
from forbsolo.optimizer import build_optimizer, build_lr_scheduler
from forbsolo.runner import build_runner

from forbsolo.parallel import SOLODataParallel


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def assemble(cfg_fp, checkpoint='', test_mode=False):

    # 1.logging
    print('build config!')
    cfg = Config.fromfile(cfg_fp)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    seed = cfg.get('seed', None)
    deterministic = cfg.get('deterministic', False)
    if seed is not None:
        set_random_seed(seed, deterministic)

    # 2. data
    # 2.1 dataset
    print('build data!')
    train_tf = build_transform(cfg['data']['train']['transforms'])
    train_dataset = build_dataset(cfg['data']['train']['dataset'], dict(transforms=train_tf))

    if cfg['data'].get('val'):
        val_tf = build_transform(cfg['data']['val']['transforms'])
        val_dataset = build_dataset(cfg['data']['val']['dataset'], dict(transforms=val_tf))

    # 2.2 dataloader
    train_loader = build_dataloader(cfg['data']['train']['loader'], dict(dataset=train_dataset))
    loader = {'train': train_loader}
    if cfg['data'].get('val'):
        val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset))
        loader['val'] = val_loader

    # 3. model
    print('build model!')
    model = build_model(cfg['model'])
    if torch.cuda.is_available():
        gpu = True
        model = SOLODataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.cuda()
    else:
        gpu = False

    # 4. optim
    print('build optimizer!')
    optimizer = build_optimizer(
        cfg['optim']['optimizer'],
        dict(params=model.parameters())
    )
    lr_scheduler = build_lr_scheduler(
        cfg['optim']['lr_scheduler'],
        dict(optimizer=optimizer, niter_per_epoch=len(train_loader))
    )

    # 5. runner
    print('build runner!')
    runner = build_runner(
        cfg['runner'],
        dict(
            loader=loader,
            model=model,
            optim=optimizer,
            lr_scheduler=lr_scheduler,
            workdir=cfg['workdir'],
            gpu=gpu,
        )
    )

    return runner
