import torch.optim as torch_optim
from forbsolo.utils import build_from_cfg

from .registry import LR_SCHEDULERS


def build_optimizer(cfg, default_args=None):
    optimizer = build_from_cfg(cfg, torch_optim, default_args, 'module')
    return optimizer

def build_lr_scheduler(cfg, default_args=None):
    lr_scheduler = build_from_cfg(cfg, LR_SCHEDULERS, default_args)
    return lr_scheduler
