from forbsolo.utils import build_from_cfg

from .registry import LOSSES, CRITERIA


def build_loss(cfg, default_args=None):
    loss = build_from_cfg(cfg, LOSSES, default_args)
    return loss


def build_criterion(cfg, default_args=None):
    criterion = build_from_cfg(cfg, CRITERIA, default_args)
    return criterion
