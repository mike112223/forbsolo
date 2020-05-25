from forbsolo.utils import build_from_cfg

from .registry import BACKBONES, NECKS, HEADS, GRIDS, CRITERIA, LOSSES, MODELS


def build_backbone(cfg, default_args=None):
    backbone = build_from_cfg(cfg, BACKBONES, default_args)
    return backbone


def build_neck(cfg, default_args=None):
    neck = build_from_cfg(cfg, NECKS, default_args)
    return neck


def build_head(cfg, default_args=None):
    head = build_from_cfg(cfg, HEADS, default_args)
    return head


def build_grid(cfg, default_args=None):
    grid = build_from_cfg(cfg, GRIDS, default_args)
    return grid


def build_criterion(cfg, default_args=None):
    criterion = build_from_cfg(cfg, CRITERIA, default_args)
    return criterion


def build_loss(cfg, default_args=None):
    loss = build_from_cfg(cfg, LOSSES, default_args)
    return loss


def build_model(cfg, default_args=None):
    model = build_from_cfg(cfg, MODELS, default_args)
    return model
