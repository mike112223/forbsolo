from forbsolo.utils import build_from_cfg

from .registry import BACKBONES, NECKS, HEADS, ASSIGNERS, MODELS


def build_backbone(cfg, default_args=None):
    backbone = build_from_cfg(cfg, BACKBONES, default_args)
    return backbone


def build_neck(cfg, default_args=None):
    neck = build_from_cfg(cfg, NECKS, default_args)
    return neck


def build_head(cfg, default_args=None):
    head = build_from_cfg(cfg, HEADS, default_args)
    return head


def build_assigner(cfg, default_args=None):
    head = build_from_cfg(cfg, ASSIGNERS, default_args)
    return head


def build_model(cfg, default_args=None):
    model = build_from_cfg(cfg, MODELS, default_args)
    return model
