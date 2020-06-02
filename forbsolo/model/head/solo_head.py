import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import HEADS
from ..utils import (ConvModule, bias_init_with_prob,
                     normal_init, multi_apply)


@HEADS.register_module
class SOLOHead(nn.Module):
    """
    An anchor-based head used in [1]_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes - 1)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 num_inputs,
                 in_channels,
                 feat_channels=256,
                 grid_numbers=[40, 36, 24, 16, 12],
                 fpn_strides=[4, 8, 16, 32, 64],
                 strides=[4, 8, 16, 32, 64],
                 stacked_convs=7,
                 inner_thres=0.2,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(SOLOHead, self).__init__()

        self.num_classes = num_classes
        self.num_inputs = num_inputs
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.grid_numbers = grid_numbers
        self.fpn_strides = fpn_strides
        self.strides = strides
        self.stacked_convs = stacked_convs
        self.inner_thres = inner_thres
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        self.solo_masks = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        for i in range(self.num_inputs):
            self.solo_masks.append(
                nn.Conv2d(self.feat_channels, self.grid_numbers[i] ** 2, 1)
            )

        self.solo_cate = nn.Conv2d(
            self.feat_channels, self.num_classes, 3, padding=1)

    def init_weights(self):
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cls)

        for m in self.solo_masks:
            normal_init(m, std=0.01)

    @staticmethod
    def spatial_info_encode(feat):
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
        y_range = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feat.shape[0], 1, -1, -1])
        x = x.expand([feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        feat = torch.cat([feat, coord_feat], 1)
        return feat

    def forward_single(self, x, i):
        # forward each levels
        cate_feat = x
        cate_feat = F.interpolate(
            cate_feat,
            size=(self.grid_numbers[i], self.grid_numbers[i]),
            mode='bilinear'
        )
        mask_feat = self.spatial_info_encode(x)

        for cate_conv in self.cate_convs:
            cate_feat = cate_conv(cate_feat)
        for mask_conv in self.mask_convs:
            mask_feat = mask_conv(mask_feat)

        cate = self.solo_cate(cate_feat)

        mask_feat = F.interpolate(
            mask_feat, scale_factor=2, mode='bilinear')
        mask = self.solo_masks[i](mask_feat)
        return cate, mask

    def forward(self, xs):

        xs = self.rescale_feat(xs)

        # apply level
        cls_scores, masks = multi_apply(self.forward_single, xs, range(self.num_inputs))

        return cls_scores, masks

    def rescale_feat(self, xs):
        feats = []
        base_featmap_size = np.asarray(xs[0].shape[-2:], dtype=np.int32)
        for i, x in enumerate(xs):
            feats.append(F.interpolate(
                xs[i],
                size=tuple(base_featmap_size * self.fpn_strides[0] // self.strides[i]),
                mode='bilinear'
            ))

        return tuple(feats)
