
import torch.nn as nn

from .registry import MODELS
from .builder import build_backbone, build_neck, build_head, build_grid


@MODELS.register_module
class SOLO(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 grid,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLO, self).__init__()

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.grid = build_grid(grid)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_masks,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)

        # cls_scores (list of levels, default len is 5 (p3-p7))
        # cls_score (each level, shape: [batch(imgs), num_classes * anchors, h, w])
        # similarly， bbox_preds
        cls_scores, masks = self.head(x)

        featmap_sizes = [featmap.size()[-2:] for featmap in masks]

        targets_results = self.grid.get_target(
            gt_bboxes,
            gt_masks,
            img_metas,
            gt_bboxes_ignore,
            gt_labels,
            featmap_sizes,
        )

        (grid2labels_list, grid2gt2_list, num_total_pos,
            num_total_neg) = targets_results

        pred_results = (cls_scores, masks)

        return pred_results, targets_results

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    # def simple_test(self, img, img_meta, rescale=False):
    #     x = self.extract_feat(img)
    #     outs = self.head(x)
    #     bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    #     bbox_list = self.head.get_bboxes(*bbox_inputs)
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.head.num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results[0]

    def forward(self, img, img_meta, test_mode=False, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if test_mode:
            return self.forward_test(img, img_meta, **kwargs)
        else:
            return self.forward_train(img, img_meta, **kwargs)
