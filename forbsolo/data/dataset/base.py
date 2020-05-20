import cv2

import numpy as np
import pycocotools.mask as maskUtils

from torch.utils.data import Dataset

from ..registry import DATASETS


@DATASETS.register_module
class BaseDataset(Dataset):
    """
    out: results: {
        'img_meta':...,
        'img': (dtype:torch.float32, size:[3, h, w]),
        'gt_bboxes': (dtype:torch.float32, size:[k, 4]'
        'gt_labels: (dtype:torch.int64, size:[k])'
        }
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 transforms=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 with_bbox=True,
                 with_mask=True,
                 with_label=True):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.transforms = transforms
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_label = with_label

        # load annotations
        self.img_infos = self.load_annotations(self.ann_file)
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        pass

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _load_bboxes(self, results):
        results['bbox_fields'] = []

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        results['mask_fields'] = []

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results


    def __getitem__(self, idx):

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        results = self.process(results)

        return results

    def process(self, results):

        # load img
        filename = results['img_info']['filename']
        # print('img_name!', filename)
        img = cv2.imread(filename)
        results['img'] = img
        results['filename'] = filename
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        if self.with_bbox:
            results = self._load_bboxes(results)
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)

        if self.transforms:
            results = self.transforms(results)

        return results
