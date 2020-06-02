from collections.abc import Sequence

import cv2
import numpy as np
import torch

from forbsolo.utils.misc import is_str

from ...registry import TRANSFORMS


CV2_MODE = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


@TRANSFORMS.register_module
class Resize(object):
    def __init__(self, img_scale, keep_ratio, mode='bilinear'):
        self.img_scale = img_scale
        if not isinstance(self.img_scale, tuple):
            raise TypeError('img scale must be tuple!')
        elif len(self.img_scale) != 2:
            raise ValueError('img scale must contains w and h!')

        self.keep_ratio = keep_ratio
        self.mode = CV2_MODE[mode]

    def __resize_img(self, img, interpolation, return_scale=False):
        h, w = img.shape[:2]
        if self.keep_ratio:

            max_long_edge = max(self.img_scale)
            max_short_edge = min(self.img_scale)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))

            # scale_factor = min(self.img_scale[0] / w, self.img_scale[1] / h)
            new_size = (
                int(w * float(scale_factor) + 0.5),
                int(h * float(scale_factor) + 0.5)
            )
            resized_img = cv2.resize(img, new_size, interpolation=interpolation)
        else:
            resized_img = cv2.resize(
                img, self.img_scale, interpolation=interpolation
            )
            w_scale = self.img_scale[0] / w
            h_scale = self.img_scale[1] / h

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)

        if return_scale:
            return resized_img, scale_factor
        else:
            return resized_img

    def _resize_img(self, results):
        img = results['img']
        resized_img, scale_factor = self.__resize_img(img, self.mode, True)

        results['img'] = resized_img
        results['scale'] = self.img_scale
        results['img_shape'] = resized_img.shape
        results['pad_shape'] = resized_img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            masks = [self.__resize_img(mask, CV2_MODE['nearest']) for mask in results[key]]
            results[key] = np.stack(masks)


    def __call__(self, results):
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        return results


@TRANSFORMS.register_module
class RandomFlip(object):
    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __flip_img(self, img):
        if self.direction == 'horizontal':
            flipped_img = np.flip(img, axis=1)
        else:
            flipped_img = np.flip(img, axis=0)
        return flipped_img

    def _flip_img(self, results):
        img = results['img']
        results['img'] = self.__flip_img(img)

    def _flip_bboxes(self, results):
        h, w = results['img_shape'][:2]
        for key in results.get('bbox_fields', []):
            bboxes = results[key]

            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
                flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
            else:
                flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
                flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
            results[key] = flipped

    def _flip_masks(self, results):
        for key in results.get('mask_fields', []):
            results[key] = np.stack([
                self.__flip_img(mask) for mask in results[key]
            ])

    def __call__(self, results):
        if np.random.random() < self.flip_ratio:
            results['flip'] = True
            results['flip_direction'] = self.direction
            self._flip_img(results)
            self._flip_bboxes(results)
            self._flip_masks(results)
        else:
            results['flip'] = False

        return results


@TRANSFORMS.register_module
class Normalize(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results['img'].astype(np.float32)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results['img'] = (img - self.mean) / self.std

        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results


@TRANSFORMS.register_module
class Pad(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __pad_img(self, img):
        h, w = img.shape[:2]
        if self.size:
            pad_h = self.size[1] - h
            pad_w = self.size[0] - w
        elif self.size_divisor:
            pad_h = int(np.ceil(h / self.size_divisor) * self.size_divisor) - h
            pad_w = int(np.ceil(w / self.size_divisor) * self.size_divisor) - w

        assert pad_h >= 0 and pad_w >= 0

        if len(img.shape) == 3:
            padding = np.array([[0, pad_h], [0, pad_w], [0, 0]])
        else:
            padding = np.array([[0, pad_h], [0, pad_w]])

        padded_img = np.pad(
            img,
            padding,
            'constant',
            constant_values=self.pad_val
        )
        return padded_img

    def _pad_img(self, results):
        img = results['img']
        padded_img = self.__pad_img(img)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        for key in results.get('mask_fields', []):
            results[key] = np.stack([
                self.__pad_img(mask) for mask in results[key]
            ])

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results


@TRANSFORMS.register_module
class Collect(object):
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}

        if 'img' in results:
            img = results['img']
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = to_tensor(img)

        for key in ['gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_masks']:
            if key not in results:
                continue
            results[key] = to_tensor(results[key])

        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_meta'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data
