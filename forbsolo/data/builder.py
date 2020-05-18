import copy

import torch.utils.data as torch_data

from forbsolo.utils import build_from_cfg
from .dataset.transform import Compose

from .dataset.dataset_wrappers import ConcatDataset
from .registry import DATASETS, TRANSFORMS


def _concat_dataset(cfg, default_args=None):
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    proposal_files = cfg.get('proposal_file', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataloader(cfg, default_args):
    loader = build_from_cfg(cfg, torch_data, default_args, 'module')
    return loader


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_transform(cfg):
    tfs = []
    for icfg in cfg:
        tf = build_from_cfg(icfg, TRANSFORMS)
        tfs.append(tf)
    aug = Compose(tfs)

    return aug
