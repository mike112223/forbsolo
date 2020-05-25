
from functools import partial

from torch.utils.data import DataLoader

from .collate import collate

from ..builder import build_sampler
from ..registry import DATALOADERS


@DATALOADERS.register_module
class BaseDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 sampler,
                 samples_per_gpu,
                 num_gpu,
                 shuffle=True,
                 **kwarg):

        if shuffle:
            sampler = build_sampler(
                sampler,
                dict(dataset=dataset,
                     samples_per_gpu=samples_per_gpu)
            )
        else:
            sampler = None

        batch_size = num_gpu * samples_per_gpu

        super().__init__(
            dataset,
            batch_size=batch_size,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            sampler=sampler,
            pin_memory=False,
            **kwarg)
