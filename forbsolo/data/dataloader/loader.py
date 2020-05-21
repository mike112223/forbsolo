
from torch.utils.data import DataLoader

from .collate import collate

from ..builder import build_sampler
from ..registry import DATALOADERS


@DATALOADERS.register_module
class BaseDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 sampler,
                 batch_size,
                 shuffle=True,
                 **kwarg):

        if shuffle:
            sampler = build_sampler(
                sampler,
                dict(dataset=dataset,
                     samples_per_gpu=batch_size)
            )
        else:
            sampler = None

        super().__init__(
            dataset,
            batch_size=batch_size,
            collate_fn=collate,
            sampler=sampler,
            pin_memory=False,
            **kwarg)
