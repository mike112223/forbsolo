from .base import _Iter_LRScheduler
from ..registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class StepLR(_Iter_LRScheduler):

    def __init__(self, lr_step, gamma=0.1, **kwargs):
        assert isinstance(lr_step, (list, int))
        if isinstance(lr_step, list):
            for s in lr_step:
                assert isinstance(s, int) and s > 0
        elif isinstance(lr_step, int):
            assert lr_step > 0
        else:
            raise TypeError('"lr_step" must be a list or integer')
        self.lr_step = lr_step
        self.gamma = gamma
        super(StepLR, self).__init__(**kwargs)

    def get_lr(self):

        progress = self.last_epoch

        if isinstance(self.lr_step, int):
            return [base_lr * (self.gamma**(progress // self.lr_step)) for base_lr in self.base_lrs]

        exp = len(self.lr_step)
        for i, s in enumerate(self.lr_step):
            if progress < s:
                exp = i
                break

        multiplier = self.gamma**exp

        return [base_lr * multiplier for base_lr in self.base_lrs]
