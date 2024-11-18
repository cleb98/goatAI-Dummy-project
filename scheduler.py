import math
from typing import Any
from typing import Dict

from torch.optim import Optimizer


class LRScheduler(object):
    """
    Half-cycle cosine decay scheduler with warmup.
    """


    def __init__(self, optimizer, max_lr, final_lr, dl_len, epochs, warmup):
        # type: (Optimizer, float, float, int, int, int) -> None
        """
        :param optimizer: wrapped optimizer
        :param max_lr: maximum learning rate value
        :param final_lr: final learning rate value
        :param dl_len: dataloader length -> number of step per epoch
        :param epochs: training length -> max epoch
        :param warmup: number of warmup epochs
        """

        assert max_lr > final_lr, \
            '`max_lr` must be greater than `final_lr`'

        self.optimizer = optimizer
        self.max_lr = max_lr
        self.dl_len = dl_len
        self.final_lr = final_lr
        self.max_epoch = epochs
        self.warmup = warmup

        self.delta_lr = self.max_lr - self.final_lr


    def step(self, step, epoch):
        # type: (int, int) -> float
        """
        :param step: current step [int]
        :param epoch: current epoch [int]
        :return: current learning rate
        """

        float_epoch = (step / self.dl_len) + epoch

        if float_epoch < self.warmup:
            lr = self.max_lr * float_epoch / self.warmup
        else:
            e = float_epoch - self.warmup
            e_max = self.max_epoch - self.warmup
            cos = math.cos(math.pi * e / e_max)
            lr = self.final_lr + (self.delta_lr * 0.5 * (1 + cos))

        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = lr * param_group['lr_scale']
            else:
                param_group['lr'] = lr
        return lr


    def state_dict(self):
        # type: () -> Dict[str, Any]
        return self.__dict__


    def load_state_dict(self, state_dict, *_args, **_kwargs):
        # type: (Dict[str, Any], ..., ...) -> None

        self.optimizer = state_dict['optimizer']
        self.max_lr = state_dict['max_lr']
        self.dl_len = state_dict['dl_len']
        self.final_lr = state_dict['final_lr']
        self.max_epoch = state_dict['max_epoch']
        self.warmup = state_dict['warmup']

        self.delta_lr = self.max_lr - self.final_lr
