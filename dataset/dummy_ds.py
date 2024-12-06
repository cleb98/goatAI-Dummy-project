import random
import time
from typing import Tuple

import cv2
import numpy as np
import torch
from path import Path
from torch.utils.data import Dataset

from conf import Conf

from pre_processing import PreProcessor
from data_augmentation import DataAugmentation

class DummyDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    ->> x: RGB image of size 128x128 representing a light blue circle
        (radius = 16 px) on a dark background (random circle position,
        randomly colored dark background)
    ->> y: copy of x, with the light blue circle surrounded with a red
        line (4 px internal stroke)
    """


    def __init__(self, cnf, mode):
        # type: (Conf, str) -> None
        """
        :param cnf: configuration object
        :param mode: dataset working mode;
            ->> values in {'train', 'val'}
        """
        self.cnf = cnf
        self.mode = mode


        ds_dir = self.cnf.ds_root / mode

        self.paths = []
        for x_path in ds_dir.files('*_x.png'):
            y_path = Path(x_path.replace('_x.png', '_y.png'))
            self.paths.append((x_path, y_path))

        self.pre_proc = PreProcessor(unsqueeze=False, device='cpu')
        self.aug_data = DataAugmentation()

    def __len__(self):
        # type: () -> int
        """
        :return: number of samples in the dataset
        """
        return len(self.paths)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        - reads the i-th sample from the dataset and applies pre-processing
        :param i: index of the sample you want to retrieve
        :return: a tuple (x, y) where:
            ->> x: input RGB image
            ->> y: target RGB image
        """
        x_path, y_path = self.paths[i]

        # read input and target (RGB order)

        x_img = cv2.cvtColor(cv2.imread(x_path), cv2.COLOR_BGR2RGB)
        y_img = cv2.cvtColor(cv2.imread(y_path), cv2.COLOR_BGR2GRAY)

        # apply pre processing to input and target
        x = self.pre_proc.apply(x_img)
        y = self.pre_proc.apply(y_img)

        #data augmentation
        if self.mode == 'train':
            x, y = self.aug_data.random_flip(x, y)

        # binarize target necessary for BCE loss used in training
        y = torch.where(y > 0.5, 1.0, 0.0)

        return x, y

    '''
    @staticmethod is a decorator in Python that marks a method as a static method of a class. 
    It indicates that the method does not operate on an instance of the class and does not modify any class state.
    Instead, it works independently and can be called without needing to create an instance of the class.
    Static methods are typically used for utility functions or operations that logically belong to the class but do not require access to class or instance attributes.
    '''
    @staticmethod
    def wif(worker_id):
        # type: (int) -> None
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = (int(round(time.time() * 1000)) + worker_id) % (2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    @staticmethod
    def wif_val(worker_id):
        # type: (int) -> None
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = worker_id + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def main():
    ds = DummyDS(cnf=Conf(exp_name='default'), mode='train')

    for i in range(len(ds)):
        x, y = ds[i]
        print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == '__main__':
    main()
