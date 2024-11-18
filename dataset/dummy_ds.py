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

        #se ds_dir non esiste allora scarica il dataset zippato da:
        #https://drive.google.com/file/d/1dVKbd_G039Ai6VG7HHc640IRDKTKWouT/view?usp=drive_link
        #lo unzippa e lo mette in ds_dir

        if not ds_dir.exists():
            print(f'\n▶ Dataset not found in \'{ds_dir}\'')
            print('▶ Downloading dataset...')
            ds_dir.mkdir_p()
            ds_zip = ds_dir / 'dummy_ds.zip'
            ds_zip_url = 'https://drive.google.com/file/d/1dVKbd_G039Ai6VG7HHc640IRDKTKWouT/view?usp=drive_link'
            ds_zip.download_url(ds_zip_url)
            print('▶ Unzipping dataset...')
            ds_zip.unpack(ds_dir)
            print('▶ Dataset downloaded and unzipped successfully!')

        self.paths = []
        for x_path in ds_dir.files('*_x.png'):
            y_path = Path(x_path.replace('_x.png', '_y.png'))
            self.paths.append((x_path, y_path))

        self.pre_proc = PreProcessor(unsqueeze=False, device='cpu')


    def __len__(self):
        # type: () -> int
        return len(self.paths)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        x_path, y_path = self.paths[i]

        # read input and target (RGB order)
        x_img = cv2.cvtColor(cv2.imread(x_path), cv2.COLOR_BGR2RGB)
        y_img = cv2.cvtColor(cv2.imread(y_path), cv2.COLOR_BGR2RGB)

        # apply pre processing to input and target
        x = self.pre_proc.apply(x_img)
        y = self.pre_proc.apply(y_img)

        return x, y


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
    ds = DummyDS(cnf=Conf(exp_name='default'), mode='val')

    for i in range(len(ds)):
        x, y = ds[i]
        print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == '__main__':
    main()
