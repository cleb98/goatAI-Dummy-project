import random
import time
from typing import Tuple

import cv2
import numpy as np
import torch
from path import Path
from six import binary_type
from torch.utils.data import Dataset

from conf import Conf
from pycocotools.coco import COCO
from pre_processing import PreProcessor
from data_augmentation import DataAugmentation
from torchvision.transforms import Resize, InterpolationMode



class CocoDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    ->> x: RGB image from the COCO dataset
    ->> y: Corresponding mask with 10 channels saved in .npy format (_mask.npy)
    """

    def __init__(self, cnf, mode):
        # type: (Conf, str) -> None
        """
        :param cnf: configuration object
        :param mode: dataset working mode;
            ->> values in {'train', 'val'}
        :param annotations_file: COCO annotations file path
        :param masks_dir: Directory containing masks in .npy format
        """
        self.cnf = cnf
        self.mode = mode
        if self.mode == 'train':
            self.masks_dir = self.cnf.train_mask
            self.annotation = self.cnf.train_ann
        if self.mode == 'val':
            self.masks_dir = self.cnf.val_mask
            self.annotation = self.cnf.val_ann
        self.coco = COCO(self.annotation)
        self.img_ids = self.coco.getImgIds()

        self.pre_proc = PreProcessor(unsqueeze=False, device='cpu')
        self.aug_data = DataAugmentation()

    def __len__(self):
        # type: () -> int
        """
        :return: number of samples in the dataset
        """
        return len(self.img_ids)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        - reads the i-th sample from the dataset and applies pre-processing
        :param i: index of the sample you want to retrieve
        :return: a tuple (x, y) where:
            ->> x: input RGB image
            ->> y: target mask with 10 channels (10 classes)
        """
        img_id = self.img_ids[i]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = Path(self.cnf.ds_root / self.mode / img_info['file_name'])
        mask_path = Path(self.masks_dir / f"{img_info['file_name'].replace('.jpg', '_mask.npy')}")

        # read input RGB image
        x_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # load corresponding mask
        y_mask = np.load(mask_path)
        # apply preprocessing to input image to get tensor
        x = self.pre_proc.apply(x_img)
        y = torch.from_numpy(y_mask).float()
        y = Resize((256, 256), interpolation=InterpolationMode.NEAREST)(y) #resize the mask using nearest interpolation to mantain the binary format for the mask
        # y = self.pre_proc.apply(y_mask.transpose(1, 2, 0), binary=True, binary_threshold=0.3)
        #data augmentation if in training mode
        if self.mode == 'train':
            x, y = self.aug_data.random_flip(x, y)
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
    # annotations_file = '/work/tesi_cbellucci/coco/annotations/filtered_instances_val2017.json'
    # masks_dir = Path('/work/tesi_cbellucci/coco/images/val_masks')
    ds = CocoDS(cnf=Conf(exp_name='default'), mode='val')

    for i in range(len(ds)):
        x, y = ds[i]
        print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == '__main__':
    main()
