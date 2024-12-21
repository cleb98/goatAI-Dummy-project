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
import torchvision.transforms.functional as F
import json
from visual_utils import decode_and_apply_mask_overlay
import matplotlib.pyplot as plt


class CocoDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    ->> x: RGB image from the COCO dataset
    ->> y: Corresponding mask with 10 channels saved in .npy format (_mask.npy)
    """

    def __init__(self, cnf, mode, data_augmentation = True, resize_size=(256, 256)):
        # type: (Conf, str, bool) -> None
        """
        :param cnf: configuration object
        :param mode: dataset working mode;
            ->> values in {'train', 'val'}
        :param data_augmentation: if `True`, data augmentation is applied
        :param resize_size: size to which the input image and mask are resized
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
        self.resize_size = resize_size
        if data_augmentation:
            self.da = DataAugmentation(resize_size)
        self.cat_info = json.load(open(self.masks_dir + '/img_classes.json'))

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
        y = torch.from_numpy(np.load(mask_path)).float() #do not use pre_proc.apply(y), will apply toTensor
        if self.cnf.num_classes != y.shape[0]:
            y = y[:self.cnf.num_classes] #prendo solo i primi num_classes canali
        x = self.pre_proc.apply(x_img)
        #resize input image and mask
        x, y = self.resize(x, y)
        return x, y

    def resize(self, x, y):
        """
        Resizes the input and mask to the specified size.
        """
        x = F.resize(x, self.resize_size, interpolation=InterpolationMode.BILINEAR)
        y = F.resize(y, self.resize_size, interpolation=InterpolationMode.NEAREST)
        return x, y

    def collate_fn(self, batch):
        """
        Custom collate function for applying augmentations to a batch.
        """
        x_batch, y_batch = zip(*batch)
        x_batch = torch.stack(x_batch)
        y_batch = torch.stack(y_batch)

        if self.da:  # Apply data augmentation only if enabled
            x_batch, y_batch = self.da.apply(x_batch, y_batch)

        return x_batch, y_batch

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


def plot_grid(tensor, title="Batch Grid"):
    """
    Display a grid of images from a tensor using matplotlib.

    :param tensor: Input tensor of shape (B, C, H, W), where C = 1 (grayscale) or C = 3 (RGB).
    :param title: Title of the grid display.
    """
    # Ensure the tensor is on CPU and detached for visualization
    tensor = tensor.detach().cpu()

    B, C, H, W = tensor.shape

    # Determine grid size (e.g., square-like grid)
    cols = int(torch.sqrt(torch.tensor(B)).item())
    rows = int(B / cols)

    # Create the grid for images
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  # Flatten in case of a single row or column

    for i in range(rows * cols):
        if i < B:
            img = tensor[i].permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            if C == 1:
                img = img.squeeze(-1)  # Remove channel dimension for grayscale
                axes[i].imshow(img, cmap="gray")
            else:
                axes[i].imshow(img)
            axes[i].axis('off')  # Turn off axis for better visualization
        else:
            axes[i].axis('off')  # Hide extra subplots

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()



def main():
    from visual_utils import apply_mask_overlay, tensor_to_cv2
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader

    cnf = Conf(exp_name='3classes')
    ds = CocoDS(cnf=cnf, mode='train')
    loader = DataLoader(
        dataset=ds, batch_size=cnf.batch_size,
        num_workers=1, shuffle=False, pin_memory=True,
        worker_init_fn=ds.wif, collate_fn=ds.collate_fn
    )

    for i, sample in enumerate(loader):
        x, y = sample

        # show the first 8 examples (for debugging)
        # if i < 8:
        #     x, y = x[0], y[0]
        #     overlay = apply_mask_overlay(x, y)
        #     x = tensor_to_cv2(x)
        #     overlay = tensor_to_cv2(overlay)
        #     fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        #     ax[0].imshow(x)
        #     ax[1].imshow(overlay)
        #     ax[0].title.set_text('Input Image')
        #     ax[1].title.set_text('Mask Overlay')
        #     ax[0].axis('off')
        #     ax[1].axis('off')
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close('all')
        # else:
        #     print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')
        #per ogni batch stampo la griglia 16*16 delle img decoded
        if i == 0:
            decoded = decode_and_apply_mask_overlay(x, y)
            #custom function to display the grid of a batch with matplotlib
            plot_grid(decoded, title="Batch Grid")
        else:
            print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == '__main__':
    main()
