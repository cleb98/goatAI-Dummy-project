import numpy as np
import torch

# srun --partition=all_serial --account=cbellucci --gres=gpu:1 --pty bash


def IoU(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) between two masks.

    :param mask1: numpy array of shape (B, H, W, C)
    :param mask2: numpy array of shape (B, H, W, C)

    :return: intersection / union: numpy array of shape (B,) with IoU values for each couple of masks in the batch
    """


    intersection = np.logical_and(mask1, mask2).sum(axis=(1, 2, 3))
    area1 = mask1.sum(axis=(1, 2, 3))
    area2 = mask2.sum(axis=(1, 2, 3))
    union = area1 + area2 - intersection
    union = np.where(union <= 0, 1e-5, union)

    return intersection / union


#create random and binary tensor of size (B, H, W, C)
B, H, W, C = 8, 128, 128, 1
a = torch.randint(0, 2, (B, H, W, C))
# a = torch.zeros((B, H, W, C))
b = torch.randint(0, 2, (B, H, W, C))

print(a, a.shape)
print(b, b.shape)

print(IoU(a, b).shape, IoU(a, b))
print(torch.cuda.is_available())

'''
srun --partition=all_usr_prod --account=tesi_cbellucci --time=01:00:00 --gres=gpu:1 --pty bash
'''
