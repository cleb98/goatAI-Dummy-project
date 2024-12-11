import os

import cv2
import numpy as np
import torch

def morphological_operations(y_pred):
    # type: (np.array) -> np.array
    """
    Apply morphological operations to predicted tensor, in order to obtain a cleaner mask.
    Closing for first because it will help to close the small holes in the object,
    then opening to remove the white noisy pixels in the background.
    :param y_pred:
    :return: closing, opening: np.array
    """
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(y_pred, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return closing, opening

def binarize(y_pred):
    # type: (torch.Tensor) -> torch.Tensor
    """
    Apply post-processing in validation phase to predicted tensor, in order to obtain a binary mask.
    :param y_pred:
    :return: y_pred in binary format
    """
    return torch.where(y_pred > 0.35,  1, 0)

class PostProcessor(object):
    def __init__(self, out_ch_order='RGB'):
        # type: (str) -> None
        """
        :param out_ch_order: channel order for the output image:
            ->> must be one of {'RGB', 'BGR'}
            ->> default value = 'RGB'
        """
        assert out_ch_order in ['RGB', 'BGR'], \
            f'`out_ch_order` must be "RGB" or "BGR", not {out_ch_order}'

        self.out_ch_order = out_ch_order



    def apply(self, y_pred, save_path = None):
        # type: (torch.Tensor) -> np.array
        """
        Apply post-processing to predicted tensor, in order to obtain
        the corresponding numpy image.

        :param y_pred: predicted tensor image
        :return: numpy image with shape (H,W,C) and value in [0,255]
        """

        # (1) torch tensor to numpy image array

        y_pred = y_pred.detach().cpu().numpy().squeeze(0)
        y_pred = np.clip(y_pred, 0, 1)
        # (2) binarize the image
        # y_pred = np.where(y_pred > 0.2, 1, 0)
        # from (C, H, W) to (H, W, C)
        img_pred = (255 * y_pred.transpose((1, 2, 0))).astype(np.uint8)
        # (3) from RGB to BGR (if needed) and convert to uint8 format for opencv
        if self.out_ch_order == 'BGR':
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)

        # closing, opening = morphological_operations(img_pred)

        #save images to visualize the results
        #(i dont use imshow because i'm on a server)
        try:
            cv2.imwrite(os.path.join(save_path, 'img_pred.png'), img_pred)
            cv2.imwrite(os.path.join(save_path, 'opening.png'), opening)
            cv2.imwrite(os.path.join(save_path, 'closing.png'), closing)
        except:
            cv2.imwrite('img_pred.png', img_pred)
            cv2.imwrite('opening.png', opening)
            cv2.imwrite('closing.png', closing)

        # or visualize the images with opencv
        # cv2.imshow('input', img_pred)
        # cv2.imshow('opening', opening)
        # cv2.imshow('closing', closing)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        return img_pred
