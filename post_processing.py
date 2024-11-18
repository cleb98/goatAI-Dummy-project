import cv2
import numpy as np
import torch


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


    def apply(self, y_pred):
        # type: (torch.Tensor) -> np.array
        """
        Apply post-processing to predicted tensor, in order to obtain
        the corresponding numpy image.

        :param y_pred: predicted tensor image
        :return: numpy image with shape (H,W,C) and value in [0,255]
        """

        # (1) torch tensor to numpy image array
        y_pred = y_pred.detach().cpu().numpy().squeeze()
        y_pred = np.clip(y_pred, 0, 1)
        img_pred = (255 * y_pred.transpose((1, 2, 0))).astype(np.uint8)

        # (2) from RGB to BGR (if needed)
        if self.out_ch_order == 'BGR':
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)

        return img_pred
