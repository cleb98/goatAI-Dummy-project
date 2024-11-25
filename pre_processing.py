import numpy as np
import torch
from torchvision import transforms


class PreProcessor(object):

    def __init__(self, unsqueeze, device):
        # type: (bool, str) -> None
        """
        :param unsqueeze: if `True`, unsqueeze output tensor so that it has
            singleton dimension at the beginning ==> from (C,H,W) to (1,C,H,W);
            if `False`, do not unsqueeze the tensor.
        :param device: device on which you want to have the output tensor
        """
        self.unsqueeze = unsqueeze
        self.to_tensor = transforms.ToTensor()
        self.device = device


    def apply(self, img):
        # type: (np.ndarray) -> torch.Tensor
        """
        :param img: image on which you want to apply the pre-processing;
            ->> image must be with RGB channel order
            ->> image must be with shape=(H,W,C) and values in [0, 255]
        :return: torch tensor with shape (C,H,W) or (1,C,H,W)
            and values in range [0,1]
        """

        # (1) numpy array to torch tensor
        x = self.to_tensor(img)

        # (2) move to device
        x = x.to(self.device)

        # (3) unsqueeze (if needed)
        if self.unsqueeze:
            x = x.unsqueeze(0)

        return x
