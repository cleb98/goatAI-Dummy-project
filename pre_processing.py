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
        self.resize = transforms.Resize((256, 256))


    def apply(self, img, binary=False, binary_threshold=0.5):
        # type: (np.ndarray, bool, float) -> torch.Tensor
        """
        :param img: image on which you want to apply the pre-processing;
            ->> image must be with RGB channel order
            ->> image must be with shape=(H,W,C) and values in [0, 255]
        :param binary: if `True`, apply binarization to the image after other pre-processing steps
        :param binary_threshold: (it is used only if `binary` is `True`)
            threshold value for binarization, default is 0.5
        :return: torch tensor with shape (C,H,W) or (1,C,H,W)
            and values in range [0,1] or binary values {0,1} if `binary` is `True`
        """
        # (1) numpy array to torch tensor
        x = self.to_tensor(img) #rovina y_true perche è gia fra 0 e 1
        # (1.1) resize
        x = self.resize(x)
        # (2) move to device
        x = x.to(self.device)

        # # if binary is True, apply binarization
        # if binary:
        #     x = torch.where(x > binary_threshold, 1, 0)
                # (3) unsqueeze (if needed)

        if self.unsqueeze:
            x = x.unsqueeze(0)

        return x
