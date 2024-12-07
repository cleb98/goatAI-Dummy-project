import torch
import torchvision.transforms.functional as F
import random


class DataAugmentation:
    def __init__(self, p=0.5):

        self.hflip = F.hflip
        self.vflip = F.vflip

    def random_flip(self, x, y, p=0.5):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor # return 2 torch.Tensor -> x, y
        """
        :param x: input tensor with shape (B,C,H,W)
        :param y: target tensor with shape (B,C,H,W)
        :return x_flipped, y_flipped: flipped input and target
         tensors with probability p of being flipped
          horizontally or vertically
        """
        if random.random() < p:
            x = self.hflip(x)
            y = self.hflip(y)

        if random.random() < p:
            x = self.vflip(x)
            y = self.vflip(y)

        # x, y = x.to(self.device), y.to(self.device)

        return x, y

#test above
# if __name__ == '__main__':
# #test import
# import cv2
# from path import Path
# from conf import Conf
# from pre_processing import PreProcessor
#
#     cnf = Conf(exp_name='default')
#     pre_proc = PreProcessor(unsqueeze=True, device= 'cpu')
#
#     x_path = Path(__file__).parent / 'dataset' / 'kwasir-seg' / 'val' / '20_x.png'
#     y_path = Path(__file__).parent / 'dataset' / 'kwasir-seg' / 'val' / '20_y.png'
#     # read image and apply pre-processing
#
#     cv2.waitKey(0)
#
#     x = cv2.cvtColor(cv2.imread(x_path), cv2.COLOR_BGR2RGB)
#     x = pre_proc.apply(x)
#     y = cv2.cvtColor(cv2.imread(y_path), cv2.COLOR_BGR2RGB)
#     y = pre_proc.apply(y)
#
#     da = DataAugmentation()
#     x, y = da.random_flip(x, y)
#
#     x = x.squeeze().permute(1, 2, 0).numpy()
#     y = y.squeeze().permute(1, 2, 0).numpy()
#     x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
#     y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
#     #original x,y pair
#     cv2.imshow('input', cv2.imread(x_path))
#     cv2.imshow('target', cv2.imread(y_path))
#     #augmented x,y pair
#     cv2.imshow('augmented_input', x)
#     cv2.imshow('augmented_target', y)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()