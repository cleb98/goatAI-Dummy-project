import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, InterpolationMode
import random
import cv2
import numpy as np

class DataAugmentation:
    def __init__(self, resize_size = (256, 256), rotation_degrees=30, p_flip=0.5, p_crop=0.5):
        self.rotation_degrees = rotation_degrees
        self.p_flip = p_flip
        self.p_crop = p_crop
        self.resize_size = resize_size

    def random_flip(self, x, y):
        """
        Random horizontal and vertical flips.
        """
        if random.random() < self.p_flip:
            x = F.hflip(x)
            y = F.hflip(y)

        if random.random() < self.p_flip:
            x = F.vflip(x)
            y = F.vflip(y)

        return x, y

    def random_rotation(self, x, y):
        """
        Random rotation within a specified range of degrees.
        """
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        x = F.rotate(x, angle, interpolation=InterpolationMode.BILINEAR)
        y = F.rotate(y, angle, interpolation=InterpolationMode.NEAREST)
        return x, y

    def random_crop(self, x, y):
        """
        Random crop applied to both image and mask.
        """
        crop = RandomCrop(size=(self.resize_size[0] - 32, self.resize_size[1] - 32))
        crop_params = crop.get_params(x, crop.size)
        x = F.crop(x, *crop_params)
        y = F.crop(y, *crop_params)
        return x, y

    def apply(self, x, y):
        """
        Apply a sequence of augmentations randomly.
        """
        if random.random() < self.p_flip:
            x, y = self.random_flip(x, y)
        if random.random() < self.p_crop:
            x, y = self.random_crop(x, y)
        if random.random() < 0.5:  # Random rotation applied half the time
            x, y = self.random_rotation(x, y)
        return x, y

if __name__ == '__main__':
    da = DataAugmentation()
    x_path = "path_to_image.jpg"  # Immagine di input
    y_path = "path_to_mask.npy"   # Maschera binaria

    x = cv2.imread(x_path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    y = np.load(y_path)

    x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0  # Normalizza
    y = torch.from_numpy(y).float()

    x_aug, y_aug = da.apply(x, y)

    x_aug = x_aug.permute(1, 2, 0).numpy()  # Converti a HWC per OpenCV
    y_aug = y_aug.numpy()  # Converti a HWC per OpenCV (se necessario)

    cv2.imshow('Augmented Image', cv2.cvtColor(x_aug, cv2.COLOR_RGB2BGR))
    cv2.imshow('Augmented Mask', y_aug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
