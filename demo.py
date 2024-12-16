import torch
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import json
from conf import Conf
from models.model import UNet, UNet1
from pre_processing import PreProcessor
from post_processing import PostProcessor, decode_mask_rgb, get_color_map
from torchvision.transforms import Resize, InterpolationMode

class Demo:
    def __init__(self, exp_name, mode = 'val'):
        self.cnf = Conf(exp_name=exp_name)
        self.mode = mode
        self.x_folder = os.path.join(self.cnf.ds_root, mode)
        if self.mode == 'train':
            self.masks_dir = self.cnf.train_mask
            self.annotation = self.cnf.train_ann
        if self.mode == 'val':
            self.masks_dir = self.cnf.val_mask
            self.annotation = self.cnf.val_ann
        self.pre_proc = PreProcessor(unsqueeze=False, device='cpu')
        self.post_proc = PostProcessor()
        self.model = UNet1(input_channels=3, num_classes=10)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = self.model.to(self.cnf.device)
        self.model.load_w(self.cnf.exp_log_path / 'best.pth')
        self.coco = COCO(self.annotation)
        self.cat_info = json.load(open(self.masks_dir + '/img_classes.json'))



    def show_images(self, x_path, y_path, label_name):
        x_img = mpimg.imread(x_path)
        x_img_name = os.path.basename(x_path)
        y_img = np.load(y_path)

        print(f"Class info for {x_img_name}: {label_name}")

        # Pre-process the input and mask
        x_img = self.pre_proc.apply(x_img).numpy().transpose(1, 2, 0)

        y_img = torch.from_numpy(y_img).float()
        y_img = y_img[:self.cnf.num_classes]
        y_img = Resize((256, 256), interpolation=InterpolationMode.NEAREST)(y_img)

        print("Pixel values before decoding:", np.unique(y_img.numpy()))

        # Decode the mask to RGB
        y_img = decode_mask_rgb(y_img.unsqueeze(0), colormap=get_color_map())
        y_img = y_img.squeeze(dim=0).numpy().transpose(1, 2, 0)

        print("Pixel values after decoding:", np.unique(y_img))

        # #forward pass and post-processing
        # y_pred = self.model.forward(x_img)
        # y_pred = self.post_proc.apply(y_pred.unsqueeze(0), save_path=cnf.exp_log_path)
        # y_pred = y_pred.numpy()
        y_pred = None

        # Visualize the images
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(x_img)
        ax[0].axis('off')
        ax[0].set_title('Input Image')

        ax[1].imshow(y_img)
        ax[1].axis('off')
        ax[1].set_title('Mask')

        fig.suptitle(label_name)
        plt.tight_layout()
        plt.show()

    def run(self, num_img=1):
        img_ids = self.coco.getImgIds()

        for i, img_id in enumerate(img_ids):
            img_info = self.coco.loadImgs(img_id)[0]
            x_path = os.path.join(self.x_folder, img_info['file_name'])
            y_path = os.path.join(self.masks_dir, img_info['file_name'].split('.')[0] + '_mask.npy')

            if os.path.exists(x_path) and os.path.exists(y_path):
                label_name = self.cat_info.get(img_info['file_name'], "Unknown")
                self.show_images(x_path, y_path, label_name)

            if i + 1 >= num_img:
                break

if __name__ == '__main__':
    try:
        demo = Demo(exp_name='1class', mode='val')
        demo.run(num_img=5)
    except Exception as e:
        print(f'Error: {e}')
