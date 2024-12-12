import cv2
import torch
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import json
from conf import Conf
from path import Path
from pycocotools.coco import COCO
from models.model import UNet
from pre_processing import PreProcessor
from post_processing import PostProcessor, decode_mask_rgb, get_color_map
from torchvision.transforms import Resize, InterpolationMode

def plot_images(x, y, y_pred, label_name):
    """
    Visualizza un'immagine e la sua corrispondente maschera.

    :param x: immagine di input (H, W, 3)
    :param y: maschera (H, W) o (H, W, 3)
    :param label_name: nome/etichetta della classe
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Mostra l'immagine di input (RGB)
    ax[0].imshow(x)  # Assicurati che x sia un'immagine RGB (H, W, 3)
    ax[0].axis('off')
    ax[0].set_title('Input Image')

    # Mostra la maschera
    if y.ndim == 3:  # Maschera RGB
        ax[1].imshow(y)
    else:  # Maschera in scala di grigi
        ax[1].imshow(y, cmap='gray')

    ax[1].axis('off')
    ax[1].set_title('Mask')

    # mostra la maschera predetta
    ##TOADD!
    # Titolo globale
    fig.suptitle(label_name)

    plt.tight_layout()
    plt.show()

def show_images(x_path, y_path, cat_info, cnf):

    pre_proc = PreProcessor(unsqueeze=False, device='cpu')
    post_proc = PostProcessor()
    model = UNet()
    model.eval()
    model.requires_grad(False)
    model = model.to(cnf.device)
    model.load_w(cnf.exp_log_path / 'best.pth')
    x_img = mpimg.imread(x_path)
    x_img_name = x_path.split('/')[-1]
    #info about classes in the json file, added to the plot
    cat = cat_info[x_img_name]
    print(cat)
    y_img = np.load(y_path)
    cat_val = np.unique(y_img)
    x_img = pre_proc.apply(x_img).numpy().transpose(1, 2, 0)
    y_img = torch.from_numpy(y_img).float()
    y_img = Resize((256, 256), interpolation=InterpolationMode.NEAREST)(y_img)  # resize the mask using nearest interpolation to mantain the binary firmat for the mask
    print(cat_val, 'pixels values befoire decoding') #grayscale values of the classes
    #y è un un file.npy a 10 canali, sommo i canali per ottenere un'immagine in bianco e nero
    y_img = decode_mask_rgb(y_img.unsqueeze(0), colormap=get_color_map()) #decode the mask to a single channel image
    y_img = y_img.squeeze(dim=0).numpy().transpose(1, 2, 0)
    cat_val = np.unique(y_img)
    print(cat_val, 'pixels values post decoding') #grayscale values of the classes

    # y_pred = model.forward(x)
    # y_pred = post_proc.apply(y_pred.unsqueeze(0), save_path=cnf.exp_log_path)
    # y_pred = y_pred.numpy()
    y_pred = None
    plot_images(x_img, y_img, y_pred, cat)
    print(f'Image shape: {x_img.shape}')
    print(f'Mask shape: {y_img.shape}')

def demo(exp_name, num_img = 1, img_name = None):
    # type: (str, int, str) -> None
    """
    Quick demo of the complete pipeline on a val/test image.
    Show num_img examples of input image, target mask and predicted mask in RGB
    :param exp_name:
    :param num_img:
    :param img_name:
    """
    cnf = Conf(exp_name=exp_name)
    # init model and load weights of the best epoch
    #show num_img images
    coco = COCO('/work/tesi_cbellucci/coco/annotations/filtered_instances_val2017.json')
    x_folder = os.path.join(cnf.ds_root, 'val')
    y_folder = cnf.val_mask
    cat_info = json.load(open(y_folder + '/img_classes.json'))
    for i, file in enumerate(os.listdir(x_folder)):
        #load the image from json file
        img_id = coco.getImgIds()
        img_info = coco.loadImgs(img_id[i])[0]
        x_path = os.path.join(x_folder, img_info['file_name'])
        y_path = os.path.join(y_folder, img_info['file_name'].split('.')[0] + '_mask.npy')
        show_images(x_path, y_path, cat_info, cnf)
        # forward pass and post-processing
        if i == num_img:
            break

    # # show input and output
    # cv2.imshow('input', img[..., ::-1])
    # cv2.imshow('output', img_pred[..., ::-1])
    # cv2.waitKey(0)

    # save input and output
    # cv2.imwrite(str(cnf.exp_log_path / 'input.png'), img[..., ::-1])
    # cv2.imwrite(str(cnf.exp_log_path / 'target.png'), y[..., ::-1])
    # cv2.imwrite(str(cnf.exp_log_path / 'output.png'), img_pred[..., ::-1])


if __name__ == '__main__':
    try:
        demo(exp_name='default', num_img=5)
    except Exception as e:
        print(f'Error: {e}')
