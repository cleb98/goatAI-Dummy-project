import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


#function used to decode the image form 10 channell class map to rgb image with 1 color for each class map
def get_color_map(num_classes = 10):
    #type: (int) -> torch.Tensor
    """
    Returns a colormap with distinct colors for each class
    :param num_classes: total number of classes
    :return: a tensor representing a colormap with distinct rgb colors for each class
    """
    colormap = plt.get_cmap('tab10', num_classes)
    cmap = (colormap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    return torch.from_numpy(cmap)

def decode_mask_rgb(img, colormap, background_color = (0, 0 , 0)):
    # type
    """
    Decode an image with n channels into an RGB mask with distinct colors for each class.
    :param img: immagine con n canali (B, C, H, W)
    :param num_classes: numero totale di classi
    :param background_color: colore RGB per i pixel di background
    :param colormap: tensor with sape (10, 3) with 10 different rgb color used to get the rgb version of the target masks
    :return: immagine RGB con colori distinti per classe
    """
    background_mask = torch.all(img == 0, dim = 1) #identify background pixels (B, H, W)
    class_map = img.argmax(dim=1) #determine the class with the highest probability for each pixel
    class_map[background_mask] = -1 #set -1 for background pixels

    #create a tensor rgb_colors used as a colormap during decoding
    # if colormap is None:
    #     colormap = get_color_map(num_classes=10).to(img.device)
    class_colors = colormap
    background_color = torch.tensor(background_color, dtype = torch.uint8, device = img.device).unsqueeze(dim=0)
    rgb_colors = torch.cat([background_color, class_colors], dim=0)
    # set background color to background pixels using fancy indexing
    decoded_img = rgb_colors[class_map + 1]  #color_map+1 because currently we have class_map [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] we want to map it to range [0, 10]
    return decoded_img.permute(0, 3, 1, 2)

# def decode_mask(img):
#     #type: (torch.Tensor) -> torch.Tensor
#     '''
#     Decodes the image with n channels to a single channel grayscale mask
#     :param img: tensor with n channels
#     :return: img_decoded: single channel mask torch tensor in uint8
#     '''
#     background_mask = torch.all(img == 0, dim=1) #shape = (B,1,H,W)
#     back = torch.zeros_like(img, dtype=torch.int8).sum(dim=1) #shape = (B,1,H,W)
#     back[background_mask] = -1 #background pixels mapped to -1
#     img = img.argmax(dim=1) + back #pixels ==0 are now class 0 (not background) and pixels == -1 are background
#     img = (img + 1) * 255 / 10 #background is set from -1 to 0 and class i is set to
#     #add channel dim (B, 1, H, W) with values in range [0, 255]
#     return img.unsqueeze(1).to(torch.uint8)

#morph operation used to postpèrocess the predicted binary mask
def morphological_transform(y_pred):
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
        cmap = get_color_map()
        # (1) binarize the image
        y_pred = binarize(y_pred)
        # (2) decode tensor with 10 channel to the rgb mask
        y_pred = decode_mask_rgb(y_pred, cmap)
        # (3) to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        # from (C, H, W) to (H, W, C)
        img_pred = y_pred.transpose((1, 2, 0)).astype(np.uint8)
        # (3) from RGB to BGR (if needed) and convert to uint8 format for opencv
        if self.out_ch_order == 'BGR':
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)


        # closing, opening = morphological_transform(img_pred)
        #
        # #save images to visualize the results
        # #(i dont use imshow because i'm on a server)
        # try:
        #     cv2.imwrite(os.path.join(save_path, 'img_pred.png'), img_pred)
        #     cv2.imwrite(os.path.join(save_path, 'opening.png'), opening)
        #     cv2.imwrite(os.path.join(save_path, 'closing.png'), closing)
        # except:
        #     cv2.imwrite('img_pred.png', img_pred)
        #     cv2.imwrite('opening.png', opening)
        #     cv2.imwrite('closing.png', closing)

        # or visualize the images with opencv
        # cv2.imshow('input', img_pred)
        # cv2.imshow('opening', opening)
        # cv2.imshow('closing', closing)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        return img_pred
