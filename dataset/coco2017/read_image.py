import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from pycocotools.coco import COCO
import json

from pygments.styles.dracula import background

#############################################
#config

## configura i path delle immagini
x_folder = '/work/tesi_cbellucci/coco/images/val'
y_folder = '/work/tesi_cbellucci/coco/images/val_masks'

# x_file = '000000438774.jpg'
x_file = '000000289417.jpg'
#x_file = 000000435003.jpg

# y_file = x_file.split('.')[0] + '_mask.npy'
# y_path = os.path.join(y_folder, y_file)
# x_path = os.path.join(x_folder, x_file)

#fine config
#############################################

def decode_mask(img):
    #type: (np.ndarray) -> np.ndarray
    '''
    Decodes the iamge with n channels to a single channel grayscale mask
    :param img: image with n channels
    :return: single channel image
    '''
    background_mask = np.all(img == 0, axis=0)
    back = np.zeros_like(img[0], dtype=np.int8)
    back[background_mask] = -1 #background pixels mapped to -1
    img = img.argmax(axis=0) + back #pixels ==0 are now class 0 (not background) and pixels == -1 are background
    img = (img + 1) * 255 / 10 #background is set from -1 to 0 and class i is set to
    return img.astype(np.uint8)

def plot_images(x, y, label_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(x, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Input Image')
    ax[1].imshow(y, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Mask')
    fig.suptitle(label_name)
        

    plt.show()

def show_images(x_path, y_path):
    cat_info = json.load(open(y_folder + '/img_classes.json'))
    x_img = mpimg.imread(x_path)
    x_img_name = x_path.split('/')[-1]
    y_img = np.load(y_path)
    cat_val = np.unique(y_img)
    print(cat_val, 'classes values') #grayscale values of the classes
    #y è un un file.npy a 10 canali, sommo i canali per ottenere un'immagine in bianco e nero
    y_img = decode_mask(y_img) #decode the mask to a single channel image
    #info about classes in the json file, added to the plot
    cat = cat_info[x_img_name]
    print(cat)

    plot_images(x_img, y_img, cat)

    print(f'Image shape: {x_img.shape}')
    print(f'Mask shape: {y_img.shape}')


if __name__ == '__main__':
    #show single image defined above
    #x_path and y_path are defined above in cofing section
    # show_images(x_path, y_path)

    #show num_img images
    num_img = 12
    for i, file in enumerate(os.listdir(x_folder)):
        coco = COCO('/work/tesi_cbellucci/coco/annotations/filtered_instances_val2017.json')
        #load the image from json file
        img_id = coco.getImgIds()
        img_info = coco.loadImgs(img_id[i])[0]
        x_path = os.path.join(x_folder, img_info['file_name'])
        y_path = os.path.join(y_folder, img_info['file_name'].split('.')[0] + '_mask.npy')
        show_images(x_path, y_path)
        if i == num_img:
            break

