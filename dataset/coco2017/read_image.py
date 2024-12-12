import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from pycocotools.coco import COCO
import json



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
def get_color_map(num_classes):
    """
    Crea una colormap con 'tab10' e ottiene i colori per ogni classe.

    :param num_classes: numero totale di classi
    :return: colormap con colori distinti per ogni classe
    """
    # Crea una colormap con 'tab10' e ottiene i colori per ogni classe
    colormap = plt.get_cmap('tab10', num_classes)
    class_colors = (colormap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    return class_colors

def decode_mask_rgb(img, num_classes = 10, background_color = (0, 0 , 0)):
    """
    Decodifica un'immagine con n canali in una maschera RGB con colori distinti per ogni classe.

    :param img: immagine con n canali (C, H, W)
    :param num_classes: numero totale di classi
    :param background_color: colore RGB per i pixel di background
    :return: immagine RGB con colori distinti per classe
    """
    # Identifica i pixel di background (tutti i canali a zero)
    background_mask = np.all(img == 0, axis=0)

    # Determina la classe massima per ogni pixel
    class_map = img.argmax(axis=0)
    class_map[background_mask] = -1  # Imposta -1 per i pixel di background

    # Crea una colormap con 'tab10' e ottiene i colori per ogni classe
    colormap = get_color_map(num_classes)

    # Inserisce il colore di background come primo colore
    rgb_colors = np.vstack([np.array(background_color, dtype=np.uint8), colormap])

    # Crea l'immagine RGB vuota
    decoded_img = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)

    # Assegna il colore background
    decoded_img[class_map == -1] = rgb_colors[0]

    # Assegna i colori alle classi
    for class_idx in range(num_classes):
        mask = (class_map == class_idx)
        # class_idx+1 perché l'indice 0 è stato riservato al background
        decoded_img[mask] = rgb_colors[class_idx + 1]

    return decoded_img

def decode_mask_grayscale(img):
    #type: (np.ndarray) -> np.ndarray
    '''
    Decodes the iamge with n channels to a single channel grayscale mask
    :param img: image with n channels
    :return: single channel image
    '''
    background_mask = np.all(img == 0, axis=0) #identify where backgrund is
    back = np.zeros_like(img[0], dtype=np.int8)
    back[background_mask] = -1 #background pixels mapped to -1
    img = img.argmax(axis=0) + back #pixels ==0 are now class 0 (not background) and pixels == -1 are background
    img = (img + 1) * 255 / 10 #background is set from -1 to 0 and class i is set to

    return img.astype(np.uint8)

def plot_images(x, y, label_name):
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

    # Titolo globale
    fig.suptitle(label_name)

    plt.tight_layout()
    plt.show()

def show_images(x_path, y_path):
    cat_info = json.load(open(y_folder + '/img_classes.json'))
    x_img = mpimg.imread(x_path)
    x_img_name = x_path.split('/')[-1]
    cat = cat_info[x_img_name]
    print(cat)
    y_img = np.load(y_path)
    cat_val = np.unique(y_img)
    print(cat_val, 'pixels values befoire decoding') #grayscale values of the classes
    #y è un un file.npy a 10 canali, sommo i canali per ottenere un'immagine in bianco e nero
    y_img = decode_mask_rgb(y_img) #decode the mask to a single channel image
    cat_val = np.unique(y_img)
    print(cat_val, 'pixels values post decoding') #grayscale values of the classes
    #info about classes in the json file, added to the plot


    plot_images(x_img, y_img, cat)

    print(f'Image shape: {x_img.shape}')
    print(f'Mask shape: {y_img.shape}')


if __name__ == '__main__':

    #show single image, x_path and y_path are defined above in cofing section
    # show_images(x_path, y_path)

    #show num_img images
    num_img = 12
    coco = COCO('/work/tesi_cbellucci/coco/annotations/filtered_instances_val2017.json')
    for i, file in enumerate(os.listdir(x_folder)):
        #load the image from json file
        img_id = coco.getImgIds()
        img_info = coco.loadImgs(img_id[i])[0]
        x_path = os.path.join(x_folder, img_info['file_name'])
        y_path = os.path.join(y_folder, img_info['file_name'].split('.')[0] + '_mask.npy')
        show_images(x_path, y_path)
        if i == num_img:
            break

