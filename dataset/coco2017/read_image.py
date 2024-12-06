import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

#config
## configura i path delle immagini
x_folder = '/work/tesi_cbellucci/coco/images/val2017'
y_folder = '/work/tesi_cbellucci/coco/images/val_masks'
x_file = '000000438774.jpg'

y_file = x_file.split('.')[0] + '_mask.npy'
y_path = os.path.join(y_folder, y_file)
x_path = os.path.join(x_folder, x_file)

def plot_images(x, y):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(x, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Input Image')
    ax[1].imshow(y, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Mask')
    plt.show()

def show_images(x_path, y_path):
    x_img = mpimg.imread(x_path)
    y_img = np.load(y_path)
    #y è un un file.npy a 10 canali, sommo i canali per ottenere un'immagine in bianco e nero
    y_img = y_img.sum(axis=0)
    plot_images(x_img, y_img)
    #shape of the images and masks
    print(f'Image shape: {x_img.shape}')
    print(f'Mask shape: {y_img.shape}')


if __name__ == '__main__':
    #x_path and y_path are defined above in cofing section
    show_images(x_path, y_path)
