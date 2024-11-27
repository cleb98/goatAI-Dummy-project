import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(folder_images, folder_masks, output_folder, train_ratio=0.8, seed=42):
    # type: (str, str, str, float, int) -> None
    """
    :param folder_images: path to the folder containing the images
    :param folder_masks: path to the folder containing the binary masks corresponding to the images (ie labels)
    :param output_folder: path to the folder where the split dataset will be saved
    :param train_ratio: ratio of the dataset to be used for training (default is 0.8)
    :param seed: integer to seed the random number generator (default is 42)

    Description:
    Split a dataset of images and masks into training and validation sets.
    It creates into the output folder two subfolders 'train' and 'val' containing the training and validation sets respectively,
     where each input images as the postfix '_x.png' has a corresponding mask with postfix '_y.png'.


    """
    # Set seed for reproducibility
    random.seed(seed)

    # Create directories for output
    train_images_output = os.path.join(output_folder, 'train')
    train_masks_output = os.path.join(output_folder, 'train')
    val_images_output = os.path.join(output_folder, 'val')
    val_masks_output = os.path.join(output_folder, 'val')

    for folder in [train_images_output, train_masks_output, val_images_output, val_masks_output]:
        os.makedirs(folder, exist_ok=True)

    # Get list of all image and mask filenames
    all_images = os.listdir(folder_images)
    all_masks = os.listdir(folder_masks)

    # Ensure that images and masks have a one-to-one match
    if set(all_images) != set(all_masks):
        raise ValueError("Images and masks do not match in filenames.")

    # Split the dataset into train and validation using scikit-learn's train_test_split
    train_files, val_files = train_test_split(all_images, train_size=train_ratio, random_state=seed)

    # Copy files to appropriate directories
    #training folder creation
    for idx, filename in enumerate(train_files):
        image_source = os.path.join(folder_images, filename)
        mask_source = os.path.join(folder_masks, filename)
        image_dest = os.path.join(train_images_output, f"{idx}_x.png")
        mask_dest = os.path.join(train_masks_output, f"{idx}_y.png")
        shutil.copy2(image_source, image_dest)
        shutil.copy2(mask_source, mask_dest)
    #validation folder creation
    for idx, filename in enumerate(val_files):
        image_source = os.path.join(folder_images, filename)
        mask_source = os.path.join(folder_masks, filename)
        image_dest = os.path.join(val_images_output, f"{idx}_x.png")
        mask_dest = os.path.join(val_masks_output, f"{idx}_y.png")
        shutil.copy2(image_source, image_dest)
        shutil.copy2(mask_source, mask_dest)

    print("Dataset split complete.")

if __name__ == "__main__":
    split_dataset(
        folder_images='images',
        folder_masks='masks',
        output_folder='../kwasir-seg',
        train_ratio=0.8
    )
