import json
import os
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def generate_mask(annotations_files, images_dir, output_dir):
    """
    Generate binary masks from COCO annotations files, save them in output_dir as the same name of the image + '_mask.npy'.
    Also saves a JSON mapping of the classes and their channels for each image.
    :param annotations_files: Path to COCO annotations file.
    :param images_dir: Directory containing images.
    :param output_dir: Directory where masks will be saved.
    """
    coco = COCO(annotations_files)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all category IDs and their names
    cats = coco.loadCats(coco.getCatIds())
    cat_mapping = {cat['id']: cat['name'] for cat in cats}  # Mapping category ID -> class name

    # Get all image IDs
    img_ids = coco.getImgIds()

    # Dict to store the classes for each image
    img_classes = {}

    # Path to save the JSON file
    json_output = os.path.join(output_dir, 'img_classes.json')

    for img_id in tqdm(img_ids, desc="Processing images"):
        # Load image information
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(images_dir, img_filename)

        # Load image to get its dimensions
        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        # Number of classes in the dataset
        num_classes = len(cat_mapping)

        # Initialize the mask tensor with `num_classes` channels
        mask = np.zeros((num_classes, height, width), dtype=np.uint8)

        # Get annotation IDs for the current image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Dict to track the classes for each image
        present_classes = {}

        # Generate the mask for each class
        for ann in anns:
            cat_id = ann['category_id']
            class_idx = list(cat_mapping.keys()).index(cat_id)  # Index of the channel mapped to the class

            # Decode the RLE-encoded mask
            rle = coco.annToRLE(ann)
            binary_mask = maskUtils.decode(rle)

            # Update the mask for the corresponding class
            mask[class_idx, :, :] = np.maximum(mask[class_idx, :, :], binary_mask)

            # Add the class and its channel to the dict
            present_classes[cat_mapping[cat_id]] = class_idx

        # Save the mask in NumPy format
        mask_filename = os.path.splitext(img_filename)[0] + '_mask.npy'
        mask_path = os.path.join(output_dir, mask_filename)
        np.save(mask_path, mask)

        # Add the classes to the main dictionary
        img_classes[img_filename] = present_classes

    # Save the class mapping for all images to a JSON file
    with open(json_output, 'w') as json_file:
        json.dump(img_classes, json_file, indent=4)


if __name__ == '__main__':
    # Generate masks for validation images
    annotations_files = '/work/tesi_cbellucci/coco/annotations/filtered_instances_val2017.json'
    images_dir = '/work/tesi_cbellucci/coco/images/val'
    output_dir = '/work/tesi_cbellucci/coco/images/val_masks'
    print('Generating masks for validation images...')
    generate_mask(annotations_files, images_dir, output_dir)

    # Generate masks for training images
    annotations_files = '/work/tesi_cbellucci/coco/annotations/filtered_instances_train2017.json'
    images_dir = '/work/tesi_cbellucci/coco/images/train'
    output_dir = '/work/tesi_cbellucci/coco/images/train_masks'
    print('Generating masks for training images...')
    generate_mask(annotations_files, images_dir, output_dir)
