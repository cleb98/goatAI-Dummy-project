import os
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def generate_mask(annotations_files, images_dir, output_dir):
    # type: (str, str, str) -> None
    """
    Generate binary masks from COCO annotations files, save them in output_dir as the same name of the image + '_mask.npy'.
    :param annotations_files: Path to COCO annotations file.
    :param images_dir: Directory containing images.
    :param output_dir: Directory where masks will be saved.
    """
    coco = COCO(annotations_files)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image IDs
    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids, desc="Processing images"):
        # Load image information
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(images_dir, img_filename)

        # Load image to get its dimensions
        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        # Number of classes in the dataset
        num_classes = len(coco.getCatIds())

        # Initialize the mask tensor with `num_classes` channels
        mask = np.zeros((num_classes, height, width), dtype=np.uint8)

        # Get annotation IDs for the current image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Generate the mask for each class
        for ann in anns:
            cat_id = ann['category_id']
            class_idx = coco.getCatIds().index(cat_id)  # Index of the channel mapped to the class

            # Decode the RLE-encoded mask
            rle = coco.annToRLE(ann)
            binary_mask = maskUtils.decode(rle)

            # Update the mask for the corresponding class
            mask[class_idx, :, :] = np.maximum(mask[class_idx, :, :], binary_mask)

        # Save the mask in NumPy format
        mask_filename = os.path.splitext(img_filename)[0] + '_mask.npy'
        mask_path = os.path.join(output_dir, mask_filename)
        np.save(mask_path, mask)


if __name__ == '__main__':
    # Generate masks for validation images
    annotations_files = '/work/tesi_cbellucci/coco/annotations/filtered_instances_val2017.json'
    images_dir = '/work/tesi_cbellucci/coco/images/val2017'
    output_dir = '/work/tesi_cbellucci/coco/images/val_masks'
    print('Generating masks for validation images...')
    generate_mask(annotations_files, images_dir, output_dir)

    # Generate masks for training images
    annotations_files = '/work/tesi_cbellucci/coco/annotations/filtered_instances_train2017.json'
    images_dir = '/work/tesi_cbellucci/coco/images/train2017'
    output_dir = '/work/tesi_cbellucci/coco/images/masks'
    print('Generating masks for training images...')
    generate_mask(annotations_files, images_dir, output_dir)
