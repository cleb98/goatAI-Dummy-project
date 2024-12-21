import os

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def filter_mask_from_area(annotations_files, mask_dir, output_dir, area_threshold):
    """
    prende in input il json con le immagini da filtrare per area delle maschere.
    Genera il json come quello di input in cui per ci sono solo le immagini le cui mschere hanno un'area bianca sopra una certa soglia.
    :param annotations_files:
    :param images_dir:
    :param output_dir:
    :param area_threshold:
    :return:
    """
    coco = COCO(annotations_files)
    img_ids = coco.getImgIds()

    # info for the new json of filtered images by mask area
    filtered_annotation = []
    filtered_images = []
    selected_img_ids = []

    #debugging: count the nà ofi images per class
    cats = coco.loadCats(coco.getCatIds())
    class_image_count = {cat['name']: 0 for cat in cats}

    for img_id in tqdm(img_ids, desc="Filtering images by mask area"):
        #load image info and its annotations
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        mask_path = os.path.join(mask_dir, img_filename.split('.')[0] + '_mask.npy')
        mask = np.load(mask_path)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if ann['area'] > area_threshold:
                filtered_annotation.append(ann)
                selected_img_ids.append(img_id)
                break