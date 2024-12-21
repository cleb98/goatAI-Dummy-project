import json
from pycocotools.coco import COCO
#generate mask
import json
import os
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

class COCOFilter:
    def __init__(self, coco, n_classes = None, n_samples = 0, classes=None):
        #type: (COCO, int, int, List[str]) -> None
        """
        Classe helper per filtrare le categorie e le immagini dal dataset COCO.
        """
        self.coco = coco
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.classes = classes if classes is not None else None
        self.selected_cat_ids = None
        self.selected_img_ids = set()

    def set_classes_from_names(self, class_names):
        """
        Select categories from the annotation file for the dataset
        """
        # Get category IDs from category names
        cat_ids = self.coco.getCatIds(catNms=class_names)
        # Filter categories
        self.classes = [cat for cat in self.coco.loadCats(cat_ids)]
        self.selected_cat_ids = cat_ids

    def set_classes_from_n_classes(self, n_classes):
        # Filter categories , excluding 'person'
        # Get all categories with at least `n_samples` annotations
        # by default, n_samples = 0 (all categories)
        filtered_categories = [
            cat for cat in self.coco.loadCats(self.coco.getCatIds())
            if len(self.coco.getAnnIds(catIds=[cat['id']])) >= self.n_samples and cat['name'] != 'person'
        ]

        # Sort categories by the number of images in descending order to select the most common classes
        filtered_categories.sort(
            key=lambda x: len(self.coco.getAnnIds(catIds=[x['id']])),
            reverse=True
        )

        # Select the top `n_classes` categories
        self.classes = filtered_categories[:self.n_classes]
        self.selected_cat_ids = [cat['id'] for cat in self.classes]

    def set_classes(self):
        """
        Seleziona le classi dal file json di annotazioni per il dataset.
        if `n_classes` is not None, select the top `n_classes` categories with the most annotations
        if `n_classes` is None and `classes` is not None, select categories based on their names
        if n_samples is 0, select all images for each selected class, otherwise select up to `n_samples` images per class
        """
        if self.n_classes is None and self.classes is not None:
            self.set_classes_from_names(self.classes)

        else:
            self.set_classes_from_n_classes(self.n_classes)

        self.selected_img_ids = self.get_images() #



    def get_images(self):
        """
        Seleziona un numero massimo di immagini per ogni classe selezionata.
        """
        num_images = self.n_samples
        img_ids = set()
        if num_images != 0:
            for cat_id in self.selected_cat_ids:
                # Get image IDs associated with the current category
                cat_img_ids = self.coco.getImgIds(catIds=[cat_id])
                # Select up to `num_images` images per category
                img_ids.update(cat_img_ids[:num_images])
        else: # prendi tutte le img per ogni classe selezionata
            for cat_id in self.selected_cat_ids:
                # Get image IDs associated with the current category
                cat_img_ids = self.coco.getImgIds(catIds=[cat_id])
                # Select up to `num_images` images per category
                img_ids.update(cat_img_ids)
        return img_ids


class DatasetCreator:
    def __init__(self, train_ann_file, val_ann_file, out_train_ann_file, out_val_ann_file):
        """
        Inizializza la classe DatasetCreator.
        """
        self.train_coco = COCO(train_ann_file)
        self.val_coco = COCO(val_ann_file)
        self.train_ann_file = train_ann_file
        self.out_train_ann_file = out_train_ann_file
        self.val_ann_file = val_ann_file
        self.out_val_ann_file = out_val_ann_file
        self.train_filter = None
        self.val_filter = None

    def initialize_filters(self, n_train_samples=0, n_val_samples=0, n_classes=None, class_names=None):
        """
        Inizializza i filtri COCO per il training set e il validation set.
        """
        # Initialize the training filter
        assert class_names is not None or n_classes is not None, "You must provide n_classes or class_names, not both but at least one is needed."
        self.train_filter = COCOFilter(self.train_coco, n_classes, n_train_samples, class_names)
        self.train_filter.set_classes()

        # Initialize the validation filter with a different number of samples
        self.val_filter = COCOFilter(self.val_coco, n_classes, n_val_samples, class_names)
        self.val_filter.selected_cat_ids = self.train_filter.selected_cat_ids
        self.val_filter.classes = self.train_filter.classes
        self.val_filter.selected_img_ids = self.val_filter.get_images()

    def save_filtered_dataset(self, coco_filter, output_file):
        """
        Crea un nuovo file JSON COCO con le classi selezionate e le immagini filtrate dal json di annotation originale.
        """
        coco = coco_filter.coco
        # Filter annotations and images based on the selected IDs
        filtered_annotations = [
            ann for ann in coco.anns.values()
            if ann['image_id'] in coco_filter.selected_img_ids and ann['category_id'] in coco_filter.selected_cat_ids
        ]
        filtered_images = [
            img for img in coco.imgs.values()
            if img['id'] in coco_filter.selected_img_ids
        ]

        # Create the new dataset structure
        filtered_coco = {
            'info': coco.dataset.get('info', {}),
            'licenses': coco.dataset.get('licenses', []),
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': coco_filter.classes,
        }
        print(f"Numero di categorie selezionate: {len(coco_filter.classes)}")
        print(f"Classi selezionate: {[cat['name'] for cat in coco_filter.classes]}")
        print(f"Numero di immagini selezionate: {len(filtered_images)}")



        # Save the new JSON file
        with open(output_file, 'w') as f:
            json.dump(filtered_coco, f)

        print(f"Nuovo dataset salvato in: {output_file}")

    def get_filtered_datasets(self):
        """
        Crea i nuovi dataset filtrati per il training set e il validation set.
        """
        self.save_filtered_dataset(self.train_filter, self.out_train_ann_file)
        self.save_filtered_dataset(self.val_filter, self.out_val_ann_file)

def generate_mask(annotations_files, images_dir, output_dir, area_threshold=0.05, max_samples_per_class=None):
    """
    Generate binary masks from COCO annotations files, saving only those that cover more than
    a specified percentage of the image area. Masks are saved in output_dir with the same name
    as the image plus '_mask.npy'. Also updates the input annotations JSON file with only the
    filtered images and annotations. Additionally, saves a separate JSON with class-to-image mappings.
    :param annotations_files: Path to COCO annotations file.
    :param images_dir: Directory containing images.
    :param output_dir: Directory where masks will be saved.
    :param area_threshold: Minimum area ratio (0-1) for masks to be saved.
    :param max_samples_per_class: Maximum number of samples to save per class.
    """
    print(f'Generating masks in {output_dir} from annotations in {annotations_files}')
    coco = COCO(annotations_files)
    os.makedirs(output_dir, exist_ok=True)

    # Get all category IDs and their names
    cats = coco.loadCats(coco.getCatIds())
    cat_mapping = {cat['id']: cat['name'] for cat in cats}

    # Debugging: Initialize counters for the number of masks per class
    class_mask_count = {cat['name']: 0 for cat in cats}

    # Get all image IDs
    img_ids = coco.getImgIds()

    # Dictionary to store the classes for each image
    img_classes = {}
    filtered_images = []
    filtered_annotations = []

    for img_id in tqdm(img_ids, desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(images_dir, img_filename)

        # Load image to get its dimensions
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        height, width = image.shape[:2]

        num_classes = len(cat_mapping)
        mask = np.zeros((num_classes, height, width), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        present_classes = {}
        valid_annotations = []

        for ann in anns:
            cat_id = ann['category_id']
            class_idx = list(cat_mapping.keys()).index(cat_id)

            # Check if the maximum number of samples for this class has been reached
            class_name = cat_mapping[cat_id]
            if max_samples_per_class and class_mask_count[class_name] >= max_samples_per_class:
                continue

            rle = coco.annToRLE(ann)
            binary_mask = maskUtils.decode(rle)

            # Calculate the area of the mask
            mask_area = np.sum(binary_mask > 0)
            image_area = height * width
            area_ratio = mask_area / image_area

            if area_ratio > area_threshold:
                mask[class_idx, :, :] = np.maximum(mask[class_idx, :, :], binary_mask)
                present_classes[class_name] = class_idx
                valid_annotations.append(ann)

                # Increment the mask count for the class
                class_mask_count[class_name] += 1

        if present_classes:
            mask_filename = os.path.splitext(img_filename)[0] + '_mask.npy'
            mask_path = os.path.join(output_dir, mask_filename)
            np.save(mask_path, mask)
            img_classes[img_filename] = present_classes
            filtered_images.append(img_info)
            filtered_annotations.extend(valid_annotations)

    # Update the original JSON file with filtered data
    filtered_categories = [cat for cat in cats if any(cat['id'] == ann['category_id'] for ann in filtered_annotations)]
    filtered_coco = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories,
    }

    with open(annotations_files, 'w') as f:
        json.dump(filtered_coco, f, indent=4)

    # Save the class-to-image mapping in a separate JSON
    img_classes_file = os.path.join(output_dir, 'img_classes.json')
    with open(img_classes_file, 'w') as json_file:
        json.dump(img_classes, json_file, indent=4)

    # Debugging: Print the number of masks per class
    print("Numero di maschere salvate per ogni classe:")
    for class_name, count in class_mask_count.items():
        print(f"Classe '{class_name}': {count} maschere")

    print(f"Mask generation completed. Masks saved in: {output_dir}")
    print(f"Filtered image classes JSON saved in: {img_classes_file}")


def main():

    # Percorsi dei file di annotazioni non filtrate e dei file di annotazioni filtrati da generare
    train_annotations_file = '/work/tesi_cbellucci/coco/annotations/instances_train2017.json'
    out_train_ann_file = '/work/tesi_cbellucci/coco/annotations/3class_train.json'
    val_ann_file = '/work/tesi_cbellucci/coco/annotations/instances_val2017.json'
    out_val_ann_file = '/work/tesi_cbellucci/coco/annotations/3class_val.json'

    # Crea il json (formato coco) con dataset filtrato
    dc = DatasetCreator(train_annotations_file, val_ann_file, out_train_ann_file, out_val_ann_file)
    # you can filter by class names (eg. class_names=['person', 'car']) or by number of classes (eg. n_classes=5)
    dc.initialize_filters(n_classes=3)
    dc.get_filtered_datasets()


    #GENERATE MASKS
    #genera le maschere per le immagini selezionate, filtrate per numero di classi o per nome delle classi

    #path delle immagini
    train_images_dir = '/work/tesi_cbellucci/coco/images/train'
    val_images_dir = '/work/tesi_cbellucci/coco/images/val'

    # path delle maschere
    val_mask_dir = '/work/tesi_cbellucci/coco/images/val_masks/3class'
    os.makedirs(val_mask_dir, exist_ok=True)
    train_mask_dir = '/work/tesi_cbellucci/coco/images/train_masks/3class'
    os.makedirs(train_mask_dir, exist_ok=True)

    generate_mask(annotations_files=out_val_ann_file, images_dir=val_images_dir, output_dir=val_mask_dir, area_threshold=0.05, max_samples_per_class=100)
    generate_mask(annotations_files=out_train_ann_file, images_dir=train_images_dir, output_dir=train_mask_dir, area_threshold=0.05, max_samples_per_class=1000)

if __name__ == '__main__':
    main()