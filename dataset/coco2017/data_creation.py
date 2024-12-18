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
        self.classes = classes

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

def generate_mask(annotations_files, images_dir, output_dir):
    """
    FOR NOW GENERATE MASKS FOR all the images
    Generate binary masks from COCO annotations files, save them in output_dir as the same name of the image + '_mask.npy'.
    Also saves a JSON mapping of the classes and their channels for each image.
    :param annotations_files: Path to COCO annotations file.
    :param images_dir: Directory containing images.
    :param output_dir: Directory where masks will be saved.
    """
    print('generating masks in ', output_dir, ' from annotations in ', annotations_files)
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
    # Path to save the JSONs file
    json_img_info = os.path.join(output_dir, 'img_classes.json')

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
    with open(json_img_info, 'w') as json_file:
        json.dump(img_classes, json_file, indent=4)

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

    # Struttura per il nuovo dataset filtrato
    filtered_annotations = []
    filtered_images = []
    selected_img_ids = set()

    for img_id in tqdm(img_ids, desc="Filtering images by mask area ..."):
        # Load image information
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        mask_path = os.path.join(mask_dir, img_filename.split('.')[0] + '_mask.npy')
        mask = np.load(mask_path)
        #calcola l'area bianca
        white_area = np.sum(mask > 0) / mask.size
        if white_area > area_threshold:
            # Aggiungi l'immagine ai dati filtrati
            selected_img_ids.add(img_id)
            filtered_images.append(img_info)

            # Aggiungi le annotazioni corrispondenti
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            filtered_annotations.extend(anns)

    # Mantieni solo le categorie usate
    used_cat_ids = {ann['category_id'] for ann in filtered_annotations}
    filtered_categories = [cat for cat in coco.loadCats(coco.getCatIds()) if cat['id'] in used_cat_ids]

    # Crea la struttura del nuovo dataset
    filtered_coco = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories,
    }

    # Salva il nuovo dataset in un file JSON
    with open(output_dir, 'w') as f:
        json.dump(filtered_coco, f, indent=4)

    print(f"Filtraggio completato: {len(filtered_images)} immagini selezionate su {len(img_ids)}.")
    print(f"Nuovo file JSON salvato in: {output_dir}")

def main():

    # Percorsi dei file di annotazioni non filtrate e dei file di annotazioni filtrati da generare
    train_annotations_file = '/work/tesi_cbellucci/coco/annotations/instances_train2017.json'
    out_train_ann_file = '/work/tesi_cbellucci/coco/annotations/10class_train.json'
    val_ann_file = '/work/tesi_cbellucci/coco/annotations/instances_val2017.json'
    out_val_ann_file = '/work/tesi_cbellucci/coco/annotations/10class_val.json'

    # Crea il json (formato coco) con dataset filtrato
    dc = DatasetCreator(train_annotations_file, val_ann_file, out_train_ann_file, out_val_ann_file)
    # you can filter by class names (eg. class_names=['person', 'car']) or by number of classes (eg. n_classes=5)
    dc.initialize_filters(n_classes=1)
    dc.get_filtered_datasets()


    #GENERATE MASKS
    #genera le maschere per le immagini selezionate, filtrate per numero di classi o per nome delle classi

    #path delle immagini
    train_images_dir = '/work/tesi_cbellucci/coco/images/train'
    val_images_dir = '/work/tesi_cbellucci/coco/images/val'

    # path delle maschere
    val_mask_dir = '/work/tesi_cbellucci/coco/images/val_masks/10class'
    os.makedirs(val_mask_dir, exist_ok=True)
    train_mask_dir = '/work/tesi_cbellucci/coco/images/train_masks/10class'
    os.makedirs(train_mask_dir, exist_ok=True)

    generate_mask(annotations_files=out_train_ann_file, images_dir=val_images_dir, output_dir=val_mask_dir)
    generate_mask(out_train_ann_file, train_images_dir, train_mask_dir)


    # KEEP JUST THE IMAGES WITH MASKS WITH WHITE AREA > threshold = 0.05
    train_ann_file = out_train_ann_file #the output of generate_mask function is the input of filter_mask_from_area
    val_ann_file = out_val_ann_file
    #path del json filtrato con solo le immagini che hanno maschere con area bianca sopra una certa soglia
    out_train_ann_file = '/work/tesi_cbellucci/coco/annotations/10class_train_filtered.json'
    out_val_ann_file = '/work/tesi_cbellucci/coco/annotations/10class_val_filtered.json'
    filter_mask_from_area(annotations_files=val_ann_file, mask_dir=val_mask_dir, output_dir=out_val_ann_file, area_threshold=0.05)
    filter_mask_from_area(annotations_files=train_ann_file, mask_dir=train_mask_dir, output_dir=out_train_ann_file,area_threshold= 0.05)

if __name__ == '__main__':
    main()