import json
from pycocotools.coco import COCO


class COCOFilter:
    def __init__(self, coco, n_samples, n_classes):
        """
        Classe helper per filtrare le categorie e le immagini dal dataset COCO.
        """
        self.coco = coco
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.classes = None
        self.selected_cat_ids = None
        self.selected_img_ids = set()

    def set_classes(self):
        """
        Seleziona le classi dal file json di annotazioni per il dataset.
        """
        # Filter categories with at least `n_samples` images, excluding 'person'
        filtered_categories = [
            cat for cat in self.coco.loadCats(self.coco.getCatIds())
            if len(self.coco.getAnnIds(catIds=[cat['id']])) >= self.n_samples and cat['name'] != 'person'
        ]

        # Sort categories by the number of images in descending order
        filtered_categories.sort(
            key=lambda x: len(self.coco.getAnnIds(catIds=[x['id']])),
            reverse=True
        )

        # Select the top `n_classes` categories
        self.classes = filtered_categories[:self.n_classes]
        self.selected_cat_ids = [cat['id'] for cat in self.classes]

        # Select a fixed number of images per class
        self.selected_img_ids = self.take_n_images(self.n_samples)

    def take_n_images(self, num_images):
        """
        Seleziona un numero massimo di immagini per ogni classe selezionata.
        """
        img_ids = set()
        for cat_id in self.selected_cat_ids:
            # Get image IDs associated with the current category
            cat_img_ids = self.coco.getImgIds(catIds=[cat_id])
            # Select up to `num_images` images per category
            img_ids.update(cat_img_ids[:num_images])
        return img_ids


class DatasetCreator:
    def __init__(self, train_ann_file, val_ann_file, output_train, output_val):
        """
        Inizializza la classe DatasetCreator.
        """
        self.train_coco = COCO(train_ann_file)
        self.val_coco = COCO(val_ann_file)
        self.train_ann_file = train_ann_file
        self.output_train = output_train
        self.val_ann_file = val_ann_file
        self.output_val = output_val
        self.train_filter = None
        self.val_filter = None

    def initialize_filters(self, n_train_samples=1000, n_val_samples=100, n_classes=10):
        """
        Inizializza i filtri COCO per il training set e il validation set.
        """
        # Initialize the training filter
        self.train_filter = COCOFilter(self.train_coco, n_train_samples, n_classes)
        self.train_filter.set_classes()

        # Initialize the validation filter with a different number of samples
        self.val_filter = COCOFilter(self.val_coco, n_val_samples, n_classes)
        self.val_filter.selected_cat_ids = self.train_filter.selected_cat_ids
        self.val_filter.classes = self.train_filter.classes
        self.val_filter.selected_img_ids = self.val_filter.take_n_images(n_val_samples)

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
        self.save_filtered_dataset(self.train_filter, self.output_train)
        self.save_filtered_dataset(self.val_filter, self.output_val)


if __name__ == '__main__':
    # Percorsi
    train_annotations_file = '/work/tesi_cbellucci/coco/annotations/instances_train2017.json'
    output_train = '/work/tesi_cbellucci/coco/annotations/filtered_instances_train2017.json'
    val_ann_file = '/work/tesi_cbellucci/coco/annotations/instances_val2017.json'
    output_val = '/work/tesi_cbellucci/coco/annotations/filtered_instances_val2017.json'

    # Crea il json (formato coco) con dataset filtrato
    dc = DatasetCreator(train_annotations_file, val_ann_file, output_train, output_val)
    dc.initialize_filters(n_train_samples=1000, n_val_samples=100, n_classes=10)
    dc.get_filtered_datasets()
