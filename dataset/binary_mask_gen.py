import os
import shutil
import cv2
import numpy as np
from pathlib import Path

class BinaryMaskGenerator:

    def __init__(self, dest_folder):
        self.dest_folder = Path(dest_folder)

        # Crea la cartella di destinazione se non esiste
        self.dest_folder.mkdir(parents=True, exist_ok=True)

        # Legge il colore target dal file salvato
        with open("target_color.txt", "r") as f:
            b, g, r = map(int, f.read().split(","))
            self.target_color = np.array([b, g, r], dtype=np.uint8)

    def extract_images(self, src_folder, dest_folder):
        """
        Estrai solo le immagini con suffisso '_x.png' dalla cartella di origine alla cartella di destinazione.
        """
        src_folder = Path(src_folder)
        dest_folder = Path(dest_folder)

        # Crea la cartella di destinazione se non esiste
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Copia le immagini con suffisso '_x.png' nella cartella di destinazione
        for file in src_folder.glob('*_x.png'):
            if file.is_file():
                shutil.copy(file, dest_folder / file.name)
                print(f"Immagine '{file.name}' copiata nella cartella '{dest_folder}'")

    def generate_mask(self, image_path: str, dest_folder: Path):
        """
        Genera una maschera binaria per l'immagine specificata.
        :param image_path: Percorso dell'immagine da elaborare.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"L'immagine '{image_path}' non può essere caricata.")

        # Calcola la distanza tra ogni pixel e il colore target
        distance = np.linalg.norm(image - self.target_color, axis=2)

        # Crea una maschera per i pixel vicini al colore target
        mask = np.where(distance == 0, 1, 0).astype(np.uint8) # 1 se il pixel è uguale al colore target, 0 altrimenti

        # Salva la maschera
        mask_output_path = dest_folder / f"{Path(image_path).stem.replace('_x', '_y')}.png"
        #save img in binary format
        cv2.imwrite(str(mask_output_path), mask * 255, )

    def process_all_images(self, dest_folder: Path):
        """
        Elabora tutte le immagini nella cartella di destinazione per creare maschere binarie.
        """
        for file in dest_folder.glob('*_x.png'):
            try:
                self.generate_mask(str(file), dest_folder)
                print(f"Maschera binaria salvata per '{file.name}'")
            except Exception as e:
                print(f"Errore durante l'elaborazione dell'immagine '{file}': {e}")

if __name__ == "__main__":
    source_folder = "samples"
    destination_folder = "samples_seg"
    mask_generator = BinaryMaskGenerator(destination_folder)

    for subfolder in os.listdir(source_folder): #generate mask for train and val folder
        source = os.path.join(source_folder, subfolder)
        dest = os.path.join(destination_folder, subfolder)

        if not os.path.isdir(source):
            continue  # skip files are not directories

        mask_generator.extract_images(source, dest)
        mask_generator.process_all_images(Path(dest))
