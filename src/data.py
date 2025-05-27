import os
from torch.utils.data import Dataset
from PIL import Image

class MVTecDataset(Dataset):
    """
    Dataset custom per il caricamento delle immagini dal dataset MVTec AD.

    Supporta lettura ricorsiva di immagini da directory strutturate come:
    - data/mvtec_ad/<categoria>/train/good/
    - data/mvtec_ad/<categoria>/test/<tipo_difetto>/

    Parametri:
        root_dir (str): percorso alla directory da cui leggere le immagini.
        transform (callable): trasformazioni da applicare alle immagini (es. resize, normalize).
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # Lista completa dei path assoluti delle immagini

        # Estensioni supportate
        IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

        # Cammina ricorsivamente in tutte le sottocartelle della root_dir
        for root, _, files in os.walk(root_dir):
            for fname in sorted(files):
                # Aggiungi al dataset solo file immagine validi
                if fname.lower().endswith(IMG_EXTENSIONS):
                    full_path = os.path.join(root, fname)
                    self.image_paths.append(full_path)

    def __len__(self):
        """
        Ritorna il numero totale di immagini trovate.
        Necessario per compatibilità con PyTorch DataLoader.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Carica e ritorna l'immagine all'indice `idx`, eventualmente trasformata.
        """
        img_path = self.image_paths[idx]

        # Prova ad aprire l'immagine (con gestione errori)
        try:
            image = Image.open(img_path).convert('RGB')  # Assicura 3 canali
        except Exception as e:
            raise RuntimeError(f"Errore nel caricamento dell'immagine {img_path}: {e}")

        # Applica le trasformazioni (se fornite)
        if self.transform:
            image = self.transform(image)

        # Ritorna l'immagine trasformata (eventualmente aggiungibile anche img_path)
        return image, img_path

