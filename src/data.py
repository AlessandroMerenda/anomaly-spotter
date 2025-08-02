import os
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Optional, Callable
import logging

# Import utilities per error handling
from .utils.logging_utils import (
    setup_logger, DataError, handle_exception, 
    validate_image_file, safe_execute
)

class MVTecDataset(Dataset):
    """
    Dataset custom per il caricamento delle immagini dal dataset MVTec AD.

    Supporta lettura ricorsiva di immagini da directory strutturate come:
    - data/mvtec_ad/<categoria>/train/good/
    - data/mvtec_ad/<categoria>/test/<tipo_difetto>/

    Parametri:
        root_dir (str): percorso alla directory da cui leggere le immagini.
        transform (callable): trasformazioni da applicare alle immagini (es. resize, normalize).
        logger (logging.Logger): Logger per monitoraggio (opzionale).
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None, 
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger("MVTecDataset")
        self.root_dir = str(Path(root_dir).resolve())
        self.transform = transform
        self.image_paths: List[str] = []

        # Estensioni supportate
        self.SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        try:
            self._load_image_paths()
            self.logger.info(f"Dataset inizializzato: {len(self.image_paths)} immagini da {self.root_dir}")
        except Exception as e:
            handle_exception(self.logger, e, f"Inizializzazione dataset da {root_dir}")
            raise DataError(f"Impossibile inizializzare dataset da {root_dir}", {"original_error": str(e)})

    def _load_image_paths(self) -> None:
        """Carica tutti i percorsi delle immagini valide dalla directory root."""
        if not os.path.exists(self.root_dir):
            raise DataError(f"Directory non trovata: {self.root_dir}")
        
        if not os.path.isdir(self.root_dir):
            raise DataError(f"Il percorso non è una directory: {self.root_dir}")

        # Cammina ricorsivamente in tutte le sottocartelle
        loaded_count = 0
        for root, dirs, files in os.walk(self.root_dir):
            for fname in sorted(files):
                if self._is_valid_image_file(fname):
                    full_path = os.path.join(root, fname)
                    
                    # Validazione aggiuntiva del file
                    if self._validate_image_path(full_path):
                        self.image_paths.append(full_path)
                        loaded_count += 1

        if loaded_count == 0:
            self.logger.warning(f"Nessuna immagine valida trovata in {self.root_dir}")
        else:
            self.logger.debug(f"Caricate {loaded_count} immagini da {self.root_dir}")

    def _is_valid_image_file(self, filename: str) -> bool:
        """Controlla se un file ha un'estensione immagine supportata."""
        return filename.lower().endswith(self.SUPPORTED_EXTENSIONS)

    def _validate_image_path(self, file_path: str) -> bool:
        """Valida che un percorso immagine sia accessibile."""
        try:
            # Controlli di sicurezza e accessibilità
            if not os.path.exists(file_path):
                self.logger.debug(f"File non esistente saltato: {file_path}")
                return False
            
            if not os.access(file_path, os.R_OK):
                self.logger.warning(f"File non leggibile saltato: {file_path}")
                return False
                
            # Controllo dimensione file (evita file corrotti di 0 bytes)
            if os.path.getsize(file_path) == 0:
                self.logger.warning(f"File vuoto saltato: {file_path}")
                return False
                
            return True
            
        except OSError as e:
            self.logger.warning(f"Errore accesso file {file_path}: {e}")
            return False

    def __len__(self) -> int:
        """
        Ritorna il numero totale di immagini trovate.
        Necessario per compatibilità con PyTorch DataLoader.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Carica e ritorna l'immagine all'indice `idx`, eventualmente trasformata.
        
        Args:
            idx: Indice dell'immagine da caricare
            
        Returns:
            Immagine trasformata (se transform specificato)
            
        Raises:
            DataError: Se l'immagine non può essere caricata
        """
        if idx < 0 or idx >= len(self.image_paths):
            raise DataError(f"Indice non valido: {idx} (dataset size: {len(self.image_paths)})")

        img_path = self.image_paths[idx]
        
        try:
            # Validazione percorso per sicurezza
            validated_path = validate_image_file(img_path)
            
            # Caricamento immagine con gestione errori robusta
            image = self._load_image_safe(str(validated_path))
            
            # Applicazione trasformazioni
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception as e:
                    self.logger.error(f"Errore nelle trasformazioni per {img_path}: {e}")
                    raise DataError(f"Errore trasformazione immagine", {"path": img_path, "transform_error": str(e)})
            
            return image, img_path
            
        except DataError:
            # Re-raise DataError as-is
            raise
        except Exception as e:
            handle_exception(self.logger, e, f"Caricamento immagine {img_path}")
            raise DataError(f"Impossibile caricare immagine: {img_path}", {"original_error": str(e)})

    def _load_image_safe(self, img_path: str) -> Image.Image:
        """
        Carica un'immagine in modo sicuro con controlli di validità.
        
        Args:
            img_path: Percorso dell'immagine
            
        Returns:
            Immagine PIL caricata
            
        Raises:
            DataError: Se l'immagine non può essere caricata
        """
        try:
            # Tentativo di apertura
            image = Image.open(img_path)
            
            # Verifica che l'immagine sia valida leggendo i metadati
            _ = image.size  # Trigger load dei metadati
            
            # Converti sempre a RGB per consistenza
            if image.mode != 'RGB':
                image = image.convert('RGB')
                self.logger.debug(f"Immagine convertita a RGB: {img_path}")
            
            # Controllo dimensioni minime ragionevoli
            if image.size[0] < 10 or image.size[1] < 10:
                raise DataError(f"Immagine troppo piccola: {image.size}", {"path": img_path})
            
            return image
            
        except Image.UnidentifiedImageError:
            raise DataError(f"Formato immagine non riconosciuto o file corrotto", {"path": img_path})
        except FileNotFoundError:
            raise DataError(f"File immagine non trovato", {"path": img_path})
        except PermissionError:
            raise DataError(f"Permessi insufficienti per leggere l'immagine", {"path": img_path})
        except OSError as e:
            raise DataError(f"Errore OS durante caricamento immagine: {e}", {"path": img_path})

    def get_stats(self) -> dict:
        """
        Ritorna statistiche sul dataset caricato.
        
        Returns:
            Dict con statistiche del dataset
        """
        stats = {
            'total_images': len(self.image_paths),
            'root_directory': self.root_dir,
            'extensions_found': set(),
            'directories_scanned': set()
        }
        
        for path in self.image_paths:
            # Estensioni trovate
            ext = Path(path).suffix.lower()
            stats['extensions_found'].add(ext)
            
            # Directory scansionate
            parent_dir = str(Path(path).parent)
            stats['directories_scanned'].add(parent_dir)
        
        # Converti set in liste per serializzazione
        stats['extensions_found'] = sorted(list(stats['extensions_found']))
        stats['directories_scanned'] = sorted(list(stats['directories_scanned']))
        
        return stats

