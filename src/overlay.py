import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import logging
from typing import Optional, Tuple
from pathlib import Path

# Gestione PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import ANOMALY_CONFIG, VISUALIZATION_CONFIG
from src.utils.logging_utils import (
    setup_logger, ModelError, DataError, handle_exception, 
    validate_image_file, safe_execute
)

def save_overlay_triplet(input_np: np.ndarray, diff_np: np.ndarray, out_path: str, 
                        mask_threshold: Optional[float] = None, logger: Optional[logging.Logger] = None) -> bool:
    """
    Salva un'immagine con 3 pannelli affiancati: originale, heatmap, binaria.
    Ottimizzata per una migliore visualizzazione dell'errore con error handling robusto.
    
    Args:
        input_np: Immagine originale come numpy array
        diff_np: Mappa delle differenze come numpy array
        out_path: Percorso di salvataggio
        mask_threshold: Soglia per maschera binaria (opzionale)
        logger: Logger per monitoring
        
    Returns:
        True se successo, False altrimenti
    """
    if logger is None:
        logger = setup_logger("overlay")
    
    try:
        # Validazione input
        if input_np is None or diff_np is None:
            raise DataError("Array numpy input non validi")
        
        if input_np.shape[:2] != diff_np.shape[:2]:
            raise DataError(f"Dimensioni non compatibili: input {input_np.shape}, diff {diff_np.shape}")
        
        # Usa il threshold dalle configurazioni se non specificato
        if mask_threshold is None:
            mask_threshold = ANOMALY_CONFIG['threshold_pixel']
        
        # Calcola errore medio per canale e applica smoothing
        if len(diff_np.shape) == 3:
            diff_gray = diff_np.mean(axis=2)
        else:
            diff_gray = diff_np
        
        # Validazione array
        if np.isnan(diff_gray).any() or np.isinf(diff_gray).any():
            logger.warning("Valori non validi nella mappa differenze - applicando clipping")
            diff_gray = np.nan_to_num(diff_gray, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Smoothing
        try:
            diff_smooth = gaussian_filter(diff_gray, sigma=ANOMALY_CONFIG['gaussian_sigma'])
        except Exception as e:
            logger.warning(f"Gaussian filter fallito: {e} - usando differenze originali")
            diff_smooth = diff_gray
        
        # Normalizzazione min-max per migliorare il contrasto
        diff_min, diff_max = diff_smooth.min(), diff_smooth.max()
        if diff_max - diff_min > 1e-8:
            diff_norm = (diff_smooth - diff_min) / (diff_max - diff_min)
        else:
            logger.warning("Range differenze troppo piccolo - usando valori uniformi")
            diff_norm = np.zeros_like(diff_smooth)
        
        # Calcola maschera binaria usando Otsu thresholding
        try:
            thresh = threshold_otsu(diff_norm)
            mask_binary = (diff_norm > thresh).astype(np.uint8)
            logger.debug(f"Otsu threshold calcolato: {thresh:.4f}")
        except Exception as e:
            # Fallback su threshold fisso se Otsu fallisce
            logger.warning(f"Otsu thresholding fallito: {e} - usando threshold fisso")
            thresh = mask_threshold
            mask_binary = (diff_norm > thresh).astype(np.uint8)

        # Crea figura con gestione errori
        try:
            fig, axs = plt.subplots(1, 3, figsize=VISUALIZATION_CONFIG['figure_size'])
            fig.suptitle('Anomaly Detection Results', fontsize=VISUALIZATION_CONFIG['font_size'] + 6)

            # Plot originale
            axs[0].imshow(np.clip(input_np, 0, 1))
            axs[0].set_title("Input Image")
            axs[0].axis('off')

            # Plot heatmap con colormap ottimizzata
            heatmap = axs[1].imshow(diff_norm, cmap=VISUALIZATION_CONFIG['colormap'], 
                                   interpolation='nearest', vmin=0, vmax=1)
            axs[1].set_title("Anomaly Heatmap")
            axs[1].axis('off')
            plt.colorbar(heatmap, ax=axs[1], fraction=0.046, pad=0.04)

            # Plot maschera binaria
            axs[2].imshow(mask_binary, cmap='gray', vmin=0, vmax=1)
            axs[2].set_title(f"Binary Mask (Threshold: {thresh:.3f})")
            axs[2].axis('off')

            # Aggiunge statistiche
            stats_text = f'Max Error: {diff_smooth.max():.4f}\n'
            stats_text += f'Mean Error: {diff_smooth.mean():.4f}\n'
            stats_text += f'Anomaly Pixels: {mask_binary.sum():,}'
            fig.text(0.02, 0.02, stats_text, fontsize=VISUALIZATION_CONFIG['font_size'], 
                    family='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            plt.tight_layout()
            
            # Assicurati che la directory esista
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            plt.savefig(out_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.debug(f"Overlay salvato: {out_path}")
            return True
            
        except Exception as e:
            logger.error(f"Errore creazione figura: {e}")
            # Cleanup se figura parzialmente creata
            try:
                plt.close('all')
            except:
                pass
            return False
            
    except Exception as e:
        handle_exception(logger, e, f"Salvataggio overlay {out_path}")
        return False


def generate_overlay(image_path: str, model: torch.nn.Module, transform: transforms.Compose, 
                    threshold_pixel: Optional[float] = None, save_path: Optional[str] = None,
                    logger: Optional[logging.Logger] = None) -> bool:
    """
    Genera una visualizzazione affiancata (input, heatmap, binaria) da un'immagine.
    Con error handling completo e validazioni.
    
    Args:
        image_path: Percorso dell'immagine input
        model: Modello PyTorch per la ricostruzione
        transform: Trasformazioni da applicare
        threshold_pixel: Soglia per maschera binaria
        save_path: Percorso di salvataggio (opzionale)
        logger: Logger per monitoring
        
    Returns:
        True se successo, False altrimenti
    """
    if logger is None:
        logger = setup_logger("generate_overlay")
    
    try:
        # Validazione input
        if model is None:
            raise ModelError("Modello non fornito")
        
        if transform is None:
            raise ModelError("Trasformazioni non fornite")
        
        # Validazione percorso immagine
        validated_path = validate_image_file(image_path)
        
        # Usa il threshold dalle configurazioni se non specificato
        if threshold_pixel is None:
            threshold_pixel = ANOMALY_CONFIG['threshold_pixel']
        
        # Ottieni device dal modello
        try:
            device = next(model.parameters()).device
        except StopIteration:
            raise ModelError("Modello senza parametri")
        
        # Caricamento e preprocessing con error handling
        try:
            img = Image.open(str(validated_path)).convert("RGB")
            
            # Verifica dimensioni ragionevoli
            if img.size[0] < 10 or img.size[1] < 10:
                raise DataError(f"Immagine troppo piccola: {img.size}")
            
            if img.size[0] > 4096 or img.size[1] > 4096:
                logger.warning(f"Immagine molto grande: {img.size} - potrebbe causare problemi di memoria")
            
        except Exception as e:
            raise DataError(f"Impossibile caricare immagine: {e}")
        
        # Applicazione trasformazioni
        try:
            tensor = transform(img).unsqueeze(0).to(device)
            
            # Verifica shape tensore
            expected_shape = (1, 3, *transform.transforms[0].size)  # Assumendo Resize come primo transform
            if tensor.shape[1:] != expected_shape[1:]:
                logger.warning(f"Shape tensore inaspettata: {tensor.shape}, expected: {expected_shape}")
                
        except Exception as e:
            raise ModelError(f"Errore applicazione trasformazioni: {e}")
        
        # Inferenza con error handling
        try:
            model.eval()
            with torch.no_grad():
                recon = model(tensor)
                
            # Verifica output
            if recon.shape != tensor.shape:
                raise ModelError(f"Shape output non corretta: {recon.shape} vs {tensor.shape}")
            
            # Verifica valori validi
            if torch.isnan(recon).any() or torch.isinf(recon).any():
                raise ModelError("Output modello contiene valori non validi (NaN/Inf)")
                
        except torch.cuda.OutOfMemoryError:
            raise ModelError("Memoria GPU insufficiente per l'inferenza")
        except Exception as e:
            raise ModelError(f"Errore durante inferenza: {e}")
        
        # Denormalizzazione e calcolo differenze
        try:
            # Denormalizzazione (assumendo normalizzazione [-1, 1])
            input_np = (tensor.squeeze().cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy()
            recon_np = (recon.squeeze().cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy()
            
            # Clipping per sicurezza
            input_np = np.clip(input_np, 0, 1)
            recon_np = np.clip(recon_np, 0, 1)
            
            # Calcolo differenze
            diff_np = np.abs(input_np - recon_np)
            
        except Exception as e:
            raise ModelError(f"Errore calcolo differenze: {e}")
        
        # Percorso di salvataggio
        if save_path is None:
            save_path = str(validated_path).replace(".png", "_overlay.png")
        
        # Salvataggio overlay
        success = save_overlay_triplet(input_np, diff_np, save_path, 
                                      mask_threshold=threshold_pixel, logger=logger)
        
        if success:
            logger.info(f"Overlay generato con successo: {save_path}")
            return True
        else:
            logger.error("Fallimento generazione overlay")
            return False
            
    except Exception as e:
        handle_exception(logger, e, f"Generazione overlay per {image_path}")
        return False
    
    finally:
        # Cleanup memoria se necessario
        try:
            if 'tensor' in locals():
                del tensor
            if 'recon' in locals():
                del recon
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
