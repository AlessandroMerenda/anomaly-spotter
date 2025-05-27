import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

# Gestione PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import ANOMALY_CONFIG, VISUALIZATION_CONFIG

def save_overlay_triplet(input_np, diff_np, out_path, mask_threshold=None):
    """
    Salva un'immagine con 3 pannelli affiancati: originale, heatmap, binaria.
    Ottimizzata per una migliore visualizzazione dell'errore.
    """
    # Usa il threshold dalle configurazioni se non specificato
    if mask_threshold is None:
        mask_threshold = ANOMALY_CONFIG['threshold_pixel']
    
    # Calcola errore medio per canale e applica smoothing
    diff_gray = diff_np.mean(axis=2)
    diff_smooth = gaussian_filter(diff_gray, sigma=ANOMALY_CONFIG['gaussian_sigma'])
    
    # Normalizzazione min-max per migliorare il contrasto
    diff_norm = (diff_smooth - diff_smooth.min()) / (diff_smooth.max() - diff_smooth.min() + 1e-8)
    
    # Calcola maschera binaria usando Otsu thresholding
    try:
        thresh = threshold_otsu(diff_norm)
        mask_binary = (diff_norm > thresh).astype(np.uint8)
    except:
        # Fallback su threshold fisso se Otsu fallisce
        mask_binary = (diff_norm > mask_threshold).astype(np.uint8)

    # Crea figura
    fig, axs = plt.subplots(1, 3, figsize=VISUALIZATION_CONFIG['figure_size'])
    fig.suptitle('Anomaly Detection Results', fontsize=VISUALIZATION_CONFIG['font_size'] + 6)

    # Plot originale
    axs[0].imshow(input_np)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    # Plot heatmap con colormap ottimizzata
    heatmap = axs[1].imshow(diff_norm, cmap=VISUALIZATION_CONFIG['colormap'], interpolation='nearest')
    axs[1].set_title("Anomaly Heatmap")
    axs[1].axis('off')
    plt.colorbar(heatmap, ax=axs[1], fraction=0.046, pad=0.04)

    # Plot maschera binaria
    axs[2].imshow(mask_binary, cmap='gray')
    axs[2].set_title(f"Binary Mask (Otsu)")
    axs[2].axis('off')

    # Aggiunge statistiche
    stats_text = f'Max Error: {diff_smooth.max():.4f}\n'
    stats_text += f'Mean Error: {diff_smooth.mean():.4f}\n'
    stats_text += f'Threshold: {thresh if "thresh" in locals() else mask_threshold:.4f}'
    fig.text(0.02, 0.02, stats_text, fontsize=VISUALIZATION_CONFIG['font_size'], family='monospace')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()


def generate_overlay(image_path, model, transform, threshold_pixel=None, save_path=None):
    """
    Genera una visualizzazione affiancata (input, heatmap, binaria) da un'immagine.
    """
    # Usa il threshold dalle configurazioni se non specificato
    if threshold_pixel is None:
        threshold_pixel = ANOMALY_CONFIG['threshold_pixel']
        
    device = next(model.parameters()).device

    # Caricamento e preprocessing
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(tensor)

    # Denormalizzazione e diff
    input_np = (tensor.squeeze().cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy()
    recon_np = (recon.squeeze().cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy()
    diff_np = np.abs(input_np - recon_np)

    if save_path is None:
        save_path = image_path.replace(".png", "_overlay.png")

    # Chiamata alla funzione di salvataggio tripla
    save_overlay_triplet(input_np, diff_np, save_path, mask_threshold=threshold_pixel)
