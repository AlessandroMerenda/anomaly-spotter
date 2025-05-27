import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

def extract_image_metrics(category_name, image_folder, label, model, transform, output_csv):
    """
    Calcola gli anomaly score per ogni immagine in una cartella e salva in CSV.

    Args:
        category_name (str): Nome logico della categoria (es. screw_scratch_head)
        image_folder (str): Percorso alla cartella immagini da analizzare
        label (int): 0 = good, 1 = difettosa
        model (torch.nn.Module): Modello autoencoder già addestrato
        transform (torchvision.transforms): Trasformazioni da applicare
        output_csv (str): Path del file CSV di output
    """
    model.eval()
    device = next(model.parameters()).device
    records = []

    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')

    for fname in tqdm(sorted(os.listdir(image_folder)), desc=f"[{category_name}]"):
        if not fname.lower().endswith(IMG_EXTENSIONS):
            continue

        fpath = os.path.join(image_folder, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Errore su {fname}: {e}")
            continue

        with torch.no_grad():
            recon = model(tensor)

        # Calcolo errore per pixel (denormalizzato come nel notebook)
        input_img = tensor.squeeze().cpu()
        recon_img = recon.squeeze().cpu()
        input_np = (input_img * 0.5 + 0.5).permute(1, 2, 0).numpy()
        recon_np = (recon_img * 0.5 + 0.5).permute(1, 2, 0).numpy()
        diff = np.abs(input_np - recon_np)

        # Score immagine: media, max, std, percentile
        mean_score = diff.mean()
        max_score = diff.max()
        std_score = diff.std()
        percentile_90 = np.percentile(diff, 90)

        records.append({
            "filename": fname,
            "label": label,
            "mean": mean_score,
            "max": max_score,
            "std": std_score,
            "percentile_90": percentile_90
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ File salvato: {output_csv}")