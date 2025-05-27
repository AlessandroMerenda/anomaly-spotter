import os
import json
from glob import glob
from torchvision import transforms
import torch
from PIL import Image

# -------- GESTIONE PYTHONPATH --------
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -------- IMPORT PERSONALIZZATI --------
from src.model import AutoencoderUNetLite
from src.overlay import generate_overlay
from src.config import MODEL_CONFIG, PATHS

# -------- CONFIG --------
DATA_ROOT = os.path.join(ROOT_DIR, "data", "mvtec_ad")

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize(MODEL_CONFIG['input_size']),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*MODEL_CONFIG['channels'], 
                        [0.5]*MODEL_CONFIG['channels'])
])

# -------- MODELLO --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoencoderUNetLite().to(device)
model.load_state_dict(torch.load(PATHS['model'], map_location=device))
model.eval()

# -------- CARICA SOGLIE --------
with open(PATHS['thresholds_json']) as f:
    thresholds = json.load(f)

# -------- LOOP SU CATEGORIE DIFETTOSE --------
for entry in thresholds:
    category = entry["category"]                # es: screw_scratch_head
    threshold = entry["threshold"]              # soglia pixel
    macro, sub = category.split("_", 1)         # macro = screw, sub = scratch_head

    image_dir = os.path.join(DATA_ROOT, macro, "test", sub)
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))

    print(f"\n📁 {category} - {len(image_paths)} immagini")

    for path in image_paths:
        filename = os.path.basename(path)
        save_dir = os.path.join(PATHS['all_results'], "overlay", category)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename.replace(".png", "_overlay.png"))

        generate_overlay(
            image_path=path,
            model=model,
            transform=transform,
            threshold_pixel=threshold,
            save_path=save_path
        )
