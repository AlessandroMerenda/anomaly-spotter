import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from torchvision import transforms

# -------- GESTIONE PYTHONPATH --------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -------- IMPORT --------
from src.extract_metrics import extract_image_metrics
from src.metrics import find_best_threshold_min_recall
from src.model import AutoencoderUNetLite

# -------- CONFIGURAZIONE --------

CATEGORIES = {
    "screw": [
        "good",
        "manipulated_front",
        "scratch_head",
        "scratch_neck",
        "thread_side",
        "thread_top"
    ],
    "capsule": [
        "good",
        "crack",
        "faulty_imprint",
        "poke",
        "scratch",
        "squeeze"
    ],
    "hazelnut": [
        "good",
        "crack",
        "cut",
        "hole",
        "print"
    ]
}

RECALL_LEVELS = [1.0, 0.98, 0.95, 0.9, 0.85, 0.8]

DATA_ROOT = os.path.join(ROOT_DIR, "data", "mvtec_ad")
OUTPUT_CSV_DIR = os.path.join(ROOT_DIR, "outputs", "stats")
THRESHOLD_OUTPUT_CSV = os.path.join(ROOT_DIR, "outputs", "thresholds.csv")
THRESHOLD_OUTPUT_JSON = os.path.join(ROOT_DIR, "outputs", "thresholds.json")
MODEL_WEIGHTS = os.path.join(ROOT_DIR, "outputs", "model.pth")

# -------- TRASFORMAZIONI --------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------- CARICA MODELLO --------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoencoderUNetLite().to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.eval()

# -------- ESTRAZIONE + SOGLIE --------

results = []

for category, subfolders in CATEGORIES.items():
    for sub in subfolders:
        category_name = f"{category}_{sub}"
        image_folder = os.path.join(DATA_ROOT, category, "test", sub)
        output_csv = os.path.join(OUTPUT_CSV_DIR, f"{category_name}_metrics.csv")
        label = 0 if sub == "good" else 1

        # Estrai metriche per tutte le immagini della sottoclasse
        extract_image_metrics(
            category_name=category_name,
            image_folder=image_folder,
            label=label,
            model=model,
            transform=transform,
            output_csv=output_csv
        )

        # Calcola soglia SOLO per i difettosi
        if label == 1:
            df = pd.read_csv(output_csv)
            y_true = df["label"]
            y_scores = df["mean"]

            for recall_min in RECALL_LEVELS:
                best = find_best_threshold_min_recall(y_true, y_scores, min_recall=recall_min)
                if best:
                    best["category"] = category_name
                    best["recall_level"] = recall_min  # aggiungi il livello di recall effettivo
                    print(f"[✓] {category_name} → soglia: {best['threshold']:.6f} (recall ≥ {recall_min})")
                    results.append(best)
                    break
            else:
                print(f"[✗] Nessuna soglia trovata per {category_name} (neanche con recall = {RECALL_LEVELS[-1]})")

# -------- SALVA CSV + JSON --------

os.makedirs(os.path.dirname(THRESHOLD_OUTPUT_CSV), exist_ok=True)
df_out = pd.DataFrame(results)
df_out.to_csv(THRESHOLD_OUTPUT_CSV, index=False)
print(f"\n✅ Soglie salvate in: {THRESHOLD_OUTPUT_CSV}")

json_results = []
for row in results:
    json_results.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                         for k, v in row.items()})

with open(THRESHOLD_OUTPUT_JSON, "w") as f:
    json.dump(json_results, f, indent=2)
print(f"✅ Soglie serializzate in: {THRESHOLD_OUTPUT_JSON}")