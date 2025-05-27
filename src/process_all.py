import os
import sys
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
from scipy.ndimage import gaussian_filter

# Gestione PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import AutoencoderUNetLite
from src.config import MODEL_CONFIG, ANOMALY_CONFIG, PATHS, CATEGORIES, DATA_ROOT

def process_image(model, image_path, device):
    """Processa una singola immagine."""
    # Trasformazioni
    transform = transforms.Compose([
        transforms.Resize(MODEL_CONFIG['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*MODEL_CONFIG['channels'], 
                           [0.5]*MODEL_CONFIG['channels'])
    ])
    
    # Carica e preprocessa
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Genera ricostruzione
    with torch.no_grad():
        reconstructed = model(image_tensor)
    
    # Calcola mappa anomalie
    diff = torch.abs(image_tensor - reconstructed)
    anomaly_map = diff.mean(dim=1).squeeze().cpu().numpy()
    
    # Applica smoothing gaussiano
    anomaly_map = gaussian_filter(anomaly_map, sigma=ANOMALY_CONFIG['gaussian_sigma'])
    
    # Normalizzazione min-max
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Calcola statistiche
    binary_mask = anomaly_map > ANOMALY_CONFIG['threshold_pixel']
    stats = {
        'mean_score': float(anomaly_map.mean()),
        'max_score': float(anomaly_map.max()),
        'anomaly_percentage': float(100 * binary_mask.mean())
    }
    
    return anomaly_map, stats

def process_category(model, category, device):
    """Processa tutte le immagini di una categoria."""
    results = {}
    
    # Processa ogni tipo di difetto
    for defect_type in CATEGORIES[category]:
        print(f"\n🔍 Processing {category}/{defect_type}...")
        
        # Directory input/output
        test_dir = os.path.join(DATA_ROOT, category, "test", defect_type)
        out_dir = os.path.join(PATHS['all_results'], category, defect_type)
        os.makedirs(out_dir, exist_ok=True)
        
        defect_results = []
        
        # Processa ogni immagine
        for img_name in tqdm(sorted(os.listdir(test_dir))):
            if not img_name.endswith('.png'):
                continue
                
            img_path = os.path.join(test_dir, img_name)
            anomaly_map, stats = process_image(model, img_path, device)
            
            # Salva mappa anomalie
            np.save(os.path.join(out_dir, f"{img_name[:-4]}_anomaly.npy"), anomaly_map)
            
            # Aggiungi statistiche
            stats['image'] = img_name
            defect_results.append(stats)
        
        # Salva risultati del tipo di difetto
        results[defect_type] = defect_results
        
        # Salva statistiche in JSON
        with open(os.path.join(out_dir, "stats.json"), 'w') as f:
            json.dump(defect_results, f, indent=4)
    
    return results

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")
    
    # Carica modello
    print("\n📥 Loading model...")
    model = AutoencoderUNetLite().to(device)
    model.load_state_dict(torch.load(PATHS['model'], map_location=device))
    model.eval()
    
    # Processa ogni categoria
    all_results = {}
    for category in CATEGORIES:
        print(f"\n📁 Processing category: {category}")
        all_results[category] = process_category(model, category, device)
    
    # Salva risultati globali
    print("\n💾 Saving global results...")
    with open(os.path.join(PATHS['all_results'], "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n✅ Processing completed!")

if __name__ == "__main__":
    main() 