import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Gestione PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import AutoencoderUNetLite
from src.overlay import generate_overlay
from src.config import MODEL_CONFIG, PATHS, CATEGORIES, DATA_ROOT

def test_images(model, transform, test_cases):
    """Testa il modello su una lista di immagini."""
    for category, defect_type, img_number in test_cases:
        # Costruisci i percorsi
        img_path = os.path.join(DATA_ROOT, category, "test", defect_type, f"{img_number:03d}.png")
        out_dir = os.path.join(PATHS['test_results'], category, defect_type)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{img_number:03d}_result.png")
        
        print(f"\n🔍 Testing {category}/{defect_type}/{img_number:03d}.png")
        generate_overlay(img_path, model, transform, save_path=out_path)
        print(f"✨ Saved result to: {out_path}")

def main():
    # Test cases: (categoria, tipo_difetto, numero_immagine)
    TEST_CASES = [
        # Capsule
        ("capsule", "crack", 0),        # Crepa
        ("capsule", "squeeze", 0),      # Schiacciamento
        ("capsule", "good", 0),         # Normale (controllo)
        
        # Hazelnut
        ("hazelnut", "crack", 0),       # Crepa
        ("hazelnut", "hole", 0),        # Buco
        ("hazelnut", "good", 0),        # Normale (controllo)
        
        # Screw
        ("screw", "scratch_head", 0),   # Graffio sulla testa
        ("screw", "thread_side", 0),    # Difetto nella filettatura
        ("screw", "good", 0),           # Normale (controllo)
    ]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    # Carica il modello
    print("\n📥 Loading model...")
    model = AutoencoderUNetLite().to(device)
    model.load_state_dict(torch.load(PATHS['model'], map_location=device))
    model.eval()

    # Trasformazioni
    transform = transforms.Compose([
        transforms.Resize(MODEL_CONFIG['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*MODEL_CONFIG['channels'], 
                           [0.5]*MODEL_CONFIG['channels'])
    ])

    # Esegui i test
    print("\n🔬 Starting tests...")
    test_images(model, transform, TEST_CASES)
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main() 