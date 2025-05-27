import os

# Percorsi base
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT_DIR, "data", "mvtec_ad")
OUTPUT_ROOT = os.path.join(ROOT_DIR, "outputs")

# Configurazione modello
MODEL_CONFIG = {
    'input_size': (128, 128),
    'channels': 3,
    'latent_dim': 512
}

# Parametri training
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'validation_split': 0.2,
    'num_workers': 4
}

# Configurazione anomaly detection
ANOMALY_CONFIG = {
    'threshold_pixel': 0.015,
    'gaussian_sigma': 1.0,
    'min_recall': 0.95
}

# Categorie e tipi di difetti
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

# Configurazione visualizzazione
VISUALIZATION_CONFIG = {
    'figure_size': (15, 5),
    'dpi': 150,
    'colormap': 'inferno',
    'font_size': 10
}

# Percorsi output
PATHS = {
    'model': os.path.join(OUTPUT_ROOT, "model.pth"),
    'thresholds_csv': os.path.join(OUTPUT_ROOT, "thresholds.csv"),
    'thresholds_json': os.path.join(OUTPUT_ROOT, "thresholds.json"),
    'training_plot': os.path.join(OUTPUT_ROOT, "training_history.png"),
    'test_results': os.path.join(OUTPUT_ROOT, "test_results"),
    'all_results': os.path.join(OUTPUT_ROOT, "all_results")
} 