"""
Configuration module - DEPRECATED
Questo file è mantenuto per compatibilità ma è deprecato.
Usa il nuovo sistema: src.utils.config_manager

MIGRATION GUIDE:
Vecchio: from src.config import MODEL_CONFIG
Nuovo:   from src.utils.config_manager import get_config
         config = get_config()
         model_config = config.model
"""

import warnings
from .utils.config_manager import get_config

# Emetti warning di deprecazione
warnings.warn(
    "src.config è deprecato. Usa src.utils.config_manager.get_config() invece.",
    DeprecationWarning,
    stacklevel=2
)

# Per compatibilità, esponi le configurazioni vecchio stile
def _get_legacy_config():
    """Ottieni configurazione in formato legacy per compatibilità."""
    config = get_config()
    
    return {
        'MODEL_CONFIG': {
            'input_size': config.model.input_size,
            'channels': config.model.channels,
            'latent_dim': config.model.latent_dim
        },
        'TRAINING_CONFIG': {
            'batch_size': config.training.batch_size,
            'num_epochs': config.training.num_epochs,
            'learning_rate': config.training.learning_rate,
            'validation_split': config.training.validation_split,
            'num_workers': config.training.num_workers
        },
        'ANOMALY_CONFIG': {
            'threshold_pixel': config.anomaly.threshold_pixel,
            'gaussian_sigma': config.anomaly.gaussian_sigma,
            'min_recall': config.anomaly.min_recall
        },
        'VISUALIZATION_CONFIG': {
            'figure_size': config.visualization.figure_size,
            'dpi': config.visualization.dpi,
            'colormap': config.visualization.colormap,
            'font_size': config.visualization.font_size
        },
        'CATEGORIES': {
            "screw": ["good", "manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
            "capsule": ["good", "crack", "faulty_imprint", "poke", "scratch", "squeeze"],
            "hazelnut": ["good", "crack", "cut", "hole", "print"]
        },
        'PATHS': {
            'model': config.paths.model_file,
            'thresholds_csv': config.paths.thresholds_csv,
            'thresholds_json': config.paths.thresholds_json,
            'training_plot': config.paths.training_plot,
            'test_results': config.paths.test_results_dir,
            'all_results': config.paths.all_results_dir
        },
        'DATA_ROOT': config.paths.data_root,
        'OUTPUT_ROOT': config.paths.output_root
    }

# Carica configurazione legacy
_legacy = _get_legacy_config()

# Esporta per compatibilità (CON WARNING)
MODEL_CONFIG = _legacy['MODEL_CONFIG']
TRAINING_CONFIG = _legacy['TRAINING_CONFIG'] 
ANOMALY_CONFIG = _legacy['ANOMALY_CONFIG']
VISUALIZATION_CONFIG = _legacy['VISUALIZATION_CONFIG']
CATEGORIES = _legacy['CATEGORIES']
PATHS = _legacy['PATHS']
DATA_ROOT = _legacy['DATA_ROOT']
OUTPUT_ROOT = _legacy['OUTPUT_ROOT'] 