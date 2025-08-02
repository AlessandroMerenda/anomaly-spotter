import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Optional

# Gestione PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import AutoencoderUNetLite
from src.overlay import generate_overlay
from src.utils.config_manager import get_config
from src.utils.logging_utils import (
    setup_logger, ModelError, DataError, ResourceError,
    handle_exception, validate_file_path, safe_execute
)

def test_images(model: torch.nn.Module, transform: transforms.Compose, 
                test_cases: List[Tuple[str, str, int]], config, logger: logging.Logger) -> None:
    """
    Testa il modello su una lista di immagini con error handling robusto.
    
    Args:
        model: Modello PyTorch caricato
        transform: Trasformazioni da applicare
        test_cases: Lista di tuple (categoria, tipo_difetto, numero_immagine)
        config: Configurazione dell'applicazione
        logger: Logger per monitoring
    """
    success_count = 0
    total_count = len(test_cases)
    
    for category, defect_type, img_number in test_cases:
        try:
            # Costruisci i percorsi
            img_path = os.path.join(config.paths.data_root, category, "test", defect_type, f"{img_number:03d}.png")
            
            # Validazione percorso immagine
            validated_path = validate_file_path(img_path, must_exist=True)
            
            # Directory output
            out_dir = os.path.join(config.paths.test_results_dir, category, defect_type)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{img_number:03d}_result.png")
            
            logger.info(f"🔍 Testing {category}/{defect_type}/{img_number:03d}.png")
            
            # Generazione overlay con error handling
            def generate_safe():
                return generate_overlay(str(validated_path), model, transform, save_path=out_path)
            
            result = safe_execute(
                func=generate_safe,
                logger=logger,
                context=f"Test {category}/{defect_type}/{img_number:03d}",
                default_return=False
            )
            
            if result is not False:
                logger.info(f"✅ Risultato salvato: {out_path}")
                success_count += 1
            else:
                logger.error(f"❌ Test fallito per {category}/{defect_type}/{img_number:03d}")
                
        except Exception as e:
            handle_exception(logger, e, f"Test case {category}/{defect_type}/{img_number:03d}")
            logger.error(f"❌ Errore nel test case: {category}/{defect_type}/{img_number:03d}")
    
    logger.info(f"📊 Test completato: {success_count}/{total_count} successi")


def load_model_safe(model_path: str, device: torch.device, config, logger: logging.Logger) -> torch.nn.Module:
    """
    Carica il modello in modo sicuro con validazioni.
    
    Args:
        model_path: Percorso del file modello
        device: Device PyTorch
        config: Configurazione dell'applicazione
        logger: Logger
        
    Returns:
        Modello caricato
        
    Raises:
        ModelError: Se il caricamento fallisce
    """
    try:
        # Validazione percorso
        model_file = validate_file_path(model_path, must_exist=True)
        
        # Verifica dimensione file
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        if file_size_mb < 1:  # Modello troppo piccolo, probabilmente corrotto
            raise ModelError(f"File modello troppo piccolo ({file_size_mb:.2f} MB) - potrebbe essere corrotto")
        
        logger.info(f"Caricamento modello ({file_size_mb:.1f} MB): {model_file}")
        
        # Inizializzazione modello
        model = AutoencoderUNetLite()
        
        # Caricamento state dict
        try:
            state_dict = torch.load(str(model_file), map_location=device)
            model.load_state_dict(state_dict)
            
        except RuntimeError as e:
            if "size mismatch" in str(e):
                raise ModelError(f"Architettura modello non compatibile: {e}")
            else:
                raise ModelError(f"Errore caricamento state dict: {e}")
        
        model.to(device)
        model.eval()
        
        # Test forward pass per verificare il modello
        try:
            with torch.no_grad():
                test_input = torch.randn(1, config.model.channels, *config.model.input_size).to(device)
                test_output = model(test_input)
                
                if test_output.shape != test_input.shape:
                    raise ModelError(f"Output shape mismatch: expected {test_input.shape}, got {test_output.shape}")
                    
        except Exception as e:
            raise ModelError(f"Test forward pass fallito: {e}")
        
        logger.info("✅ Modello caricato e validato con successo")
        return model
        
    except Exception as e:
        if isinstance(e, ModelError):
            raise
        handle_exception(logger, e, "Caricamento modello")
        raise ModelError(f"Impossibile caricare modello da {model_path}", {"original_error": str(e)})


def main():
    """Funzione principale di test con error handling completo."""
    logger = setup_logger("test_model", level="INFO")
    
    try:
        logger.info("🚀 Avvio test anomaly detection model")
        
        # Carica configurazione
        config = get_config()
        logger.info(f"Configurazione caricata per ambiente: {config.environment.value}")
        
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
        
        # Validazione configurazione
        if not TEST_CASES:
            raise DataError("Nessun test case definito")
        
        # Validazione paths
        if not os.path.exists(config.paths.data_root):
            raise DataError(f"Directory dati non trovata: {config.paths.data_root}")
        
        if not os.path.exists(config.paths.model_file):
            raise ModelError(f"File modello non trovato: {config.paths.model_file}")
        
        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device selezionato: {device}")
        
        # Caricamento modello
        logger.info("📥 Caricamento modello...")
        model = load_model_safe(config.paths.model_file, device, config, logger)
        
        # Trasformazioni
        try:
            transform = transforms.Compose([
                transforms.Resize(config.model.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*config.model.channels, 
                                   [0.5]*config.model.channels)
            ])
            logger.debug("Trasformazioni inizializzate")
        except Exception as e:
            raise ModelError(f"Errore inizializzazione trasformazioni: {e}")
        
        # Verifica che le directory di test esistano
        missing_dirs = []
        for category, defect_type, _ in TEST_CASES:
            test_dir = os.path.join(config.paths.data_root, category, "test", defect_type)
            if not os.path.exists(test_dir):
                missing_dirs.append(test_dir)
        
        if missing_dirs:
            logger.warning(f"Directory di test mancanti: {missing_dirs}")
        
        # Esegui i test
        logger.info("🔬 Avvio test...")
        test_images(model, transform, TEST_CASES, config, logger)
        logger.info("✅ Testing completato!")
        
    except Exception as e:
        handle_exception(logger, e, "Testing principale")
        logger.error("❌ Testing fallito")
        sys.exit(1)
    
    finally:
        # Cleanup risorse
        if 'device' in locals() and device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.info("🧹 Cleanup risorse completato")

if __name__ == "__main__":
    main() 