#!/usr/bin/env python3
"""
Script di test per il Configuration Management System.
Verifica che tutte le configurazioni siano caricate correttamente.
"""

import os
import sys
from pathlib import Path

# Aggiungi il path del progetto
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_manager import get_config_manager, get_config, Environment
from src.utils.logging_utils import setup_logger

def test_configuration_system():
    """Test completo del sistema di configurazione."""
    logger = setup_logger("config_test", level="INFO")
    
    try:
        logger.info("🧪 Avvio test Configuration Management System")
        
        # Test 1: Caricamento configurazione default
        logger.info("Test 1: Caricamento configurazione default")
        config = get_config()
        logger.info(f"✅ Configurazione caricata per ambiente: {config.environment.value}")
        
        # Test 2: Validazione configurazione
        logger.info("Test 2: Validazione configurazione")
        config.validate()
        logger.info("✅ Configurazione validata con successo")
        
        # Test 3: Test environment variables override
        logger.info("Test 3: Test environment variables")
        
        # Simula environment variable
        original_batch_size = config.training.batch_size
        os.environ["ANOMALY_SPOTTER_BATCH_SIZE"] = "64"
        
        # Ricarica configurazione
        config_manager = get_config_manager()
        config_with_env = config_manager.load_config(force_reload=True)
        
        if config_with_env.training.batch_size == 64:
            logger.info("✅ Environment variable override funziona")
        else:
            logger.warning(f"❌ Environment variable non applicata: {config_with_env.training.batch_size}")
        
        # Cleanup
        del os.environ["ANOMALY_SPOTTER_BATCH_SIZE"]
        
        # Test 4: Test diversi ambienti
        logger.info("Test 4: Test configurazioni per diversi ambienti")
        
        for env in Environment:
            try:
                env_config = config_manager.load_config(environment=env, force_reload=True)
                env_config.validate()
                logger.info(f"✅ Configurazione {env.value}: OK")
            except Exception as e:
                logger.error(f"❌ Configurazione {env.value}: {e}")
        
        # Test 5: Test creazione directory
        logger.info("Test 5: Test creazione directory")
        config.paths.create_directories()
        
        directories_to_check = [
            config.paths.output_root,
            config.paths.logs_root,
            config.paths.test_results_dir
        ]
        
        for dir_path in directories_to_check:
            if os.path.exists(dir_path):
                logger.info(f"✅ Directory creata: {dir_path}")
            else:
                logger.warning(f"❌ Directory non creata: {dir_path}")
        
        # Test 6: Test serializzazione configurazione
        logger.info("Test 6: Test serializzazione configurazione")
        
        config_dict = config.to_dict()
        if isinstance(config_dict, dict) and len(config_dict) > 0:
            logger.info("✅ Serializzazione configurazione: OK")
        else:
            logger.warning("❌ Serializzazione configurazione: FAILED")
        
        # Test 7: Test salvataggio configurazione
        logger.info("Test 7: Test salvataggio configurazione")
        
        try:
            save_path = config_manager.save_config(config, format="yaml")
            if save_path.exists():
                logger.info(f"✅ Configurazione salvata: {save_path}")
            else:
                logger.warning(f"❌ Configurazione non salvata: {save_path}")
        except Exception as e:
            logger.error(f"❌ Errore salvataggio configurazione: {e}")
        
        logger.info("🎉 Tutti i test Configuration Management completati!")
        
        # Stampa summary configurazione
        logger.info("\n📊 Summary Configurazione Corrente:")
        logger.info(f"  Ambiente: {config.environment.value}")
        logger.info(f"  Debug: {config.debug}")
        logger.info(f"  Batch Size: {config.training.batch_size}")
        logger.info(f"  Learning Rate: {config.training.learning_rate}")
        logger.info(f"  Input Size: {config.model.input_size}")
        logger.info(f"  Data Root: {config.paths.data_root}")
        logger.info(f"  Output Root: {config.paths.output_root}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test Configuration Management fallito: {e}")
        return False


if __name__ == "__main__":
    success = test_configuration_system()
    sys.exit(0 if success else 1)
