"""
Modulo per logging strutturato e error handling robusto.
Centralizza la configurazione di logging per tutto il progetto.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import traceback


class ColoredFormatter(logging.Formatter):
    """Formatter colorato per output console."""
    
    # Codici ANSI per colori
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Verde
        'WARNING': '\033[33m',   # Giallo
        'ERROR': '\033[31m',     # Rosso
        'CRITICAL': '\033[35m',  # Magenta
        'ENDC': '\033[0m'        # Reset
    }

    def format(self, record):
        # Aggiungi colore al livello
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['ENDC']}"
        
        return super().format(record)


def setup_logger(
    name: str = "anomaly-spotter",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configura un logger strutturato con output su file e console.
    
    Args:
        name: Nome del logger
        level: Livello di logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory per i file di log (default: logs/)
        console_output: Se True, mostra log anche su console
        
    Returns:
        Logger configurato
    """
    
    # Crea logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Evita duplicazione handlers se già configurato
    if logger.handlers:
        return logger
    
    # Formato dettagliato per file
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formato semplificato per console
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Handler per file
    if log_dir is None:
        # Default: logs/ nella root del progetto
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # File di log con timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # File sempre DEBUG
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Handler per console
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Log iniziale
    logger.info(f"Logger '{name}' inizializzato - Level: {level} - Log file: {log_file}")
    
    return logger


class AnomalySpotterError(Exception):
    """Eccezione base per il progetto Anomaly Spotter."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ModelError(AnomalySpotterError):
    """Errori relativi al modello (caricamento, inferenza, training)."""
    pass


class DataError(AnomalySpotterError):
    """Errori relativi ai dati (caricamento, preprocessing, validazione)."""
    pass


class ConfigError(AnomalySpotterError):
    """Errori di configurazione."""
    pass


class ResourceError(AnomalySpotterError):
    """Errori di risorse (GPU, memoria, spazio disco)."""
    pass


def handle_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """
    Gestisce un'eccezione in modo strutturato con logging dettagliato.
    
    Args:
        logger: Logger da usare
        exc: Eccezione da gestire
        context: Contesto aggiuntivo per il debug
    """
    error_details = {
        'exception_type': type(exc).__name__,
        'exception_message': str(exc),
        'context': context,
        'traceback': traceback.format_exc()
    }
    
    # Log con dettagli differenti per tipo di errore
    if isinstance(exc, AnomalySpotterError):
        logger.error(f"Application Error in {context}: {exc.message}")
        if exc.details:
            logger.debug(f"Error details: {exc.details}")
    else:
        logger.error(f"Unexpected error in {context}: {str(exc)}")
    
    # Traceback completo solo in DEBUG
    logger.debug(f"Full traceback:\n{error_details['traceback']}")


def safe_execute(func, logger: logging.Logger, context: str = "", default_return=None):
    """
    Esegue una funzione in modo sicuro con error handling.
    
    Args:
        func: Funzione da eseguire
        logger: Logger da usare
        context: Contesto per logging
        default_return: Valore di default in caso di errore
        
    Returns:
        Risultato della funzione o default_return se errore
    """
    try:
        return func()
    except Exception as e:
        handle_exception(logger, e, context)
        return default_return


def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """
    Valida e converte un percorso file con controlli di sicurezza.
    
    Args:
        file_path: Percorso da validare
        must_exist: Se True, il file deve esistere
        
    Returns:
        Path object validato
        
    Raises:
        DataError: Se il percorso non è valido
    """
    try:
        path = Path(file_path).resolve()
        
        # Controllo esistenza
        if must_exist and not path.exists():
            raise DataError(f"File non trovato: {file_path}")
        
        # Controllo path traversal (sicurezza)
        if ".." in str(path) or str(path).startswith("/"):
            # Permetti solo percorsi assoluti sicuri o relativi alla working dir
            pass
        
        return path
        
    except Exception as e:
        raise DataError(f"Percorso file non valido: {file_path}", {"original_error": str(e)})


def validate_image_file(file_path: str) -> Path:
    """
    Valida che un file sia un'immagine supportata.
    
    Args:
        file_path: Percorso dell'immagine
        
    Returns:
        Path object validato
        
    Raises:
        DataError: Se non è un'immagine valida
    """
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    path = validate_file_path(file_path, must_exist=True)
    
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DataError(
            f"Formato immagine non supportato: {path.suffix}",
            {"supported_formats": list(SUPPORTED_EXTENSIONS)}
        )
    
    return path


def check_system_resources(logger: logging.Logger) -> dict:
    """
    Controlla le risorse di sistema disponibili.
    
    Returns:
        Dict con informazioni sulle risorse
    """
    import torch
    import psutil
    
    resources = {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if resources['cuda_available']:
        resources['cuda_memory_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    
    logger.info(f"System resources: {resources}")
    return resources
