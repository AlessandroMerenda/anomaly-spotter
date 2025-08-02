"""
Sistema di Configuration Management avanzato per Anomaly Spotter.
Supporta environment variables, config files, profiles e hot-reload.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime

# Import utilities
from .logging_utils import setup_logger, ConfigError, handle_exception


class Environment(Enum):
    """Ambienti supportati."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class ModelConfig:
    """Configurazione del modello."""
    input_size: tuple = (128, 128)
    channels: int = 3
    latent_dim: int = 512
    architecture: str = "AutoencoderUNetLite"
    
    def validate(self):
        """Valida la configurazione del modello."""
        if not isinstance(self.input_size, (tuple, list)) or len(self.input_size) != 2:
            raise ConfigError("input_size deve essere una tupla (width, height)")
        
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ConfigError("input_size deve avere dimensioni positive")
        
        if self.channels not in [1, 3]:
            raise ConfigError("channels deve essere 1 (grayscale) o 3 (RGB)")
        
        if self.latent_dim <= 0:
            raise ConfigError("latent_dim deve essere positivo")


@dataclass
class TrainingConfig:
    """Configurazione del training."""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    validation_split: float = 0.2
    num_workers: int = 4
    early_stopping_patience: int = 10
    gradient_clip_value: float = 1.0
    save_every_n_epochs: int = 10
    
    def validate(self):
        """Valida la configurazione del training."""
        if self.batch_size <= 0:
            raise ConfigError("batch_size deve essere positivo")
        
        if self.num_epochs <= 0:
            raise ConfigError("num_epochs deve essere positivo")
        
        if not 0 < self.learning_rate < 1:
            raise ConfigError("learning_rate deve essere tra 0 e 1")
        
        if not 0 < self.validation_split < 1:
            raise ConfigError("validation_split deve essere tra 0 e 1")
        
        if self.num_workers < 0:
            raise ConfigError("num_workers deve essere >= 0")


@dataclass
class AnomalyConfig:
    """Configurazione per anomaly detection."""
    threshold_pixel: float = 0.015
    gaussian_sigma: float = 1.0
    min_recall: float = 0.95
    use_otsu_threshold: bool = True
    morphological_operations: bool = False
    min_anomaly_area: int = 10
    
    def validate(self):
        """Valida la configurazione anomaly detection."""
        if not 0 <= self.threshold_pixel <= 1:
            raise ConfigError("threshold_pixel deve essere tra 0 e 1")
        
        if self.gaussian_sigma <= 0:
            raise ConfigError("gaussian_sigma deve essere positivo")
        
        if not 0 <= self.min_recall <= 1:
            raise ConfigError("min_recall deve essere tra 0 e 1")


@dataclass
class VisualizationConfig:
    """Configurazione per visualizzazione."""
    figure_size: tuple = (15, 5)
    dpi: int = 150
    colormap: str = 'inferno'
    font_size: int = 10
    save_format: str = 'png'
    
    def validate(self):
        """Valida la configurazione visualizzazione."""
        if not isinstance(self.figure_size, (tuple, list)) or len(self.figure_size) != 2:
            raise ConfigError("figure_size deve essere una tupla (width, height)")
        
        if self.dpi <= 0:
            raise ConfigError("dpi deve essere positivo")
        
        if self.font_size <= 0:
            raise ConfigError("font_size deve essere positivo")


@dataclass
class PathsConfig:
    """Configurazione dei percorsi."""
    # Directory base
    project_root: str = ""
    data_root: str = ""
    output_root: str = ""
    logs_root: str = ""
    config_root: str = ""
    
    # File specifici
    model_file: str = ""
    thresholds_csv: str = ""
    thresholds_json: str = ""
    training_plot: str = ""
    
    # Directory output
    test_results_dir: str = ""
    all_results_dir: str = ""
    overlay_dir: str = ""
    stats_dir: str = ""
    
    def __post_init__(self):
        """Inizializza i percorsi se non specificati."""
        if not self.project_root:
            # Auto-detect project root
            current_file = Path(__file__).resolve()
            self.project_root = str(current_file.parent.parent.parent)
        
        # Costruisci percorsi derivati
        self._build_derived_paths()
    
    def _build_derived_paths(self):
        """Costruisce i percorsi derivati dalla configurazione base."""
        root = Path(self.project_root)
        
        # Directory base
        if not self.data_root:
            self.data_root = str(root / "data" / "mvtec_ad")
        if not self.output_root:
            self.output_root = str(root / "outputs")
        if not self.logs_root:
            self.logs_root = str(root / "logs")
        if not self.config_root:
            self.config_root = str(root / "config")
        
        # File specifici
        output_path = Path(self.output_root)
        if not self.model_file:
            self.model_file = str(output_path / "model.pth")
        if not self.thresholds_csv:
            self.thresholds_csv = str(output_path / "thresholds.csv")
        if not self.thresholds_json:
            self.thresholds_json = str(output_path / "thresholds.json")
        if not self.training_plot:
            self.training_plot = str(output_path / "training_history.png")
        
        # Directory output
        if not self.test_results_dir:
            self.test_results_dir = str(output_path / "test_results")
        if not self.all_results_dir:
            self.all_results_dir = str(output_path / "all_results")
        if not self.overlay_dir:
            self.overlay_dir = str(output_path / "overlay")
        if not self.stats_dir:
            self.stats_dir = str(output_path / "stats")
    
    def validate(self):
        """Valida la configurazione dei percorsi."""
        # Verifica che il project root esista
        if not os.path.exists(self.project_root):
            raise ConfigError(f"Project root non trovato: {self.project_root}")
        
        # Verifica che data_root esista (se specificato)
        if self.data_root and not os.path.exists(self.data_root):
            raise ConfigError(f"Data root non trovato: {self.data_root}")
    
    def create_directories(self):
        """Crea le directory necessarie se non esistono."""
        dirs_to_create = [
            self.output_root,
            self.logs_root,
            self.config_root,
            self.test_results_dir,
            self.all_results_dir,
            self.overlay_dir,
            self.stats_dir
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configurazione del logging."""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    max_file_size_mb: int = 100
    backup_count: int = 5
    format_console: str = "%(asctime)s | %(levelname)s | %(message)s"
    format_file: str = "%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s"
    
    def validate(self):
        """Valida la configurazione logging."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ConfigError(f"level deve essere uno di: {valid_levels}")


@dataclass
class AppConfig:
    """Configurazione completa dell'applicazione."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # Configurazioni componenti
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Metadata
    config_version: str = "1.0"
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def validate(self):
        """Valida tutta la configurazione."""
        self.model.validate()
        self.training.validate()
        self.anomaly.validate()
        self.visualization.validate()
        self.paths.validate()
        self.logging.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte la configurazione in dizionario."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Crea configurazione da dizionario."""
        # Converti environment se è stringa
        if 'environment' in data and isinstance(data['environment'], str):
            data['environment'] = Environment(data['environment'])
        
        # Crea istanze dei sotto-config
        config = cls()
        
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'anomaly' in data:
            config.anomaly = AnomalyConfig(**data['anomaly'])
        if 'visualization' in data:
            config.visualization = VisualizationConfig(**data['visualization'])
        if 'paths' in data:
            config.paths = PathsConfig(**data['paths'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        # Altri campi
        for key, value in data.items():
            if hasattr(config, key) and key not in ['model', 'training', 'anomaly', 'visualization', 'paths', 'logging']:
                setattr(config, key, value)
        
        return config


class ConfigManager:
    """
    Manager centralizzato per la gestione delle configurazioni.
    Supporta environment variables, file YAML/JSON, profiles e hot-reload.
    """
    
    def __init__(self, config_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger("ConfigManager")
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._config: Optional[AppConfig] = None
        self._config_file_path: Optional[Path] = None
        self._last_modified: Optional[float] = None
        
        # Environment variable prefix
        self.env_prefix = "ANOMALY_SPOTTER_"
        
        # Categorie MVTec supportate (può essere configurabile)
        self.default_categories = {
            "screw": ["good", "manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
            "capsule": ["good", "crack", "faulty_imprint", "poke", "scratch", "squeeze"],
            "hazelnut": ["good", "crack", "cut", "hole", "print"]
        }
    
    def load_config(self, 
                   config_file: Optional[str] = None,
                   environment: Optional[Union[str, Environment]] = None,
                   force_reload: bool = False) -> AppConfig:
        """
        Carica la configurazione da file, environment variables e default.
        
        Args:
            config_file: Percorso del file di configurazione (opzionale)
            environment: Ambiente di esecuzione (dev/staging/prod)
            force_reload: Forza il reload anche se già caricata
            
        Returns:
            Configurazione caricata e validata
        """
        try:
            # Determina environment
            env = self._determine_environment(environment)
            self.logger.info(f"Caricamento configurazione per ambiente: {env.value}")
            
            # Carica configurazione base
            config = self._load_base_config(config_file, env)
            
            # Applica environment variables
            config = self._apply_environment_variables(config)
            
            # Validazione finale
            config.validate()
            
            # Crea directory necessarie
            config.paths.create_directories()
            
            # Cache configurazione
            self._config = config
            if config_file:
                self._config_file_path = Path(config_file)
                self._last_modified = self._config_file_path.stat().st_mtime
            
            self.logger.info("Configurazione caricata e validata con successo")
            return config
            
        except Exception as e:
            handle_exception(self.logger, e, "Caricamento configurazione")
            raise ConfigError(f"Impossibile caricare configurazione: {e}")
    
    def _determine_environment(self, environment: Optional[Union[str, Environment]]) -> Environment:
        """Determina l'ambiente di esecuzione."""
        if environment:
            if isinstance(environment, str):
                try:
                    return Environment(environment.lower())
                except ValueError:
                    raise ConfigError(f"Ambiente non valido: {environment}")
            return environment
        
        # Controlla environment variable
        env_var = os.getenv(f"{self.env_prefix}ENVIRONMENT", "development")
        try:
            return Environment(env_var.lower())
        except ValueError:
            self.logger.warning(f"Ambiente da env var non valido: {env_var}, usando development")
            return Environment.DEVELOPMENT
    
    def _load_base_config(self, config_file: Optional[str], env: Environment) -> AppConfig:
        """Carica la configurazione base da file o default."""
        
        # Determina il file di configurazione
        if config_file:
            config_path = Path(config_file)
        else:
            # Cerca file specifico per ambiente
            config_path = self.config_dir / f"{env.value}.yaml"
            if not config_path.exists():
                config_path = self.config_dir / f"{env.value}.yml"
            if not config_path.exists():
                config_path = self.config_dir / f"{env.value}.json"
            if not config_path.exists():
                # File generico
                config_path = self.config_dir / "config.yaml"
                if not config_path.exists():
                    config_path = self.config_dir / "config.yml"
                if not config_path.exists():
                    config_path = self.config_dir / "config.json"
        
        # Carica da file se esiste
        if config_path and config_path.exists():
            self.logger.info(f"Caricamento configurazione da: {config_path}")
            return self._load_config_from_file(config_path, env)
        else:
            self.logger.info("File configurazione non trovato, usando configurazione default")
            return self._create_default_config(env)
    
    def _load_config_from_file(self, config_path: Path, env: Environment) -> AppConfig:
        """Carica configurazione da file YAML o JSON."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ConfigError(f"Formato file non supportato: {config_path.suffix}")
            
            if not data:
                raise ConfigError("File configurazione vuoto")
            
            # Imposta environment
            data['environment'] = env
            
            return AppConfig.from_dict(data)
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigError(f"Errore parsing file configurazione: {e}")
        except Exception as e:
            raise ConfigError(f"Errore lettura file configurazione: {e}")
    
    def _create_default_config(self, env: Environment) -> AppConfig:
        """Crea configurazione default per l'ambiente specificato."""
        config = AppConfig(environment=env)
        
        # Personalizzazioni per ambiente
        if env == Environment.DEVELOPMENT:
            config.debug = True
            config.logging.level = "DEBUG"
            config.training.num_epochs = 10  # Epochs ridotte per dev
        elif env == Environment.STAGING:
            config.debug = False
            config.logging.level = "INFO"
            config.training.num_epochs = 50
        elif env == Environment.PRODUCTION:
            config.debug = False
            config.logging.level = "WARNING"
            config.training.num_epochs = 100
            config.logging.console_output = False  # Solo file in prod
        elif env == Environment.TESTING:
            config.debug = False
            config.logging.level = "ERROR"
            config.training.num_epochs = 1  # Testing veloce
            config.training.batch_size = 4
        
        return config
    
    def _apply_environment_variables(self, config: AppConfig) -> AppConfig:
        """Applica environment variables alla configurazione."""
        
        # Mappatura environment variables a configurazione
        env_mappings = {
            f"{self.env_prefix}DEBUG": ("debug", self._parse_bool),
            f"{self.env_prefix}LOG_LEVEL": ("logging.level", str),
            f"{self.env_prefix}BATCH_SIZE": ("training.batch_size", int),
            f"{self.env_prefix}LEARNING_RATE": ("training.learning_rate", float),
            f"{self.env_prefix}NUM_EPOCHS": ("training.num_epochs", int),
            f"{self.env_prefix}DATA_ROOT": ("paths.data_root", str),
            f"{self.env_prefix}OUTPUT_ROOT": ("paths.output_root", str),
            f"{self.env_prefix}MODEL_FILE": ("paths.model_file", str),
            f"{self.env_prefix}INPUT_SIZE": ("model.input_size", self._parse_tuple),
            f"{self.env_prefix}CHANNELS": ("model.channels", int),
            f"{self.env_prefix}THRESHOLD_PIXEL": ("anomaly.threshold_pixel", float),
            f"{self.env_prefix}GAUSSIAN_SIGMA": ("anomaly.gaussian_sigma", float),
        }
        
        for env_var, (config_path, parser) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    parsed_value = parser(value)
                    self._set_nested_config(config, config_path, parsed_value)
                    self.logger.debug(f"Applicata env var {env_var} = {parsed_value}")
                except Exception as e:
                    self.logger.warning(f"Errore parsing env var {env_var}={value}: {e}")
        
        # Ricostruisci percorsi derivati se paths è stato modificato
        config.paths._build_derived_paths()
        
        return config
    
    def _parse_bool(self, value: str) -> bool:
        """Parse booleano da stringa."""
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def _parse_tuple(self, value: str) -> tuple:
        """Parse tupla da stringa (es. "128,128")."""
        try:
            parts = [int(x.strip()) for x in value.split(',')]
            return tuple(parts)
        except:
            raise ValueError(f"Formato tupla non valido: {value}")
    
    def _set_nested_config(self, config: AppConfig, path: str, value: Any):
        """Imposta valore in configurazione annidata (es. "training.batch_size")."""
        parts = path.split('.')
        obj = config
        
        # Naviga fino al penultimo oggetto
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Imposta il valore finale
        setattr(obj, parts[-1], value)
    
    def save_config(self, config: AppConfig, file_path: Optional[str] = None, format: str = "yaml") -> Path:
        """
        Salva la configurazione su file.
        
        Args:
            config: Configurazione da salvare
            file_path: Percorso file (opzionale)
            format: Formato (yaml/json)
            
        Returns:
            Path del file salvato
        """
        try:
            if file_path:
                save_path = Path(file_path)
            else:
                env = config.environment.value
                extension = "yaml" if format == "yaml" else "json"
                save_path = self.config_dir / f"{env}.{extension}"
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Aggiorna timestamp
            config.last_updated = datetime.now().isoformat()
            
            data = config.to_dict()
            # Converti Environment enum in stringa
            data['environment'] = config.environment.value
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if format == "yaml":
                    yaml.dump(data, f, default_flow_style=False, indent=2, allow_unicode=True)
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configurazione salvata: {save_path}")
            return save_path
            
        except Exception as e:
            handle_exception(self.logger, e, f"Salvataggio configurazione {file_path}")
            raise ConfigError(f"Impossibile salvare configurazione: {e}")
    
    def check_hot_reload(self) -> bool:
        """
        Controlla se la configurazione è cambiata su disco e ricarica se necessario.
        
        Returns:
            True se ricaricata, False altrimenti
        """
        if not self._config_file_path or not self._last_modified:
            return False
        
        try:
            current_mtime = self._config_file_path.stat().st_mtime
            if current_mtime > self._last_modified:
                self.logger.info("Rilevata modifica configurazione, ricaricamento...")
                self.load_config(str(self._config_file_path), force_reload=True)
                return True
        except Exception as e:
            self.logger.warning(f"Errore controllo hot-reload: {e}")
        
        return False
    
    def get_config(self, reload_if_changed: bool = True) -> AppConfig:
        """
        Ottieni la configurazione corrente.
        
        Args:
            reload_if_changed: Se True, controlla e ricarica se cambiata
            
        Returns:
            Configurazione corrente
        """
        if self._config is None:
            return self.load_config()
        
        if reload_if_changed:
            self.check_hot_reload()
        
        return self._config
    
    def create_sample_configs(self):
        """Crea file di configurazione di esempio per tutti gli ambienti."""
        for env in Environment:
            config = self._create_default_config(env)
            self.save_config(config, format="yaml")
            self.logger.info(f"Creato config di esempio per {env.value}")


# Istanza globale del config manager
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """Ottieni l'istanza singleton del ConfigManager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager

def get_config(reload_if_changed: bool = True) -> AppConfig:
    """Shortcut per ottenere la configurazione corrente."""
    return get_config_manager().get_config(reload_if_changed)
