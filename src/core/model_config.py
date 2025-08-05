"""
Advanced Model Configuration for AutoencoderUNetLite.
Combines dataclass-based configuration with advanced features.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch
import logging
from pathlib import Path


@dataclass
class AutoencoderConfig:
    """
    Comprehensive configuration for AutoencoderUNetLite.
    
    Features:
    - Dataclass-based structure for easy serialization
    - Category-specific configurations
    - Advanced training parameters
    - Loss function configuration
    - Anomaly detection settings
    """
    
    # ========================
    # Model Architecture
    # ========================
    input_channels: int = 3
    output_channels: int = 3
    base_channels: int = 64  # Match notebook default
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    
    # Bottleneck configuration
    bottleneck_channels: int = 512
    use_attention: bool = False  # Future: attention mechanisms
    dropout_rate: float = 0.0
    
    # ========================
    # Input/Output Dimensions
    # ========================
    input_size: Tuple[int, int] = (128, 128)  # Match notebook default
    
    # ========================
    # Training Configuration
    # ========================
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "T_max": 100,  # For cosine annealing
        "eta_min": 1e-6
    })
    
    # ========================
    # Loss Function Configuration
    # ========================
    loss_type: str = "combined"  # "mse", "l1", "combined", "perceptual"
    
    # Loss weights
    reconstruction_weight: float = 1.0
    perceptual_weight: float = 0.1
    ssim_weight: float = 0.5
    edge_weight: float = 0.0  # Edge preservation loss
    
    # MSE/L1 combination (if loss_type == "combined")
    mse_weight: float = 0.7
    l1_weight: float = 0.3
    
    # ========================
    # Optimization
    # ========================
    optimizer_type: str = "adam"  # "adam", "adamw", "sgd"
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "betas": (0.9, 0.999),
        "eps": 1e-8
    })
    
    # Mixed precision training
    use_amp: bool = True
    gradient_clip: float = 1.0
    
    # ========================
    # Training Monitoring
    # ========================
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    monitor_metric: str = "val_loss"  # "val_loss", "reconstruction_error"
    
    # Checkpointing
    save_best_model: bool = True
    save_checkpoint_every: int = 10  # epochs
    
    # Validation
    val_split: float = 0.2
    val_check_interval: int = 1  # epochs
    
    # ========================
    # Anomaly Detection
    # ========================
    threshold_method: str = "percentile"  # "percentile", "statistics", "otsu"
    threshold_percentile: float = 95.0
    threshold_std_factor: float = 3.0  # For statistics method
    
    # Post-processing
    apply_gaussian_blur: bool = True
    blur_kernel_size: int = 5
    apply_morphology: bool = False
    
    # ========================
    # Data Configuration
    # ========================
    normalize_type: str = "tanh"  # "imagenet", "tanh", "none"
    augmentation_strength: float = 0.5  # 0.0 = no aug, 1.0 = full aug
    
    # Multi-category training
    categories: Optional[List[str]] = None
    category_weights: Optional[Dict[str, float]] = None
    
    # ========================
    # Device and Performance
    # ========================
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # ========================
    # Logging and Monitoring
    # ========================
    log_level: str = "INFO"
    log_every_n_steps: int = 50
    plot_samples: bool = True
    num_sample_plots: int = 8
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "anomaly-spotter"
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Auto-detect device if needed
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set default categories if not specified
        if self.categories is None:
            self.categories = ["screw", "capsule", "hazelnut"]
        
        # Set experiment name if not specified
        if self.experiment_name is None:
            self.experiment_name = f"autoencoder_{'-'.join(self.categories)}"
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.input_channels > 0, "input_channels must be positive"
        assert self.base_channels > 0, "base_channels must be positive"
        assert len(self.encoder_depths) == len(self.decoder_depths), \
            "encoder_depths and decoder_depths must have same length"
        assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert 0 <= self.dropout_rate < 1, "dropout_rate must be in [0, 1)"
        assert self.device in ["auto", "cuda", "cpu"], "Invalid device"
    
    @classmethod
    def from_category(cls, category: str) -> 'AutoencoderConfig':
        """
        Create category-specific configurations.
        
        Args:
            category: Category name ("capsule", "hazelnut", "screw", "multi")
            
        Returns:
            Configured AutoencoderConfig instance
        """
        # Base configuration
        base_config = {
            "categories": [category] if category != "multi" else ["screw", "capsule", "hazelnut"],
            "experiment_name": f"autoencoder_{category}"
        }
        
        # Category-specific optimizations
        category_configs = {
            'capsule': {
                **base_config,
                'base_channels': 64,
                'bottleneck_channels': 256,
                'learning_rate': 1e-4,
                'batch_size': 32,
                'threshold_percentile': 95.0,
                'augmentation_strength': 0.3,
                'input_size': (128, 128)
            },
            'hazelnut': {
                **base_config,
                'base_channels': 64,
                'bottleneck_channels': 512,
                'learning_rate': 8e-5,
                'batch_size': 32,
                'threshold_percentile': 96.0,
                'augmentation_strength': 0.4,
                'input_size': (128, 128),
                'apply_gaussian_blur': True
            },
            'screw': {
                **base_config,
                'base_channels': 64,
                'bottleneck_channels': 256,
                'learning_rate': 1e-4,
                'batch_size': 32,
                'threshold_percentile': 94.0,
                'augmentation_strength': 0.2,
                'input_size': (128, 128),
                # Screws benefit from edge preservation
                'edge_weight': 0.1
            },
            'multi': {
                **base_config,
                'base_channels': 64,
                'bottleneck_channels': 512,
                'learning_rate': 1e-4,
                'batch_size': 32,
                'threshold_percentile': 95.0,
                'augmentation_strength': 0.3,
                'input_size': (128, 128),
                'num_epochs': 150,  # More epochs for multi-category
                'patience': 20,
                # Balanced loss for multi-category
                'perceptual_weight': 0.15,
                'ssim_weight': 0.3
            }
        }
        
        # Get category-specific config or default
        config_dict = category_configs.get(category, category_configs['multi'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AutoencoderConfig':
        """Load configuration from file (JSON/YAML)."""
        import json
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**config_dict)
    
    def save(self, config_path: str):
        """Save configuration to file."""
        import json
        from dataclasses import asdict
        
        config_path = Path(config_path)
        config_dict = asdict(self)
        
        # Convert tuples to lists for JSON serialization
        config_dict['input_size'] = list(config_dict['input_size'])
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters for model initialization."""
        return {
            'in_channels': self.input_channels,
            'out_channels': self.output_channels,
            'base_channels': self.base_channels,
            'bottleneck_channels': self.bottleneck_channels,
            'encoder_depths': self.encoder_depths,
            'decoder_depths': self.decoder_depths,
            'dropout_rate': self.dropout_rate,
            'use_attention': self.use_attention
        }
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """Get parameters for optimizer initialization."""
        base_params = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        base_params.update(self.optimizer_params)
        return base_params
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """Get parameters for learning rate scheduler."""
        return self.scheduler_params.copy()
    
    def summary(self) -> str:
        """Get a summary of the configuration."""
        summary_lines = [
            "AutoencoderConfig Summary:",
            "=" * 50,
            f"Categories: {self.categories}",
            f"Input size: {self.input_size}",
            f"Architecture: {self.base_channels} -> {self.bottleneck_channels}",
            f"Learning rate: {self.learning_rate}",
            f"Batch size: {self.batch_size}",
            f"Epochs: {self.num_epochs}",
            f"Loss type: {self.loss_type}",
            f"Device: {self.device}",
            f"Use AMP: {self.use_amp}",
            f"Threshold method: {self.threshold_method} ({self.threshold_percentile}%)",
            "=" * 50
        ]
        return "\n".join(summary_lines)


# Convenience functions for common configurations
def get_default_config() -> AutoencoderConfig:
    """Get default multi-category configuration."""
    return AutoencoderConfig.from_category("multi")

def get_production_config() -> AutoencoderConfig:
    """Get optimized production configuration."""
    config = AutoencoderConfig.from_category("multi")
    
    # Production optimizations
    config.use_amp = True
    config.num_workers = 8
    config.pin_memory = True
    config.save_best_model = True
    config.early_stopping = True
    config.log_level = "WARNING"  # Less verbose in production
    
    return config

def get_debug_config() -> AutoencoderConfig:
    """Get configuration for debugging/development."""
    config = AutoencoderConfig.from_category("multi")
    
    # Debug optimizations
    config.batch_size = 4
    config.num_epochs = 5
    config.log_every_n_steps = 1
    config.val_check_interval = 1
    config.plot_samples = True
    config.log_level = "DEBUG"
    
    return config
