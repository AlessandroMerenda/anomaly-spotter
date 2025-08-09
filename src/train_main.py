#!/usr/bin/env python3
"""
Main training script for anomaly detection system.
Orchestrates the complete training pipeline with all advanced components.
"""

import argparse
import yaml
import json
from pathlib import Path
import torch
from datetime import datetime
import sys
import logging
from typing import Optional, Dict, Any

# Import core components
from src.core.model import AutoencoderUNetLite
from src.core.model_config import AutoencoderConfig
from src.core.losses import create_loss_function
from src.training.trainer import AnomalyDetectorTrainer, train_anomaly_detector
from src.data.loaders import MVTecDataset, create_dataloaders
from src.data.preprocessing import MVTecPreprocessor
from src.utils.logging_utils import setup_logger


def setup_environment() -> None:
    """Setup training environment and check dependencies."""
    # Set torch settings for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set multiprocessing start method
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


def load_config(args: argparse.Namespace) -> AutoencoderConfig:
    """
    Load configuration from category defaults and optional custom config.
    
    Args:
        args: Command line arguments
    
    Returns:
        Configured AutoencoderConfig instance
    """
    logger = logging.getLogger(__name__)
    
    # Start with category-specific config
    try:
        config = AutoencoderConfig.from_category(args.category)
        logger.info(f"Loaded base config for category: {args.category}")
    except Exception as e:
        logger.warning(f"Failed to load category config: {e}")
        # Fallback to default config
        config = AutoencoderConfig()
        logger.info("Using default configuration")
    
    # Override with custom config if provided
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r') as f:
                custom_config = yaml.safe_load(f)
            
            # Apply custom settings
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.info(f"Config override: {key} = {value}")
                else:
                    logger.warning(f"Unknown config parameter: {key}")
            
            logger.info(f"Applied custom config from: {args.config}")
        except Exception as e:
            logger.error(f"Failed to load custom config: {e}")
            logger.info("Continuing with base configuration")
    
    # Apply command line overrides
    if hasattr(args, 'batch_size') and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, 'epochs') and args.epochs:
        config.num_epochs = args.epochs
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.learning_rate = args.learning_rate
    
    return config


def create_output_directory(args: argparse.Namespace, config: AutoencoderConfig) -> Path:
    """
    Create timestamped output directory for training results.
    
    Args:
        args: Command line arguments
        config: Training configuration
    
    Returns:
        Path to output directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create descriptive directory name
    if hasattr(args, 'experiment_name') and args.experiment_name:
        dir_name = f"{args.category}_{args.experiment_name}_{timestamp}"
    else:
        dir_name = f"{args.category}_{timestamp}"
    
    output_dir = Path(args.output_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_training_metadata(output_dir: Path, config: AutoencoderConfig, 
                          args: argparse.Namespace) -> None:
    """
    Save training metadata and configuration.
    
    Args:
        output_dir: Output directory
        config: Training configuration
        args: Command line arguments
    """
    # Save configuration as YAML
    config_dict = config.__dict__.copy()
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    # Save configuration as JSON (more portable)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Save command line arguments
    args_dict = vars(args)
    with open(output_dir / 'arguments.json', 'w') as f:
        json.dump(args_dict, f, indent=2, default=str)
    
    # Save system information
    system_info = {
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    if torch.cuda.is_available():
        system_info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                   for i in range(torch.cuda.device_count())]
    
    with open(output_dir / 'system_info.json', 'w') as f:
        json.dump(system_info, f, indent=2)


def validate_dataset_structure(data_dir: Path, category: str, logger: logging.Logger) -> bool:
    """
    Validate MVTec dataset structure.
    
    Args:
        data_dir: Path to dataset root
        category: Category name
        logger: Logger instance
    
    Returns:
        True if structure is valid
    """
    category_path = data_dir / category
    
    required_paths = [
        category_path / 'train' / 'good',
        category_path / 'test'
    ]
    
    for path in required_paths:
        if not path.exists():
            logger.error(f"Required dataset path missing: {path}")
            return False
    
    # Check if we have training images
    train_images = list((category_path / 'train' / 'good').glob('*.png'))
    if not train_images:
        logger.error(f"No training images found in {category_path / 'train' / 'good'}")
        return False
    
    logger.info(f"Dataset validation passed for {category}")
    logger.info(f"Found {len(train_images)} training images")
    
    return True


def main(args: argparse.Namespace) -> None:
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Setup environment
    setup_environment()
    
    # Setup logging
    logger = setup_logger("MainTraining", level=logging.INFO)
    logger.info("=" * 80)
    logger.info("ANOMALY DETECTION TRAINING STARTED")
    logger.info("=" * 80)
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        logger.info("Using CPU (CUDA not available)")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration for category: {args.category}")
        config = load_config(args)
        
        # Create output directory
        output_dir = create_output_directory(args, config)
        logger.info(f"Output directory: {output_dir}")
        
        # Save metadata
        save_training_metadata(output_dir, config, args)
        
        # Validate dataset
        data_dir = Path(args.data_dir)
        if not validate_dataset_structure(data_dir, args.category, logger):
            logger.error("Dataset validation failed!")
            return
        
        # Create preprocessing pipeline
        logger.info("Setting up preprocessing pipeline...")
        preprocessor = MVTecPreprocessor(
            image_size=config.input_size,
            normalize=True,
            augmentation_prob=0.8 if args.category != 'test' else 0.5  # Less aggressive for test
        )
        
        # Create datasets and dataloaders
        logger.info("Creating datasets and dataloaders...")
        train_loader, val_loader = create_dataloaders(
            root_dir=data_dir,
            categories=args.category,
            preprocessor=preprocessor,
            batch_size=config.batch_size,
            num_workers=args.num_workers,
            load_masks=True  # Load masks for validation
        )
        
        # Log dataset statistics
        train_stats = train_loader.dataset.get_category_stats()
        val_stats = val_loader.dataset.get_category_stats()
        
        logger.info("Dataset Statistics:")
        logger.info(f"  Training: {train_stats}")
        logger.info(f"  Validation: {val_stats}")
        
        # Create model
        logger.info("Initializing model...")
        model = AutoencoderUNetLite(config)
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: {total_params * 4 / 1e6:.1f} MB (fp32)")
        
        # Create trainer
        logger.info("Setting up trainer...")
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=config,
            device=device,
            logger=logger,
            log_dir=output_dir / 'tensorboard'
        )
        
        # Resume from checkpoint if specified
        resume_epoch = 0
        if hasattr(args, 'resume') and args.resume:
            resume_path = Path(args.resume)
            if resume_path.exists():
                logger.info(f"Resuming from checkpoint: {resume_path}")
                resume_epoch = trainer.load_checkpoint(resume_path)
            else:
                logger.warning(f"Resume checkpoint not found: {resume_path}")
        
        # Start training
        logger.info("=" * 50)
        logger.info("TRAINING STARTED")
        logger.info("=" * 50)
        logger.info(f"Category: {args.category}")
        logger.info(f"Epochs: {config.num_epochs}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info(f"Device: {device}")
        logger.info(f"Mixed precision: {config.use_amp}")
        
        # Execute training
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=output_dir,
            resume_from=Path(args.resume) if hasattr(args, 'resume') and args.resume else None
        )
        
        # Training completed successfully
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Best validation loss: {results['best_loss']:.6f}")
        logger.info(f"Final epoch: {results['final_epoch']}")
        logger.info(f"Total training time: {results['total_training_time']:.1f}s")
        
        # Log threshold information
        if 'thresholds' in results:
            logger.info("Anomaly Detection Thresholds:")
            for name, value in results['thresholds'].items():
                logger.info(f"  {name}: {value:.6f}")
        
        # Save final results summary
        results_summary = {
            'category': args.category,
            'training_completed': True,
            'best_validation_loss': results['best_loss'],
            'final_epoch': results['final_epoch'],
            'total_training_time': results['total_training_time'],
            'thresholds': results.get('thresholds', {}),
            'model_path': str(output_dir / 'model.pth'),
            'config': config.__dict__
        }
        
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"All results saved to: {output_dir}")
        logger.info("Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train anomaly detection model on MVTec AD dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--category', 
        type=str, 
        required=True,
        choices=['capsule', 'hazelnut', 'screw', 'bottle', 'cable', 'carpet', 
                'grid', 'leather', 'metal_nut', 'pill', 'tile', 'toothbrush', 
                'transistor', 'wood', 'zipper'],
        help='MVTec AD category to train on'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/mvtec_ad',
        help='Path to MVTec AD dataset root directory'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for training results'
    )
    
    # Configuration arguments
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to custom configuration YAML file'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Custom experiment name for output directory'
    )
    
    # Training parameters (optional overrides)
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=None,
        help='Learning rate (overrides config)'
    )
    
    # System arguments
    parser.add_argument(
        '--num-workers', 
        type=int, 
        default=4,
        help='Number of dataloader workers'
    )
    
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Debugging arguments
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser


if __name__ == '__main__':
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run main training
    main(args)
