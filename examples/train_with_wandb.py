#!/usr/bin/env python3
"""
Example training script with Weights & Biases integration.

Demonstrates how to integrate the WandbLogger with the existing training pipeline
for comprehensive experiment tracking and monitoring.

Usage:
    python examples/train_with_wandb.py --category capsule --use-wandb
    python examples/train_with_wandb.py --category hazelnut --use-wandb --wandb-project my-project
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import core components
from core.model import AutoencoderUNetLite
from core.config import AnomalySpotterConfig
from data.loaders import create_dataloaders
from data.preprocessing import MVTecPreprocessor
from training.trainer import AnomalyTrainer
from evaluation.evaluator import AnomalyEvaluator
from utils.wandb_logger import WandbLogger, init_wandb_logging
from utils.logging_utils import setup_logger


class WandbIntegratedTrainer:
    """
    Training pipeline with integrated W&B logging.
    
    Demonstrates best practices for experiment tracking:
    - Automatic configuration logging
    - Real-time metrics tracking
    - Image visualization
    - Model artifact saving
    - Evaluation results integration
    """
    
    def __init__(self, 
                 config: AnomalySpotterConfig,
                 use_wandb: bool = True,
                 wandb_project: str = "anomaly-detection",
                 wandb_tags: list = None):
        """Initialize integrated trainer."""
        self.config = config
        self.logger = setup_logger("WandbIntegratedTrainer")
        
        # Initialize W&B logging
        self.wandb_logger = None
        if use_wandb:
            wandb_config = self._prepare_wandb_config()
            self.wandb_logger = init_wandb_logging(
                config=wandb_config,
                project=wandb_project,
                name=f"anomaly_{config.data.category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=wandb_tags or ["anomaly-detection", config.data.category]
            )
            
            if self.wandb_logger.is_active():
                self.wandb_logger.log_system_info()
                self.logger.info("✅ W&B logging initialized")
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trainer = None
        self.evaluator = None
    
    def _prepare_wandb_config(self) -> dict:
        """Prepare configuration for W&B logging."""
        return {
            # Model configuration
            "model_architecture": self.config.model.architecture,
            "model_input_size": self.config.model.input_size,
            
            # Training configuration
            "category": self.config.data.category,
            "batch_size": self.config.training.batch_size,
            "learning_rate": self.config.training.learning_rate,
            "epochs": self.config.training.epochs,
            "optimizer": self.config.training.optimizer,
            
            # Loss configuration
            "loss_mse_weight": self.config.training.loss_weights.get("mse", 1.0),
            "loss_ssim_weight": self.config.training.loss_weights.get("ssim", 0.5),
            "loss_perceptual_weight": self.config.training.loss_weights.get("perceptual", 0.1),
            
            # Data configuration
            "data_root": str(self.config.data.root_dir),
            "augmentation_enabled": self.config.data.augmentation.enabled,
            "image_size": self.config.data.image_size,
            
            # System configuration
            "device": str(self.device),
            "num_workers": self.config.training.num_workers,
        }
    
    def setup_model_and_training(self):
        """Setup model, dataloaders, and training components."""
        self.logger.info("Setting up model and training components...")
        
        # Initialize model
        self.model = AutoencoderUNetLite()
        self.model.to(self.device)
        
        # Watch model with W&B
        if self.wandb_logger and self.wandb_logger.is_active():
            self.wandb_logger.watch_model(self.model, log_freq=100)
        
        # Setup preprocessing
        preprocessor = MVTecPreprocessor(
            image_size=self.config.model.input_size,
            augmentation_config=self.config.data.augmentation
        )
        
        # Create dataloaders
        self.train_loader, self.test_loader = create_dataloaders(
            root_dir=self.config.data.root_dir,
            categories=self.config.data.category,
            preprocessor=preprocessor,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            load_masks=True
        )
        
        # Initialize trainer with W&B integration
        self.trainer = AnomalyTrainer(
            model=self.model,
            config=self.config,
            device=self.device,
            logger=self.logger
        )
        
        # Initialize evaluator
        self.evaluator = AnomalyEvaluator(
            model=self.model,
            device=self.device,
            logger=self.logger
        )
        
        self.logger.info("✅ Model and training setup completed")
    
    def train_with_logging(self) -> dict:
        """
        Execute training with comprehensive W&B logging.
        
        Returns:
            Training results dictionary
        """
        self.logger.info("Starting training with W&B logging...")
        
        # Setup training callbacks for W&B logging
        def on_epoch_end(epoch, train_metrics, val_metrics, model_state):
            """Callback executed at the end of each epoch."""
            
            # Log training metrics
            if self.wandb_logger:
                wandb_metrics = {
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()}
                }
                self.wandb_logger.log_metrics(wandb_metrics, step=epoch)
                
                # Log sample reconstructions every 10 epochs
                if epoch % 10 == 0:
                    self._log_reconstruction_samples(epoch)
        
        def on_training_complete(final_metrics, model_path):
            """Callback executed when training completes."""
            
            if self.wandb_logger:
                # Log final metrics
                final_wandb_metrics = {f"final/{k}": v for k, v in final_metrics.items()}
                self.wandb_logger.log_metrics(final_wandb_metrics)
                
                # Log model artifact
                self.wandb_logger.log_model_artifact(
                    model_path=model_path,
                    artifact_name=f"model_{self.config.data.category}",
                    metadata={
                        "category": self.config.data.category,
                        "final_loss": final_metrics.get("loss", 0),
                        "epochs_trained": final_metrics.get("epoch", 0)
                    }
                )
        
        # Execute training with callbacks
        training_results = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.test_loader,
            on_epoch_end=on_epoch_end,
            on_training_complete=on_training_complete
        )
        
        self.logger.info("✅ Training completed")
        return training_results
    
    def evaluate_with_logging(self, model_path: str = None) -> dict:
        """
        Execute evaluation with W&B logging.
        
        Args:
            model_path: Path to trained model (optional)
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Starting evaluation with W&B logging...")
        
        # Load model if path provided
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        
        # Execute comprehensive evaluation
        eval_results = self.evaluator.evaluate_dataset(
            test_loader=self.test_loader,
            thresholds={},  # Will compute optimal thresholds
            save_dir=None  # Don't save to disk, log to W&B instead
        )
        
        # Log evaluation results to W&B
        if self.wandb_logger:
            self.wandb_logger.log_evaluation_results(eval_results)
            
            # Log detailed anomaly visualizations
            self._log_detailed_evaluation_results()
        
        self.logger.info("✅ Evaluation completed")
        return eval_results
    
    def _log_reconstruction_samples(self, epoch: int):
        """Log sample reconstructions to W&B."""
        if not self.wandb_logger or not self.wandb_logger.is_active():
            return
        
        try:
            # Get a batch of test data
            batch = next(iter(self.test_loader))
            images = batch['image'][:8].to(self.device)  # First 8 samples
            labels = batch['label'][:8]
            masks = batch['mask'][:8] if 'mask' in batch else None
            
            # Generate reconstructions
            with torch.no_grad():
                reconstructed = self.model(images)
                
                # Compute anomaly maps
                anomaly_maps = torch.abs(images - reconstructed).mean(dim=1)
            
            # Convert to numpy for logging
            orig_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            recon_np = reconstructed.detach().cpu().numpy().transpose(0, 2, 3, 1)
            anomaly_np = anomaly_maps.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy() if masks is not None else None
            labels_np = labels.numpy()
            
            # Log to W&B
            self.wandb_logger.log_anomaly_results(
                original_images=orig_np,
                reconstructed_images=recon_np,
                anomaly_maps=anomaly_np,
                ground_truth_masks=masks_np,
                labels=labels_np,
                step=epoch,
                max_images=8
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to log reconstruction samples: {e}")
    
    def _log_detailed_evaluation_results(self):
        """Log detailed evaluation visualizations to W&B."""
        if not self.wandb_logger or not self.wandb_logger.is_active():
            return
        
        try:
            # This would generate and log comprehensive evaluation visualizations
            # Including ROC curves, PR curves, confusion matrices, etc.
            
            # For now, log a placeholder
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'Detailed Evaluation Results\n(Implementation Placeholder)', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            self.wandb_logger.log_images({
                "evaluation_summary": fig
            })
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"Failed to log detailed evaluation results: {e}")
    
    def run_complete_pipeline(self) -> dict:
        """
        Run complete training and evaluation pipeline with W&B logging.
        
        Returns:
            Complete results dictionary
        """
        self.logger.info("🚀 Starting complete pipeline with W&B integration")
        
        # Setup
        self.setup_model_and_training()
        
        # Training
        training_results = self.train_with_logging()
        
        # Evaluation
        eval_results = self.evaluate_with_logging()
        
        # Combine results
        complete_results = {
            "training": training_results,
            "evaluation": eval_results,
            "config": self._prepare_wandb_config()
        }
        
        # Final W&B logging
        if self.wandb_logger:
            # Log summary metrics
            summary_metrics = {
                "pipeline/training_loss": training_results.get("final_loss", 0),
                "pipeline/evaluation_auroc": eval_results.get("auroc", 0),
                "pipeline/evaluation_f1": eval_results.get("f1", 0),
            }
            self.wandb_logger.log_metrics(summary_metrics)
            
            # Finish W&B run
            self.wandb_logger.finish()
        
        self.logger.success("✅ Complete pipeline finished successfully")
        return complete_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Training with Weights & Biases Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--category', type=str, default='capsule',
                       choices=['capsule', 'hazelnut', 'screw'],
                       help='MVTec category to train on')
    parser.add_argument('--data-dir', type=str, default='data/mvtec_ad',
                       help='Path to MVTec dataset')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    
    # W&B arguments
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='anomaly-detection-demo',
                       help='W&B project name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', 
                       default=['demo', 'example'],
                       help='W&B run tags')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = AnomalySpotterConfig(
            data={
                "category": args.category,
                "root_dir": args.data_dir,
                "image_size": [128, 128]
            },
            training={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "optimizer": "adam"
            },
            model={
                "architecture": "AutoencoderUNetLite",
                "input_size": [128, 128]
            }
        )
        
        # Initialize integrated trainer
        trainer = WandbIntegratedTrainer(
            config=config,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags
        )
        
        # Run complete pipeline
        results = trainer.run_complete_pipeline()
        
        print("\n🎉 Training completed successfully!")
        if args.use_wandb and trainer.wandb_logger:
            print(f"📊 View results at: https://wandb.ai/{args.wandb_project}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
