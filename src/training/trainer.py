"""
Advanced Trainer for Anomaly Detection with AutoencoderUNetLite.
Integrates with existing architecture and provides comprehensive training pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, List
import time
import logging
from datetime import datetime

# Import core components
from src.core.model import AutoencoderUNetLite
from src.core.model_config import AutoencoderConfig
from src.core.losses import CombinedLoss, create_loss_function
from src.utils.logging_utils import setup_logger


class AnomalyDetectorTrainer:
    """
    Advanced Trainer for Anomaly Detection Models.
    
    Features:
    - Mixed precision training (AMP)
    - Advanced scheduling and optimization
    - Tensorboard logging with image visualization
    - Comprehensive checkpointing
    - Threshold calculation for inference
    - Early stopping with patience
    - Robust error handling
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: AutoencoderConfig,
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None,
                 log_dir: Optional[Path] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Training device ('cuda' or 'cpu')
            logger: Logger instance
            log_dir: Directory for tensorboard logs
        """
        self.logger = logger or setup_logger("AnomalyTrainer")
        self.config = config
        self.device = device
        
        # Setup model
        self.model = model.to(device)
        self.logger.info(f"Model moved to device: {device}")
        
        # Setup optimizer with advanced settings
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Mixed precision setup
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        if self.use_amp:
            self.logger.info("Mixed precision training enabled")
        
        # Tensorboard logging
        if log_dir is None:
            log_dir = Path('runs') / f'anomaly_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        self.writer = SummaryWriter(str(log_dir))
        self.logger.info(f"Tensorboard logs: {log_dir}")
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Statistics tracking
        self.loss_stats = {}
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with advanced configuration."""
        if self.config.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        self.logger.info(f"Optimizer: {self.config.optimizer_type}")
        return optimizer
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.scheduler_patience,
                min_lr=1e-7,
                verbose=True
            )
        elif self.config.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-7
            )
        elif self.config.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        else:
            # No scheduler
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0
            )
        
        self.logger.info(f"Scheduler: {self.config.scheduler_type}")
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        try:
            loss_fn = create_loss_function(self.config)
            self.logger.info(f"Loss function created: {type(loss_fn).__name__}")
            return loss_fn
        except Exception as e:
            self.logger.warning(f"Failed to create advanced loss function: {e}")
            # Fallback to simple MSE
            return nn.MSELoss()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, loss_components_dict)
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = len(dataloader)
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch:3d}/{self.config.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move data to device
                images = batch['image'].to(self.device, non_blocking=True)
                batch_size = images.size(0)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    reconstructed = self.model(images)
                    
                    # Calculate loss
                    if hasattr(self.criterion, 'forward') and len(self.criterion.forward.__code__.co_varnames) > 2:
                        # Advanced loss function that returns components
                        loss, losses_dict = self.criterion(reconstructed, images)
                    else:
                        # Simple loss function
                        loss = self.criterion(reconstructed, images)
                        losses_dict = {'total': loss}
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    # Scaled backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping before step
                    self.scaler.unscale_(self.optimizer)
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard backward pass
                    loss.backward()
                    
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip
                        )
                    
                    self.optimizer.step()
                
                # Update tracking
                total_loss += loss.item()
                
                # Track loss components
                if isinstance(losses_dict, dict):
                    for k, v in losses_dict.items():
                        if torch.is_tensor(v):
                            v = v.item()
                        loss_components[k] = loss_components.get(k, 0) + v
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # Tensorboard logging
                if batch_idx % max(1, num_batches // 20) == 0:  # Log 20 times per epoch
                    step = epoch * num_batches + batch_idx
                    self.writer.add_scalar('train/batch_loss', loss.item(), step)
                    self.writer.add_scalar('train/learning_rate', current_lr, step)
                    
                    # Log loss components
                    if isinstance(losses_dict, dict):
                        for k, v in losses_dict.items():
                            if torch.is_tensor(v):
                                v = v.item()
                            self.writer.add_scalar(f'train/loss_{k}', v, step)
                    
                    # Log images periodically
                    if batch_idx % max(1, num_batches // 5) == 0:  # Log 5 times per epoch
                        self._log_images(images, reconstructed, step, 'train')
                
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        for k in loss_components:
            loss_components[k] /= num_batches
        
        return avg_loss, loss_components
    
    def validate(self, dataloader: DataLoader, epoch: int) -> Tuple[float, np.ndarray]:
        """
        Validate model on validation set.
        
        Args:
            dataloader: Validation dataloader
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, reconstruction_errors)
        """
        self.model.eval()
        total_loss = 0.0
        all_errors = []
        num_batches = len(dataloader)
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move data to device
                    images = batch['image'].to(self.device, non_blocking=True)
                    labels = batch.get('label', None)
                    
                    # Forward pass
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        reconstructed = self.model(images)
                        
                        # Calculate loss
                        if hasattr(self.criterion, 'forward') and len(self.criterion.forward.__code__.co_varnames) > 2:
                            loss, _ = self.criterion(reconstructed, images)
                        else:
                            loss = self.criterion(reconstructed, images)
                    
                    total_loss += loss.item()
                    
                    # Calculate pixel-wise reconstruction errors
                    pixel_errors = torch.mean((images - reconstructed) ** 2, dim=1, keepdim=True)
                    image_errors = torch.mean(pixel_errors, dim=(1, 2, 3))
                    
                    all_errors.extend(image_errors.cpu().numpy())
                    
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                    
                    # Log validation images periodically
                    if batch_idx % max(1, num_batches // 3) == 0:
                        step = epoch * num_batches + batch_idx
                        self._log_images(images, reconstructed, step, 'val')
                
                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches
        return avg_loss, np.array(all_errors)
    
    def calculate_threshold(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Calculate thresholds for anomaly detection using training data.
        
        Args:
            train_loader: Training dataloader (normal samples only)
        
        Returns:
            Dictionary with various threshold values
        """
        self.logger.info("Calculating anomaly detection thresholds...")
        
        self.model.eval()
        all_errors = []
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc='Computing thresholds'):
                try:
                    images = batch['image'].to(self.device, non_blocking=True)
                    
                    # Forward pass
                    reconstructed = self.model(images)
                    
                    # Calculate pixel-wise errors
                    pixel_errors = torch.mean((images - reconstructed) ** 2, dim=1, keepdim=True)
                    image_errors = torch.mean(pixel_errors, dim=(1, 2, 3))
                    
                    all_errors.extend(image_errors.cpu().numpy())
                
                except Exception as e:
                    self.logger.error(f"Error calculating thresholds: {e}")
                    continue
        
        if not all_errors:
            self.logger.error("No valid errors computed for threshold calculation")
            return {}
        
        all_errors = np.array(all_errors)
        
        # Calculate various threshold strategies
        thresholds = {
            'percentile_99': float(np.percentile(all_errors, 99)),
            'percentile_95': float(np.percentile(all_errors, 95)),
            'percentile_90': float(np.percentile(all_errors, 90)),
            'mean': float(np.mean(all_errors)),
            'std': float(np.std(all_errors)),
            'adaptive': float(np.mean(all_errors) + 3 * np.std(all_errors)),
            'iqr_outlier': float(np.percentile(all_errors, 75) + 1.5 * (np.percentile(all_errors, 75) - np.percentile(all_errors, 25))),
            'median_mad': float(np.median(all_errors) + 3 * np.median(np.abs(all_errors - np.median(all_errors))))
        }
        
        self.logger.info("Thresholds calculated:")
        for name, value in thresholds.items():
            self.logger.info(f"  {name}: {value:.6f}")
        
        return thresholds
    
    def save_checkpoint(self, epoch: int, loss: float, save_dir: Path) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Current validation loss
            save_dir: Directory to save checkpoint
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        # Save latest checkpoint
        latest_path = save_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model if improved
        if loss < self.best_loss:
            self.logger.info(f"New best model! Loss: {loss:.6f} (prev: {self.best_loss:.6f})")
            self.best_loss = loss
            
            # Save best checkpoint
            best_path = save_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            
            # Save production model (state dict only)
            model_path = save_dir / 'model.pth'
            torch.save(self.model.state_dict(), model_path)
            
            # Reset patience counter
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        self.logger.info(f"Checkpoint saved: {latest_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Last epoch number
        """
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', {
                'train_loss': [], 'val_loss': [], 'learning_rates': [], 'epoch_times': []
            })
            
            # Load scaler state if available
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.logger.info(f"Checkpoint loaded: epoch {self.current_epoch}, best_loss: {self.best_loss:.6f}")
            return self.current_epoch
        
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return 0
    
    def _log_images(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                    step: int, phase: str = 'train') -> None:
        """
        Log images to tensorboard.
        
        Args:
            original: Original images
            reconstructed: Reconstructed images
            step: Current step
            phase: Training phase ('train' or 'val')
        """
        try:
            # Take first 4 images
            n = min(4, original.size(0))
            orig_sample = original[:n].clone()
            recon_sample = reconstructed[:n].clone()
            
            # Denormalize if using ImageNet normalization
            if hasattr(self.config, 'normalize') and self.config.normalize:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
                
                orig_sample = orig_sample * std + mean
                recon_sample = recon_sample * std + mean
            
            # Clamp values to [0, 1]
            orig_sample = torch.clamp(orig_sample, 0, 1)
            recon_sample = torch.clamp(recon_sample, 0, 1)
            
            # Create comparison grid: [original1, recon1, original2, recon2, ...]
            comparison = torch.cat([orig_sample, recon_sample], dim=0)
            
            # Log to tensorboard
            self.writer.add_images(f'{phase}/reconstruction', comparison, step)
            
            # Also log error maps
            error_maps = torch.abs(orig_sample - recon_sample)
            self.writer.add_images(f'{phase}/error_maps', error_maps, step)
        
        except Exception as e:
            self.logger.error(f"Failed to log images: {e}")
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return (self.config.early_stopping_patience > 0 and 
                self.patience_counter >= self.config.early_stopping_patience)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              save_dir: Path, resume_from: Optional[Path] = None) -> Dict[str, any]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            save_dir: Directory to save outputs
            resume_from: Path to checkpoint to resume from
        
        Returns:
            Training results dictionary
        """
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        self.logger.info(f"Starting training from epoch {start_epoch}")
        self.logger.info(f"Total epochs: {self.config.epochs}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        
        # Training loop
        for epoch in range(start_epoch, self.config.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_components = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss, val_errors = self.validate(val_loader, epoch)
            
            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Update history
            epoch_time = time.time() - epoch_start_time
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epoch_times'].append(epoch_time)
            
            # Tensorboard logging
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log epoch summary
            self.logger.info(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                           f"lr={self.optimizer.param_groups[0]['lr']:.2e}, time={epoch_time:.1f}s")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, save_dir)
            
            # Early stopping check
            if self.should_stop_early():
                self.logger.info(f"Early stopping at epoch {epoch} (patience: {self.patience_counter})")
                break
        
        # Calculate final thresholds
        thresholds = self.calculate_threshold(train_loader)
        
        # Save thresholds
        thresholds_path = save_dir / 'thresholds.json'
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        
        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Close tensorboard
        self.writer.close()
        
        # Return results
        results = {
            'best_loss': self.best_loss,
            'final_epoch': epoch if 'epoch' in locals() else self.config.epochs - 1,
            'thresholds': thresholds,
            'training_history': self.training_history,
            'total_training_time': sum(self.training_history['epoch_times'])
        }
        
        self.logger.info("Training completed successfully!")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")
        self.logger.info(f"Total training time: {results['total_training_time']:.1f}s")
        
        return results


# Utility functions for easy trainer creation
def create_trainer(model: nn.Module, config: AutoencoderConfig, 
                  device: str = 'cuda') -> AnomalyDetectorTrainer:
    """
    Factory function to create trainer with default settings.
    
    Args:
        model: PyTorch model
        config: Training configuration
        device: Training device
    
    Returns:
        Configured trainer instance
    """
    return AnomalyDetectorTrainer(model, config, device)


def train_anomaly_detector(model: nn.Module, 
                          config: AutoencoderConfig,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          save_dir: Path,
                          device: str = 'cuda',
                          resume_from: Optional[Path] = None) -> Dict[str, any]:
    """
    Complete training pipeline function.
    
    Args:
        model: PyTorch model to train
        config: Training configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        save_dir: Directory to save outputs
        device: Training device
        resume_from: Optional checkpoint to resume from
    
    Returns:
        Training results dictionary
    """
    trainer = create_trainer(model, config, device)
    return trainer.train(train_loader, val_loader, save_dir, resume_from)
