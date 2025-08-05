"""
Advanced Weights & Biases Logger for Anomaly Spotter.

Provides comprehensive experiment tracking, model monitoring, and visualization
for anomaly detection training and evaluation workflows.

Features:
- Automatic experiment initialization
- Metrics and loss tracking
- Image visualization with annotations
- Model artifact logging
- Hyperparameter sweeps support
- Custom charts and dashboards
- Integration with evaluation pipeline
"""

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings

# Import utilities
from src.utils.logging_utils import setup_logger


class WandbLogger:
    """
    Advanced Weights & Biases logger for anomaly detection experiments.
    
    Features:
    - Automatic experiment setup and configuration
    - Training metrics and loss tracking
    - Image logging with anomaly visualizations
    - Model checkpoints and artifacts
    - Evaluation results integration
    - Custom charts and dashboards
    """
    
    def __init__(self, 
                 project: str = "anomaly-detection",
                 entity: Optional[str] = None,
                 name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None,
                 resume: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            name: Run name (auto-generated if None)
            config: Experiment configuration dictionary
            tags: List of tags for the run
            notes: Run description/notes
            resume: Resume mode ('allow', 'must', 'never', 'auto')
            logger: Python logger instance
        """
        self.logger = logger or setup_logger("WandbLogger")
        self.project = project
        self.entity = entity
        
        # Generate run name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            category = config.get('category', 'unknown') if config else 'unknown'
            name = f"anomaly_detection_{category}_{timestamp}"
        
        # Initialize W&B run
        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config or {},
                tags=tags or [],
                notes=notes,
                resume=resume,
                reinit=True  # Allow multiple runs in same process
            )
            
            self.logger.info(f"✅ W&B run initialized: {self.run.name}")
            self.logger.info(f"   Project: {project}")
            self.logger.info(f"   URL: {self.run.url}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {e}")
            self.logger.warning("Continuing without W&B logging")
            self.run = None
    
    def is_active(self) -> bool:
        """Check if W&B logging is active."""
        return self.run is not None
    
    def log_metrics(self, 
                   metrics: Dict[str, Union[float, int]], 
                   step: Optional[int] = None,
                   commit: bool = True) -> None:
        """
        Log training/validation metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Step number (epoch, iteration, etc.)
            commit: Whether to commit the log entry
        """
        if not self.is_active():
            return
        
        try:
            # Filter out None values and ensure numeric types
            clean_metrics = {}
            for key, value in metrics.items():
                if value is not None and isinstance(value, (int, float, np.number)):
                    clean_metrics[key] = float(value)
            
            if clean_metrics:
                wandb.log(clean_metrics, step=step, commit=commit)
                self.logger.debug(f"Logged metrics: {list(clean_metrics.keys())}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {e}")
    
    def log_images(self, 
                  images: Dict[str, Any], 
                  step: Optional[int] = None,
                  commit: bool = True) -> None:
        """
        Log images with captions and annotations.
        
        Args:
            images: Dictionary of image_name -> image_data
            step: Step number
            commit: Whether to commit the log entry
        """
        if not self.is_active():
            return
        
        try:
            wandb_images = {}
            
            for key, img_data in images.items():
                if img_data is None:
                    continue
                
                # Handle different image formats
                if isinstance(img_data, dict):
                    # Expected format: {'image': array, 'caption': str, 'masks': dict}
                    image_array = img_data.get('image')
                    caption = img_data.get('caption', key)
                    masks = img_data.get('masks', {})
                    boxes = img_data.get('boxes', {})
                    
                    if image_array is not None:
                        wandb_image = wandb.Image(
                            image_array,
                            caption=caption,
                            masks=masks,
                            boxes=boxes
                        )
                        wandb_images[key] = wandb_image
                
                elif isinstance(img_data, (np.ndarray, torch.Tensor)):
                    # Direct image array
                    wandb_images[key] = wandb.Image(img_data, caption=key)
                
                elif isinstance(img_data, plt.Figure):
                    # Matplotlib figure
                    wandb_images[key] = wandb.Image(img_data, caption=key)
                
                else:
                    # Try to convert to image
                    try:
                        wandb_images[key] = wandb.Image(img_data, caption=key)
                    except Exception as e:
                        self.logger.warning(f"Could not convert image {key}: {e}")
            
            if wandb_images:
                wandb.log(wandb_images, step=step, commit=commit)
                self.logger.debug(f"Logged images: {list(wandb_images.keys())}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log images: {e}")
    
    def log_anomaly_results(self,
                           original_images: np.ndarray,
                           reconstructed_images: np.ndarray,
                           anomaly_maps: np.ndarray,
                           ground_truth_masks: Optional[np.ndarray] = None,
                           predictions: Optional[np.ndarray] = None,
                           labels: Optional[np.ndarray] = None,
                           step: Optional[int] = None,
                           max_images: int = 8) -> None:
        """
        Log anomaly detection results with comprehensive visualizations.
        
        Args:
            original_images: Original input images [B, H, W, C]
            reconstructed_images: Model reconstructions [B, H, W, C]
            anomaly_maps: Computed anomaly maps [B, H, W]
            ground_truth_masks: GT anomaly masks [B, H, W] (optional)
            predictions: Binary predictions [B] (optional)
            labels: True labels [B] (optional)
            step: Step number
            max_images: Maximum number of images to log
        """
        if not self.is_active():
            return
        
        try:
            batch_size = min(original_images.shape[0], max_images)
            
            # Convert tensors to numpy if needed
            if isinstance(original_images, torch.Tensor):
                original_images = original_images.detach().cpu().numpy()
            if isinstance(reconstructed_images, torch.Tensor):
                reconstructed_images = reconstructed_images.detach().cpu().numpy()
            if isinstance(anomaly_maps, torch.Tensor):
                anomaly_maps = anomaly_maps.detach().cpu().numpy()
            
            logged_images = {}
            
            for i in range(batch_size):
                # Prepare image data
                orig_img = original_images[i]
                recon_img = reconstructed_images[i]
                anomaly_map = anomaly_maps[i]
                
                # Normalize images for display
                orig_img_norm = self._normalize_for_display(orig_img)
                recon_img_norm = self._normalize_for_display(recon_img)
                anomaly_map_norm = self._normalize_anomaly_map(anomaly_map)
                
                # Create image captions
                caption_parts = [f"Sample {i}"]
                if labels is not None:
                    label_str = "Anomaly" if labels[i] == 1 else "Normal"
                    caption_parts.append(f"GT: {label_str}")
                if predictions is not None:
                    pred_str = "Anomaly" if predictions[i] == 1 else "Normal"
                    caption_parts.append(f"Pred: {pred_str}")
                
                caption = " | ".join(caption_parts)
                
                # Log individual images
                logged_images[f"original_{i}"] = wandb.Image(
                    orig_img_norm,
                    caption=f"{caption} - Original"
                )
                
                logged_images[f"reconstructed_{i}"] = wandb.Image(
                    recon_img_norm,
                    caption=f"{caption} - Reconstructed"
                )
                
                # Prepare masks for anomaly map
                masks = {}
                if ground_truth_masks is not None and i < ground_truth_masks.shape[0]:
                    gt_mask = ground_truth_masks[i]
                    if isinstance(gt_mask, torch.Tensor):
                        gt_mask = gt_mask.detach().cpu().numpy()
                    
                    # Create W&B mask
                    if gt_mask.sum() > 0:  # Only if mask has content
                        masks["ground_truth"] = {
                            "mask_data": gt_mask,
                            "class_labels": {1: "anomaly", 0: "normal"}
                        }
                
                logged_images[f"anomaly_map_{i}"] = wandb.Image(
                    anomaly_map_norm,
                    caption=f"{caption} - Anomaly Map",
                    masks=masks
                )
            
            # Log all images
            wandb.log(logged_images, step=step)
            self.logger.debug(f"Logged {batch_size} anomaly result sets")
            
        except Exception as e:
            self.logger.warning(f"Failed to log anomaly results: {e}")
    
    def log_model_artifact(self,
                          model_path: Union[str, Path],
                          artifact_name: str = "model",
                          artifact_type: str = "model",
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log model as W&B artifact.
        
        Args:
            model_path: Path to model file
            artifact_name: Name for the artifact
            artifact_type: Type of artifact
            metadata: Additional metadata
        """
        if not self.is_active():
            return
        
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata or {}
            )
            
            artifact.add_file(str(model_path))
            self.run.log_artifact(artifact)
            
            self.logger.info(f"✅ Model artifact logged: {artifact_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log model artifact: {e}")
    
    def log_evaluation_results(self,
                             results: Dict[str, Any],
                             step: Optional[int] = None) -> None:
        """
        Log comprehensive evaluation results.
        
        Args:
            results: Evaluation results dictionary
            step: Step number
        """
        if not self.is_active():
            return
        
        try:
            # Extract and log metrics
            metrics = {}
            for key, value in results.items():
                if isinstance(value, (int, float, np.number)):
                    metrics[f"eval/{key}"] = float(value)
            
            if metrics:
                wandb.log(metrics, step=step)
            
            # Log results as artifact
            results_artifact = wandb.Artifact(
                name="evaluation_results",
                type="evaluation",
                metadata={"step": step}
            )
            
            # Save results to temporary file
            temp_path = Path("temp_eval_results.json")
            with open(temp_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            results_artifact.add_file(str(temp_path))
            self.run.log_artifact(results_artifact)
            
            # Cleanup
            temp_path.unlink()
            
            self.logger.debug("Evaluation results logged to W&B")
            
        except Exception as e:
            self.logger.warning(f"Failed to log evaluation results: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to W&B config.
        
        Args:
            params: Hyperparameter dictionary
        """
        if not self.is_active():
            return
        
        try:
            # Update run config
            for key, value in params.items():
                wandb.config[key] = value
            
            self.logger.debug(f"Hyperparameters logged: {list(params.keys())}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_system_info(self) -> None:
        """Log system and environment information."""
        if not self.is_active():
            return
        
        try:
            import platform
            import psutil
            
            system_info = {
                "system/platform": platform.platform(),
                "system/python_version": platform.python_version(),
                "system/cpu_count": psutil.cpu_count(),
                "system/memory_gb": psutil.virtual_memory().total / (1024**3),
            }
            
            # GPU information
            if torch.cuda.is_available():
                system_info.update({
                    "system/gpu_count": torch.cuda.device_count(),
                    "system/gpu_name": torch.cuda.get_device_name(0),
                    "system/cuda_version": torch.version.cuda,
                })
            
            # PyTorch version
            system_info["system/pytorch_version"] = torch.__version__
            
            wandb.log(system_info)
            self.logger.debug("System information logged")
            
        except Exception as e:
            self.logger.warning(f"Failed to log system info: {e}")
    
    def create_custom_chart(self,
                           chart_name: str,
                           chart_type: str = "line",
                           x_axis: str = "step",
                           y_axis: str = "value") -> None:
        """
        Create custom chart in W&B dashboard.
        
        Args:
            chart_name: Name of the chart
            chart_type: Type of chart ('line', 'bar', 'scatter')
            x_axis: X-axis metric name
            y_axis: Y-axis metric name
        """
        if not self.is_active():
            return
        
        try:
            # This would be implemented based on W&B's chart API
            # For now, just log the chart configuration
            chart_config = {
                f"chart/{chart_name}/type": chart_type,
                f"chart/{chart_name}/x_axis": x_axis,
                f"chart/{chart_name}/y_axis": y_axis
            }
            
            wandb.log(chart_config)
            self.logger.debug(f"Custom chart configured: {chart_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create custom chart: {e}")
    
    def _normalize_for_display(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for display."""
        if image.dtype == np.uint8:
            return image
        
        # Assume image is in [-1, 1] or [0, 1] range
        if image.min() >= 0 and image.max() <= 1:
            # [0, 1] range
            return (image * 255).astype(np.uint8)
        else:
            # [-1, 1] range
            image = (image + 1) / 2  # Convert to [0, 1]
            return (image * 255).astype(np.uint8)
    
    def _normalize_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Normalize anomaly map for display."""
        # Min-max normalization
        if anomaly_map.max() > anomaly_map.min():
            normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        else:
            normalized = anomaly_map
        
        # Convert to colormap
        import matplotlib.cm as cm
        colormap = cm.get_cmap('hot')
        colored = colormap(normalized)
        
        # Convert to uint8 RGB
        return (colored[:, :, :3] * 255).astype(np.uint8)
    
    def watch_model(self, 
                   model: torch.nn.Module,
                   log_freq: int = 100,
                   log_graph: bool = True) -> None:
        """
        Watch model for gradient and parameter tracking.
        
        Args:
            model: PyTorch model to watch
            log_freq: Logging frequency
            log_graph: Whether to log model graph
        """
        if not self.is_active():
            return
        
        try:
            wandb.watch(
                model,
                log="all",  # Log gradients and parameters
                log_freq=log_freq,
                log_graph=log_graph
            )
            
            self.logger.info("✅ Model watching enabled")
            
        except Exception as e:
            self.logger.warning(f"Failed to watch model: {e}")
    
    def finish(self) -> None:
        """Finish the W&B run."""
        if self.is_active():
            try:
                wandb.finish()
                self.logger.info("✅ W&B run finished")
            except Exception as e:
                self.logger.warning(f"Error finishing W&B run: {e}")
            finally:
                self.run = None


class WandbSweepManager:
    """
    Manager for W&B hyperparameter sweeps.
    
    Handles sweep configuration, execution, and results analysis
    for automated hyperparameter optimization.
    """
    
    def __init__(self, 
                 project: str = "anomaly-detection-sweeps",
                 entity: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize sweep manager."""
        self.project = project
        self.entity = entity
        self.logger = logger or setup_logger("WandbSweepManager")
    
    def create_sweep(self,
                    sweep_config: Dict[str, Any],
                    sweep_name: Optional[str] = None) -> str:
        """
        Create a new hyperparameter sweep.
        
        Args:
            sweep_config: Sweep configuration dictionary
            sweep_name: Name for the sweep
            
        Returns:
            Sweep ID
        """
        try:
            sweep_id = wandb.sweep(
                sweep=sweep_config,
                project=self.project,
                entity=self.entity
            )
            
            self.logger.info(f"✅ Sweep created: {sweep_id}")
            return sweep_id
            
        except Exception as e:
            self.logger.error(f"Failed to create sweep: {e}")
            raise
    
    def get_default_sweep_config(self,
                                category: str = "capsule") -> Dict[str, Any]:
        """
        Get default sweep configuration for anomaly detection.
        
        Args:
            category: MVTec category
            
        Returns:
            Default sweep configuration
        """
        return {
            "method": "bayes",  # Bayesian optimization
            "metric": {
                "name": "eval/auroc",
                "goal": "maximize"
            },
            "parameters": {
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-2
                },
                "batch_size": {
                    "values": [16, 32, 64]
                },
                "loss_weights": {
                    "mse_weight": {
                        "distribution": "uniform",
                        "min": 0.5,
                        "max": 2.0
                    },
                    "ssim_weight": {
                        "distribution": "uniform",
                        "min": 0.1,
                        "max": 1.0
                    }
                },
                "augmentation": {
                    "rotation_limit": {
                        "distribution": "int_uniform",
                        "min": 10,
                        "max": 45
                    },
                    "noise_factor": {
                        "distribution": "uniform",
                        "min": 0.01,
                        "max": 0.1
                    }
                },
                "category": {
                    "value": category
                }
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 10,
                "eta": 2
            }
        }


# Utility functions for easy integration
def init_wandb_logging(config: Dict[str, Any], 
                      project: str = "anomaly-detection",
                      name: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> WandbLogger:
    """
    Initialize W&B logging with sensible defaults.
    
    Args:
        config: Training configuration
        project: W&B project name
        name: Run name
        tags: Run tags
        
    Returns:
        Configured WandbLogger instance
    """
    return WandbLogger(
        project=project,
        name=name,
        config=config,
        tags=tags or ["anomaly-detection", config.get("category", "unknown")]
    )


def log_training_batch(logger: WandbLogger,
                      metrics: Dict[str, float],
                      images: Optional[Dict[str, Any]] = None,
                      step: int = 0) -> None:
    """
    Log a training batch with metrics and optional images.
    
    Args:
        logger: WandbLogger instance
        metrics: Training metrics
        images: Optional images to log
        step: Training step
    """
    logger.log_metrics(metrics, step=step, commit=False)
    
    if images:
        logger.log_images(images, step=step, commit=True)
    else:
        # Commit metrics if no images
        wandb.log({}, commit=True)


# Example usage and integration patterns
if __name__ == "__main__":
    # Example configuration
    config = {
        "category": "capsule",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 100,
        "model": "AutoencoderUNetLite"
    }
    
    # Initialize logger
    wandb_logger = init_wandb_logging(
        config=config,
        project="anomaly-detection-demo",
        tags=["demo", "capsule"]
    )
    
    # Log system info
    wandb_logger.log_system_info()
    
    # Example training loop logging
    for epoch in range(5):
        # Training metrics
        train_metrics = {
            "train/loss": 0.1 - epoch * 0.01,
            "train/learning_rate": 1e-4,
            "epoch": epoch
        }
        
        wandb_logger.log_metrics(train_metrics, step=epoch)
        
        # Validation metrics
        val_metrics = {
            "val/loss": 0.12 - epoch * 0.008,
            "val/auroc": 0.85 + epoch * 0.02
        }
        
        wandb_logger.log_metrics(val_metrics, step=epoch)
    
    # Finish run
    wandb_logger.finish()
    
    print("Demo completed successfully!")
