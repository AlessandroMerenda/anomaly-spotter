"""
Advanced Evaluation System for Anomaly Detection Models.
Provides comprehensive metrics, thresholding, and performance analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    average_precision_score, confusion_matrix,
    classification_report, f1_score
)
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import logging
from datetime import datetime

# Import core components
from src.utils.logging_utils import setup_logger


class AnomalyEvaluator:
    """
    Advanced Anomaly Detection Evaluator.
    
    Features:
    - Image-level and pixel-level evaluation
    - Multiple threshold optimization strategies
    - Comprehensive metrics calculation
    - Visualization and reporting
    - ROC/PR curve analysis
    - Statistical significance testing
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained anomaly detection model
            device: Computing device
            logger: Logger instance
        """
        self.logger = logger or setup_logger("AnomalyEvaluator")
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Cache for computed scores
        self._score_cache = {}
        
        self.logger.info(f"AnomalyEvaluator initialized on device: {device}")
    
    def compute_anomaly_map(self, 
                           image: torch.Tensor, 
                           gaussian_sigma: float = 4.0) -> np.ndarray:
        """
        Compute pixel-wise anomaly map for a single image.
        
        Args:
            image: Input image tensor [C, H, W]
            gaussian_sigma: Gaussian smoothing sigma
        
        Returns:
            Anomaly map as numpy array [H, W]
        """
        with torch.no_grad():
            # Ensure image has batch dimension
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            image = image.to(self.device)
            
            # Forward pass
            reconstructed = self.model(image)
            
            # Compute pixel-wise error (MSE)
            error = (image - reconstructed) ** 2
            anomaly_map = torch.mean(error, dim=1).squeeze(0)  # Average across channels
            
            # Convert to numpy
            anomaly_map = anomaly_map.cpu().numpy()
            
            # Apply Gaussian smoothing
            if gaussian_sigma > 0:
                anomaly_map = gaussian_filter(anomaly_map, sigma=gaussian_sigma)
        
        return anomaly_map
    
    def compute_batch_scores(self, 
                            images: torch.Tensor,
                            return_maps: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute anomaly scores for a batch of images.
        
        Args:
            images: Batch of images [B, C, H, W]
            return_maps: Whether to return anomaly maps
        
        Returns:
            Tuple of (image_scores, anomaly_maps)
        """
        with torch.no_grad():
            images = images.to(self.device)
            
            # Forward pass
            reconstructed = self.model(images)
            
            # Compute pixel-wise errors
            pixel_errors = (images - reconstructed) ** 2
            
            # Image-level scores (mean across all dimensions except batch)
            image_scores = torch.mean(pixel_errors, dim=(1, 2, 3))
            
            # Anomaly maps if requested
            anomaly_maps = None
            if return_maps:
                # Mean across channels, keep spatial dimensions
                anomaly_maps = torch.mean(pixel_errors, dim=1)
                anomaly_maps = anomaly_maps.cpu().numpy()
            
            image_scores = image_scores.cpu().numpy()
        
        return image_scores, anomaly_maps
    
    def evaluate_dataset(self, 
                        dataloader: DataLoader,
                        thresholds: Optional[Dict[str, float]] = None,
                        compute_pixel_metrics: bool = True,
                        save_predictions: bool = False) -> Dict[str, Any]:
        """
        Comprehensive evaluation on dataset.
        
        Args:
            dataloader: DataLoader with test data
            thresholds: Dictionary of threshold values to evaluate
            compute_pixel_metrics: Whether to compute pixel-level metrics
            save_predictions: Whether to save prediction results
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        self.logger.info("Starting dataset evaluation...")
        
        # Initialize collectors
        all_labels = []
        all_scores = []
        all_paths = []
        all_categories = []
        
        # Pixel-level collectors (only for anomalous images)
        pixel_labels_list = []
        pixel_scores_list = []
        
        # Predictions storage
        predictions_data = [] if save_predictions else None
        
        # Process dataset
        total_samples = 0
        anomaly_count = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating dataset')):
            try:
                images = batch['image']
                labels = batch['label'].numpy()
                paths = batch['path']
                categories = batch['category']
                masks = batch.get('mask', None)
                
                # Compute anomaly scores
                image_scores, anomaly_maps = self.compute_batch_scores(
                    images, 
                    return_maps=compute_pixel_metrics
                )
                
                # Collect image-level data
                all_labels.extend(labels)
                all_scores.extend(image_scores)
                all_paths.extend(paths)
                all_categories.extend(categories)
                
                total_samples += len(labels)
                anomaly_count += np.sum(labels)
                
                # Process pixel-level data for anomalous images
                if compute_pixel_metrics and masks is not None and anomaly_maps is not None:
                    for i, (label, mask) in enumerate(zip(labels, masks)):
                        if label == 1:  # Only for anomalous images
                            # Convert mask to numpy if needed
                            if torch.is_tensor(mask):
                                mask_np = mask.squeeze().cpu().numpy()
                            else:
                                mask_np = mask.squeeze()
                            
                            # Skip if mask is empty/dummy
                            if mask_np.shape == () or np.all(mask_np == 0):
                                continue
                            
                            # Get corresponding anomaly map
                            anom_map = anomaly_maps[i]
                            
                            # Ensure same spatial dimensions
                            if mask_np.shape != anom_map.shape:
                                # Resize anomaly map to match mask
                                from scipy.ndimage import zoom
                                scale_factors = [m/a for m, a in zip(mask_np.shape, anom_map.shape)]
                                anom_map = zoom(anom_map, scale_factors, order=1)
                            
                            # Flatten and collect
                            pixel_labels_list.append(mask_np.flatten())
                            pixel_scores_list.append(anom_map.flatten())
                
                # Store predictions if requested
                if save_predictions:
                    for i in range(len(labels)):
                        predictions_data.append({
                            'path': paths[i],
                            'category': categories[i],
                            'true_label': int(labels[i]),
                            'anomaly_score': float(image_scores[i]),
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        })
            
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        self.logger.info(f"Processed {total_samples} samples ({anomaly_count} anomalies)")
        
        # Compute metrics
        results = self._compute_comprehensive_metrics(
            all_labels, all_scores, thresholds, 
            pixel_labels_list, pixel_scores_list
        )
        
        # Add dataset info
        results['dataset_info'] = {
            'total_samples': total_samples,
            'normal_samples': total_samples - anomaly_count,
            'anomaly_samples': anomaly_count,
            'categories': list(set(all_categories)),
            'evaluation_time': datetime.now().isoformat()
        }
        
        # Add predictions if requested
        if save_predictions:
            results['predictions'] = predictions_data
        
        self.logger.info("Dataset evaluation completed")
        return results
    
    def _compute_comprehensive_metrics(self,
                                     labels: np.ndarray,
                                     scores: np.ndarray,
                                     thresholds: Optional[Dict[str, float]],
                                     pixel_labels_list: List[np.ndarray],
                                     pixel_scores_list: List[np.ndarray]) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        
        results = {}
        
        # Image-level metrics
        try:
            # ROC AUC
            auroc_image = roc_auc_score(labels, scores)
            results['auroc_image'] = float(auroc_image)
            
            # Average Precision (PR AUC)
            ap_image = average_precision_score(labels, scores)
            results['ap_image'] = float(ap_image)
            
            # ROC and PR curves
            fpr, tpr, roc_thresholds = roc_curve(labels, scores)
            precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
            
            results['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }
            
            results['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error computing image-level metrics: {e}")
            results['auroc_image'] = 0.0
            results['ap_image'] = 0.0
        
        # Threshold-based metrics
        if thresholds:
            threshold_results = {}
            for name, threshold in thresholds.items():
                try:
                    predictions = (scores > threshold).astype(int)
                    
                    # Basic metrics
                    accuracy = np.mean(predictions == labels)
                    f1 = f1_score(labels, predictions)
                    
                    # Confusion matrix
                    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                    
                    # Derived metrics
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    threshold_results[name] = {
                        'threshold': float(threshold),
                        'accuracy': float(accuracy),
                        'f1_score': float(f1),
                        'precision': float(precision),
                        'recall': float(recall),
                        'specificity': float(specificity),
                        'true_positives': int(tp),
                        'true_negatives': int(tn),
                        'false_positives': int(fp),
                        'false_negatives': int(fn)
                    }
                
                except Exception as e:
                    self.logger.error(f"Error computing metrics for threshold {name}: {e}")
                    continue
            
            results['threshold_metrics'] = threshold_results
        
        # Pixel-level metrics
        if pixel_labels_list and pixel_scores_list:
            try:
                # Concatenate all pixel data
                pixel_labels_flat = np.concatenate(pixel_labels_list)
                pixel_scores_flat = np.concatenate(pixel_scores_list)
                
                # Remove invalid entries
                valid_mask = ~(np.isnan(pixel_scores_flat) | np.isinf(pixel_scores_flat))
                pixel_labels_flat = pixel_labels_flat[valid_mask]
                pixel_scores_flat = pixel_scores_flat[valid_mask]
                
                if len(pixel_labels_flat) > 0:
                    # Pixel-level AUROC
                    auroc_pixel = roc_auc_score(pixel_labels_flat, pixel_scores_flat)
                    ap_pixel = average_precision_score(pixel_labels_flat, pixel_scores_flat)
                    
                    results['auroc_pixel'] = float(auroc_pixel)
                    results['ap_pixel'] = float(ap_pixel)
                    
                    # Pixel statistics
                    results['pixel_stats'] = {
                        'total_pixels': len(pixel_labels_flat),
                        'anomalous_pixels': int(np.sum(pixel_labels_flat)),
                        'normal_pixels': int(len(pixel_labels_flat) - np.sum(pixel_labels_flat))
                    }
                else:
                    self.logger.warning("No valid pixel data for evaluation")
            
            except Exception as e:
                self.logger.error(f"Error computing pixel-level metrics: {e}")
        
        # Score statistics
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        results['score_statistics'] = {
            'normal_mean': float(np.mean(normal_scores)),
            'normal_std': float(np.std(normal_scores)),
            'normal_min': float(np.min(normal_scores)),
            'normal_max': float(np.max(normal_scores)),
            'anomaly_mean': float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0,
            'anomaly_std': float(np.std(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0,
            'anomaly_min': float(np.min(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0,
            'anomaly_max': float(np.max(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0,
        }
        
        return results
    
    def find_optimal_threshold(self,
                             dataloader: DataLoader,
                             metric: str = 'f1',
                             validation_split: float = 0.5) -> Dict[str, float]:
        """
        Find optimal threshold using various strategies.
        
        Args:
            dataloader: DataLoader with validation data
            metric: Optimization metric ('f1', 'youden', 'balanced_accuracy')
            validation_split: Fraction of data to use for threshold optimization
        
        Returns:
            Dictionary with optimal thresholds for different strategies
        """
        self.logger.info(f"Finding optimal thresholds using {metric} metric...")
        
        # Collect all scores and labels
        all_labels = []
        all_scores = []
        
        for batch in tqdm(dataloader, desc='Collecting scores'):
            try:
                images = batch['image']
                labels = batch['label']
                
                # Compute scores
                image_scores, _ = self.compute_batch_scores(images)
                
                all_labels.extend(labels.numpy())
                all_scores.extend(image_scores)
            
            except Exception as e:
                self.logger.error(f"Error in threshold optimization batch: {e}")
                continue
        
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        if len(all_labels) == 0:
            self.logger.error("No valid data for threshold optimization")
            return {}
        
        # Split data if requested
        if validation_split < 1.0:
            n_samples = len(all_labels)
            n_val = int(n_samples * validation_split)
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            
            all_labels = all_labels[val_indices]
            all_scores = all_scores[val_indices]
        
        optimal_thresholds = {}
        
        try:
            # ROC-based thresholds
            fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores)
            
            # Youden's J statistic (maximizes TPR - FPR)
            j_scores = tpr - fpr
            best_j_idx = np.argmax(j_scores)
            optimal_thresholds['youden'] = float(roc_thresholds[best_j_idx])
            
            # Balanced accuracy threshold (closest to top-left corner)
            distances = np.sqrt((1 - tpr)**2 + fpr**2)
            best_dist_idx = np.argmin(distances)
            optimal_thresholds['balanced_accuracy'] = float(roc_thresholds[best_dist_idx])
            
            # PR-based thresholds
            precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_scores)
            
            # F1 score maximization
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            best_f1_idx = np.argmax(f1_scores)
            optimal_thresholds['f1'] = float(pr_thresholds[best_f1_idx])
            
            # Precision-Recall balance
            pr_balance = np.abs(precision[:-1] - recall[:-1])
            best_balance_idx = np.argmin(pr_balance)
            optimal_thresholds['precision_recall_balance'] = float(pr_thresholds[best_balance_idx])
            
            # Statistical thresholds
            normal_scores = all_scores[all_labels == 0]
            
            # Mean + k*std thresholds
            mean_normal = np.mean(normal_scores)
            std_normal = np.std(normal_scores)
            
            optimal_thresholds['mean_plus_2std'] = float(mean_normal + 2 * std_normal)
            optimal_thresholds['mean_plus_3std'] = float(mean_normal + 3 * std_normal)
            
            # Percentile thresholds
            optimal_thresholds['percentile_95'] = float(np.percentile(normal_scores, 95))
            optimal_thresholds['percentile_99'] = float(np.percentile(normal_scores, 99))
            optimal_thresholds['percentile_999'] = float(np.percentile(normal_scores, 99.9))
            
            # Median Absolute Deviation (MAD) threshold
            median_normal = np.median(normal_scores)
            mad_normal = np.median(np.abs(normal_scores - median_normal))
            optimal_thresholds['mad_3'] = float(median_normal + 3 * mad_normal)
            
        except Exception as e:
            self.logger.error(f"Error computing optimal thresholds: {e}")
        
        # Log results
        self.logger.info("Optimal thresholds found:")
        for name, value in optimal_thresholds.items():
            self.logger.info(f"  {name}: {value:.6f}")
        
        return optimal_thresholds
    
    def generate_evaluation_report(self,
                                 results: Dict[str, Any],
                                 save_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save report
        
        Returns:
            Report string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("ANOMALY DETECTION EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Dataset info
        if 'dataset_info' in results:
            info = results['dataset_info']
            report_lines.append("Dataset Information:")
            report_lines.append(f"  Total samples: {info['total_samples']}")
            report_lines.append(f"  Normal samples: {info['normal_samples']}")
            report_lines.append(f"  Anomaly samples: {info['anomaly_samples']}")
            report_lines.append(f"  Categories: {', '.join(info['categories'])}")
            report_lines.append(f"  Evaluation time: {info['evaluation_time']}")
            report_lines.append("")
        
        # Image-level metrics
        report_lines.append("Image-Level Performance:")
        report_lines.append(f"  AUROC: {results.get('auroc_image', 0.0):.4f}")
        report_lines.append(f"  Average Precision: {results.get('ap_image', 0.0):.4f}")
        report_lines.append("")
        
        # Pixel-level metrics
        if 'auroc_pixel' in results:
            report_lines.append("Pixel-Level Performance:")
            report_lines.append(f"  AUROC: {results['auroc_pixel']:.4f}")
            report_lines.append(f"  Average Precision: {results.get('ap_pixel', 0.0):.4f}")
            
            if 'pixel_stats' in results:
                stats = results['pixel_stats']
                report_lines.append(f"  Total pixels evaluated: {stats['total_pixels']:,}")
                report_lines.append(f"  Anomalous pixels: {stats['anomalous_pixels']:,}")
                report_lines.append("")
        
        # Threshold metrics
        if 'threshold_metrics' in results:
            report_lines.append("Threshold-Based Performance:")
            for name, metrics in results['threshold_metrics'].items():
                report_lines.append(f"  {name.upper()} (threshold: {metrics['threshold']:.6f}):")
                report_lines.append(f"    Accuracy: {metrics['accuracy']:.4f}")
                report_lines.append(f"    F1 Score: {metrics['f1_score']:.4f}")
                report_lines.append(f"    Precision: {metrics['precision']:.4f}")
                report_lines.append(f"    Recall: {metrics['recall']:.4f}")
                report_lines.append(f"    Specificity: {metrics['specificity']:.4f}")
                report_lines.append("")
        
        # Score statistics
        if 'score_statistics' in results:
            stats = results['score_statistics']
            report_lines.append("Score Statistics:")
            report_lines.append("  Normal samples:")
            report_lines.append(f"    Mean: {stats['normal_mean']:.6f}")
            report_lines.append(f"    Std: {stats['normal_std']:.6f}")
            report_lines.append(f"    Range: [{stats['normal_min']:.6f}, {stats['normal_max']:.6f}]")
            report_lines.append("  Anomaly samples:")
            report_lines.append(f"    Mean: {stats['anomaly_mean']:.6f}")
            report_lines.append(f"    Std: {stats['anomaly_std']:.6f}")
            report_lines.append(f"    Range: [{stats['anomaly_min']:.6f}, {stats['anomaly_max']:.6f}]")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # Join and optionally save
        report = "\n".join(report_lines)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Evaluation report saved to: {save_path}")
        
        return report
    
    def clear_cache(self):
        """Clear internal score cache."""
        self._score_cache.clear()
        self.logger.info("Score cache cleared")


# Utility functions
def create_evaluator(model: nn.Module, device: str = 'cuda') -> AnomalyEvaluator:
    """
    Factory function to create evaluator.
    
    Args:
        model: Trained model
        device: Computing device
    
    Returns:
        Configured evaluator instance
    """
    return AnomalyEvaluator(model, device)


def evaluate_model(model: nn.Module,
                  dataloader: DataLoader,
                  thresholds: Optional[Dict[str, float]] = None,
                  device: str = 'cuda',
                  save_results: Optional[Path] = None) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        thresholds: Threshold dictionary
        device: Computing device
        save_results: Path to save results
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = create_evaluator(model, device)
    results = evaluator.evaluate_dataset(dataloader, thresholds)
    
    if save_results:
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(save_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save text report
        report = evaluator.generate_evaluation_report(results)
        with open(save_path.with_suffix('.txt'), 'w') as f:
            f.write(report)
    
    return results
