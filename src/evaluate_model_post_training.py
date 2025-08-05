#!/usr/bin/env python3
"""
Script di Valutazione Post-Training per Anomaly Spotter.

Valuta modelli addestrati con metriche complete, visualizzazioni avanzate
e generazione di report dettagliati. Supporta valutazione singola categoria
e multi-categoria con threshold optimization avanzata.

Usage:
    python src/evaluate_model_post_training.py --model-dir outputs/capsule_20241205_143022 --category capsule
    python src/evaluate_model_post_training.py --model-dir outputs/multi_20241205_150030 --category all --find-optimal
"""

import argparse
import json
import yaml
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import system components
from src.core.model import AutoencoderUNetLite
from src.core.config import AnomalySpotterConfig
from src.evaluation.evaluator import AnomalyEvaluator
from src.data.loaders import MVTecDataset, create_dataloaders
from src.data.preprocessing import MVTecPreprocessor
from src.utils.logging_utils import setup_logger


class PostTrainingEvaluator:
    """
    Comprehensive post-training evaluation system.
    
    Features:
    - Model loading and validation
    - Dataset preparation with ground truth masks
    - Advanced metrics calculation
    - Threshold optimization
    - Professional visualizations
    - Detailed reporting
    """
    
    def __init__(self, model_dir: Path, config_path: Optional[Path] = None):
        """Initialize post-training evaluator."""
        self.model_dir = Path(model_dir)
        self.logger = setup_logger("PostTrainingEvaluator")
        
        # Validate model directory
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Required files
        self.model_path = self.model_dir / 'model.pth'
        self.config_path = config_path or self.model_dir / 'config.yaml'
        self.thresholds_path = self.model_dir / 'thresholds.json'
        
        # Validate required files
        self._validate_model_files()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize results directory
        self.results_dir = self.model_dir / 'post_training_evaluation'
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized PostTrainingEvaluator for {self.model_dir}")
    
    def _validate_model_files(self):
        """Validate that all required model files exist."""
        required_files = [self.model_path, self.config_path]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        self.logger.info("✅ All required model files validated")
    
    def _load_config(self) -> AnomalySpotterConfig:
        """Load model configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create config object
            config = AnomalySpotterConfig(**config_dict)
            self.logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Fallback to default config
            self.logger.warning("Using default configuration")
            return AnomalySpotterConfig()
    
    def load_model(self, device: str = 'auto') -> nn.Module:
        """Load trained model."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Initialize model
            model = AutoencoderUNetLite()
            
            # Load state dict
            state_dict = torch.load(self.model_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # Move to device and set eval mode
            model.to(device)
            model.eval()
            
            # Verify model
            self._verify_model(model, device)
            
            self.logger.info(f"✅ Model loaded successfully on {device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _verify_model(self, model: nn.Module, device: str):
        """Verify model works correctly."""
        try:
            with torch.no_grad():
                # Test forward pass
                test_input = torch.randn(1, 3, 
                                       self.config.model.input_size[0], 
                                       self.config.model.input_size[1]).to(device)
                test_output = model(test_input)
                
                # Verify output shape
                if test_output.shape != test_input.shape:
                    raise ValueError(f"Model output shape {test_output.shape} != input shape {test_input.shape}")
                
                self.logger.debug("Model verification passed")
                
        except Exception as e:
            raise RuntimeError(f"Model verification failed: {e}")
    
    def prepare_datasets(self, 
                        categories: List[str], 
                        data_dir: str = 'data/mvtec_ad',
                        batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and test datasets."""
        try:
            # Create preprocessor
            preprocessor = MVTecPreprocessor(
                image_size=self.config.model.input_size,
                augmentation_config=self.config.data.augmentation
            )
            
            # Create dataloaders
            train_loader, test_loader = create_dataloaders(
                root_dir=data_dir,
                categories=categories,
                preprocessor=preprocessor,
                batch_size=batch_size,
                num_workers=self.config.training.num_workers,
                load_masks=True  # Essential for evaluation
            )
            
            self.logger.info(f"✅ Datasets prepared for categories: {categories}")
            self.logger.info(f"   Train samples: {len(train_loader.dataset)}")
            self.logger.info(f"   Test samples: {len(test_loader.dataset)}")
            
            return train_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Failed to prepare datasets: {e}")
            raise
    
    def evaluate_model(self, 
                      model: nn.Module,
                      test_loader: DataLoader,
                      categories: List[str],
                      device: str = 'cuda',
                      find_optimal: bool = False) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            categories: List of categories to evaluate
            device: Device for computation
            find_optimal: Whether to find optimal thresholds
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        try:
            # Initialize evaluator
            evaluator = AnomalyEvaluator(model, device=device, logger=self.logger)
            
            self.logger.info("Starting comprehensive model evaluation...")
            
            # Load existing thresholds if available
            thresholds = {}
            if self.thresholds_path.exists():
                with open(self.thresholds_path, 'r') as f:
                    thresholds = json.load(f)
                self.logger.info(f"✅ Loaded existing thresholds: {len(thresholds)} entries")
            
            # Perform evaluation
            results = evaluator.evaluate_dataset(
                test_loader=test_loader,
                thresholds=thresholds,
                save_dir=self.results_dir
            )
            
            # Find optimal thresholds if requested
            if find_optimal:
                self.logger.info("Finding optimal thresholds...")
                optimal_results = self._find_optimal_thresholds(
                    evaluator, test_loader, categories
                )
                results.update(optimal_results)
            
            # Add metadata
            results['evaluation_metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'model_dir': str(self.model_dir),
                'categories': categories,
                'device': device,
                'total_test_samples': len(test_loader.dataset)
            }
            
            self.logger.info("✅ Model evaluation completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _find_optimal_thresholds(self, 
                                evaluator: AnomalyEvaluator,
                                test_loader: DataLoader,
                                categories: List[str]) -> Dict[str, Any]:
        """Find optimal thresholds using multiple strategies."""
        optimal_results = {}
        
        strategies = ['statistical', 'percentile_95', 'validation_f1', 'youden']
        
        for strategy in strategies:
            try:
                self.logger.info(f"Computing optimal threshold using {strategy} strategy...")
                
                threshold_results = evaluator.compute_optimal_thresholds(
                    test_loader=test_loader,
                    strategy=strategy,
                    categories=categories
                )
                
                optimal_results[f'optimal_{strategy}'] = threshold_results
                
            except Exception as e:
                self.logger.warning(f"Failed to compute {strategy} threshold: {e}")
        
        return optimal_results
    
    def visualize_results(self, 
                         model: nn.Module,
                         test_dataset: MVTecDataset,
                         results: Dict[str, Any],
                         num_samples: int = 10) -> None:
        """Generate comprehensive visualizations."""
        try:
            self.logger.info("Generating evaluation visualizations...")
            
            # 1. Anomaly detection samples
            self._visualize_anomaly_samples(model, test_dataset, num_samples)
            
            # 2. Performance metrics
            self._visualize_performance_metrics(results)
            
            # 3. ROC/PR curves (if available in results)
            if 'roc_curves' in results:
                self._visualize_roc_pr_curves(results)
            
            # 4. Threshold analysis
            if any('optimal_' in key for key in results.keys()):
                self._visualize_threshold_analysis(results)
            
            self.logger.info("✅ Visualizations generated")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
    
    def _visualize_anomaly_samples(self, 
                                  model: nn.Module,
                                  dataset: MVTecDataset,
                                  num_samples: int) -> None:
        """Visualize anomaly detection results on sample images."""
        # Get anomaly samples
        anomaly_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
        
        if not anomaly_indices:
            self.logger.warning("No anomaly samples found for visualization")
            return
        
        # Sample random anomalies
        sample_indices = np.random.choice(
            anomaly_indices, 
            min(num_samples, len(anomaly_indices)), 
            replace=False
        )
        
        # Create visualization
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        device = next(model.parameters()).device
        
        for idx, sample_idx in enumerate(sample_indices):
            try:
                # Get sample
                sample = dataset[sample_idx]
                image = sample['image']
                mask = sample['mask']
                category = sample['category']
                
                # Model inference
                with torch.no_grad():
                    image_batch = image.unsqueeze(0).to(device)
                    reconstructed = model(image_batch)[0].cpu()
                
                # Compute anomaly map
                anomaly_map = torch.abs(image - reconstructed).mean(dim=0).numpy()
                
                # Denormalize for visualization
                image_vis = self._denormalize_image(image)
                recon_vis = self._denormalize_image(reconstructed)
                
                # Plot original
                axes[idx, 0].imshow(image_vis)
                axes[idx, 0].set_title(f'Original\n{category}')
                axes[idx, 0].axis('off')
                
                # Plot reconstruction
                axes[idx, 1].imshow(recon_vis)
                axes[idx, 1].set_title('Reconstructed')
                axes[idx, 1].axis('off')
                
                # Plot anomaly map
                im = axes[idx, 2].imshow(anomaly_map, cmap='hot')
                axes[idx, 2].set_title('Anomaly Map')
                axes[idx, 2].axis('off')
                plt.colorbar(im, ax=axes[idx, 2], fraction=0.046)
                
                # Plot ground truth mask
                if isinstance(mask, torch.Tensor) and mask.numel() > 1:
                    axes[idx, 3].imshow(mask.squeeze().numpy(), cmap='gray')
                    axes[idx, 3].set_title('Ground Truth')
                else:
                    axes[idx, 3].text(0.5, 0.5, 'No GT Mask', 
                                     ha='center', va='center', transform=axes[idx, 3].transAxes)
                axes[idx, 3].axis('off')
                
            except Exception as e:
                self.logger.warning(f"Failed to visualize sample {sample_idx}: {e}")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'anomaly_detection_samples.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _denormalize_image(self, tensor_image: torch.Tensor) -> np.ndarray:
        """Denormalize tensor image for visualization."""
        # Assuming normalization with mean=0.5, std=0.5 -> range [-1, 1]
        # Convert to [0, 1] range
        image = tensor_image.clone()
        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy and transpose
        if image.dim() == 3:
            return image.permute(1, 2, 0).numpy()
        else:
            return image.numpy()
    
    def _visualize_performance_metrics(self, results: Dict[str, Any]) -> None:
        """Visualize performance metrics."""
        # Extract metric values
        metrics = {}
        for key, value in results.items():
            if isinstance(value, (int, float)) and key not in ['timestamp', 'total_test_samples']:
                if any(metric in key.lower() for metric in ['auroc', 'auprc', 'f1', 'precision', 'recall', 'accuracy']):
                    metrics[key] = value
        
        if not metrics:
            self.logger.warning("No metrics found for visualization")
            return
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUROC and AUPRC metrics
        score_metrics = {k: v for k, v in metrics.items() 
                        if any(x in k.lower() for x in ['auroc', 'auprc'])}
        if score_metrics:
            ax1.bar(range(len(score_metrics)), list(score_metrics.values()))
            ax1.set_xticks(range(len(score_metrics)))
            ax1.set_xticklabels(list(score_metrics.keys()), rotation=45, ha='right')
            ax1.set_title('ROC and PR Scores')
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Score')
        
        # Other performance metrics
        other_metrics = {k: v for k, v in metrics.items() 
                        if not any(x in k.lower() for x in ['auroc', 'auprc'])}
        if other_metrics:
            ax2.bar(range(len(other_metrics)), list(other_metrics.values()))
            ax2.set_xticks(range(len(other_metrics)))
            ax2.set_xticklabels(list(other_metrics.keys()), rotation=45, ha='right')
            ax2.set_title('Performance Metrics')
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_metrics.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_roc_pr_curves(self, results: Dict[str, Any]) -> None:
        """Visualize ROC and PR curves if available."""
        # This would be implemented if ROC/PR curve data is available in results
        self.logger.info("ROC/PR curve visualization not implemented yet")
    
    def _visualize_threshold_analysis(self, results: Dict[str, Any]) -> None:
        """Visualize threshold analysis results."""
        # Extract threshold results
        threshold_results = {k: v for k, v in results.items() if 'optimal_' in k}
        
        if not threshold_results:
            return
        
        # Create comparison plot
        strategies = []
        thresholds = []
        
        for key, value in threshold_results.items():
            if isinstance(value, dict) and 'threshold' in value:
                strategy = key.replace('optimal_', '')
                strategies.append(strategy)
                thresholds.append(value['threshold'])
        
        if strategies and thresholds:
            plt.figure(figsize=(10, 6))
            plt.bar(strategies, thresholds)
            plt.title('Optimal Thresholds by Strategy')
            plt.xlabel('Strategy')
            plt.ylabel('Threshold Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'threshold_analysis.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, results: Dict[str, Any], categories: List[str]) -> None:
        """Generate comprehensive evaluation report."""
        try:
            report_path = self.results_dir / 'evaluation_report.md'
            
            with open(report_path, 'w') as f:
                f.write("# Post-Training Evaluation Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Model Directory:** {self.model_dir}\n")
                f.write(f"**Categories:** {', '.join(categories)}\n\n")
                
                # Model Information
                f.write("## Model Information\n\n")
                f.write(f"- **Architecture:** {self.config.model.architecture}\n")
                f.write(f"- **Input Size:** {self.config.model.input_size}\n")
                f.write(f"- **Device:** {results.get('evaluation_metadata', {}).get('device', 'unknown')}\n\n")
                
                # Performance Metrics
                f.write("## Performance Metrics\n\n")
                for key, value in results.items():
                    if isinstance(value, (int, float)) and key not in ['timestamp']:
                        if any(metric in key.lower() for metric in ['auroc', 'auprc', 'f1', 'precision', 'recall', 'accuracy']):
                            f.write(f"- **{key.replace('_', ' ').title()}:** {value:.4f}\n")
                
                # Threshold Analysis
                threshold_results = {k: v for k, v in results.items() if 'optimal_' in k}
                if threshold_results:
                    f.write("\n## Optimal Thresholds\n\n")
                    for key, value in threshold_results.items():
                        strategy = key.replace('optimal_', '')
                        if isinstance(value, dict) and 'threshold' in value:
                            f.write(f"- **{strategy.replace('_', ' ').title()}:** {value['threshold']:.6f}\n")
                
                # Files Generated
                f.write("\n## Generated Files\n\n")
                f.write("- `anomaly_detection_samples.png` - Sample anomaly visualizations\n")
                f.write("- `performance_metrics.png` - Performance metrics bar charts\n")
                f.write("- `evaluation_results.json` - Complete results in JSON format\n")
                f.write("- `evaluation_report.md` - This report\n")
            
            self.logger.info(f"✅ Evaluation report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON."""
        try:
            results_path = self.results_dir / 'evaluation_results.json'
            
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(results)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"✅ Results saved to {results_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Post-Training Evaluation for Anomaly Spotter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained model and configuration')
    
    # Data arguments
    parser.add_argument('--category', type=str, default='capsule',
                       help='Category to evaluate (capsule/hazelnut/screw/all)')
    parser.add_argument('--data-dir', type=str, default='data/mvtec_ad',
                       help='Path to MVTec dataset root directory')
    
    # Evaluation arguments
    parser.add_argument('--find-optimal', action='store_true',
                       help='Find optimal thresholds using multiple strategies')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for computation')
    
    # Visualization arguments
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    
    # Output arguments
    parser.add_argument('--config-path', type=str, default=None,
                       help='Custom path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = PostTrainingEvaluator(
            model_dir=args.model_dir,
            config_path=args.config_path
        )
        
        # Determine categories
        if args.category.lower() == 'all':
            categories = ['capsule', 'hazelnut', 'screw']
        else:
            categories = [args.category]
        
        print(f"\n🔍 Starting Post-Training Evaluation")
        print(f"   Model: {args.model_dir}")
        print(f"   Categories: {categories}")
        print(f"   Device: {args.device}")
        
        # Load model
        model = evaluator.load_model(device=args.device)
        
        # Prepare datasets
        train_loader, test_loader = evaluator.prepare_datasets(
            categories=categories,
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        
        # Evaluate model
        device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        results = evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            categories=categories,
            device=device,
            find_optimal=args.find_optimal
        )
        
        # Generate visualizations
        if not args.no_visualize:
            evaluator.visualize_results(
                model=model,
                test_dataset=test_loader.dataset,
                results=results,
                num_samples=args.num_samples
            )
        
        # Generate report
        evaluator.generate_report(results, categories)
        
        # Save results
        evaluator.save_results(results)
        
        # Print summary
        print(f"\n📊 EVALUATION RESULTS")
        print(f"=" * 50)
        for key, value in results.items():
            if isinstance(value, (int, float)) and key not in ['timestamp']:
                if any(metric in key.lower() for metric in ['auroc', 'auprc', 'f1', 'precision', 'recall', 'accuracy']):
                    print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {evaluator.results_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
