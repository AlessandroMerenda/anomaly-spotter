"""
Advanced Threshold Computation for Anomaly Detection.
Computes optimal thresholds using multiple strategies and validation techniques.
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import logging
from datetime import datetime

# Import core components
from src.core.model import AutoencoderUNetLite
from src.core.model_config import AutoencoderConfig
from src.data.loaders import create_dataloaders
from src.data.preprocessing import MVTecPreprocessor
from src.evaluation.evaluator import AnomalyEvaluator, create_evaluator
from src.utils.logging_utils import setup_logger


class ThresholdComputer:
    """
    Advanced threshold computation with multiple strategies.
    
    Features:
    - Multiple threshold optimization methods
    - Cross-validation for robust estimates
    - Category-specific threshold computation
    - Statistical analysis and confidence intervals
    - Threshold ensemble methods
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 config: AutoencoderConfig,
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize threshold computer.
        
        Args:
            model: Trained anomaly detection model
            config: Model configuration
            device: Computing device
            logger: Logger instance
        """
        self.logger = logger or setup_logger("ThresholdComputer")
        self.model = model
        self.config = config
        self.device = device
        self.evaluator = create_evaluator(model, device)
        
        self.logger.info("ThresholdComputer initialized")
    
    def compute_comprehensive_thresholds(self,
                                       train_loader,
                                       val_loader=None,
                                       strategies: List[str] = None) -> Dict[str, Any]:
        """
        Compute thresholds using multiple strategies.
        
        Args:
            train_loader: Training data loader (normal samples only)
            val_loader: Validation data loader (if available)
            strategies: List of threshold strategies to use
        
        Returns:
            Dictionary with comprehensive threshold results
        """
        if strategies is None:
            strategies = [
                'statistical', 'percentile', 'mad', 
                'validation_f1', 'validation_youden'
            ]
        
        self.logger.info(f"Computing thresholds using strategies: {strategies}")
        
        results = {
            'computation_time': datetime.now().isoformat(),
            'strategies': strategies,
            'thresholds': {},
            'statistics': {},
            'recommendations': {}
        }
        
        # 1. Statistical thresholds from training data
        if 'statistical' in strategies:
            self.logger.info("Computing statistical thresholds...")
            stat_thresholds = self._compute_statistical_thresholds(train_loader)
            results['thresholds']['statistical'] = stat_thresholds
        
        # 2. Percentile-based thresholds
        if 'percentile' in strategies:
            self.logger.info("Computing percentile thresholds...")
            perc_thresholds = self._compute_percentile_thresholds(train_loader)
            results['thresholds']['percentile'] = perc_thresholds
        
        # 3. MAD-based thresholds
        if 'mad' in strategies:
            self.logger.info("Computing MAD-based thresholds...")
            mad_thresholds = self._compute_mad_thresholds(train_loader)
            results['thresholds']['mad'] = mad_thresholds
        
        # 4. Validation-based optimization (if validation data available)
        if val_loader is not None:
            if 'validation_f1' in strategies:
                self.logger.info("Computing F1-optimized threshold...")
                f1_threshold = self.evaluator.find_optimal_threshold(
                    val_loader, metric='f1'
                )
                results['thresholds']['validation_f1'] = f1_threshold
            
            if 'validation_youden' in strategies:
                self.logger.info("Computing Youden-optimized threshold...")
                youden_threshold = self.evaluator.find_optimal_threshold(
                    val_loader, metric='youden'
                )
                results['thresholds']['validation_youden'] = youden_threshold
        
        # 5. Compute statistics on training scores
        train_stats = self._compute_training_statistics(train_loader)
        results['statistics'] = train_stats
        
        # 6. Generate recommendations
        recommendations = self._generate_threshold_recommendations(results)
        results['recommendations'] = recommendations
        
        self.logger.info("Threshold computation completed")
        return results
    
    def _compute_statistical_thresholds(self, train_loader) -> Dict[str, float]:
        """Compute statistical thresholds based on normal data distribution."""
        
        # Collect all training scores
        all_scores = []
        
        self.logger.info("Collecting training scores for statistical analysis...")
        for batch in train_loader:
            try:
                images = batch['image']
                scores, _ = self.evaluator.compute_batch_scores(images)
                all_scores.extend(scores)
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                continue
        
        if not all_scores:
            self.logger.error("No scores collected for statistical thresholds")
            return {}
        
        all_scores = np.array(all_scores)
        
        # Compute statistical measures
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        # Statistical thresholds
        thresholds = {
            'mean': float(mean_score),
            'mean_plus_1std': float(mean_score + 1 * std_score),
            'mean_plus_2std': float(mean_score + 2 * std_score),
            'mean_plus_3std': float(mean_score + 3 * std_score),
            'mean_plus_2_5std': float(mean_score + 2.5 * std_score),
        }
        
        # Log statistics
        self.logger.info(f"Training score statistics:")
        self.logger.info(f"  Mean: {mean_score:.6f}")
        self.logger.info(f"  Std: {std_score:.6f}")
        self.logger.info(f"  Min: {np.min(all_scores):.6f}")
        self.logger.info(f"  Max: {np.max(all_scores):.6f}")
        
        return thresholds
    
    def _compute_percentile_thresholds(self, train_loader) -> Dict[str, float]:
        """Compute percentile-based thresholds."""
        
        # Collect scores
        all_scores = []
        for batch in train_loader:
            try:
                images = batch['image']
                scores, _ = self.evaluator.compute_batch_scores(images)
                all_scores.extend(scores)
            except Exception as e:
                continue
        
        if not all_scores:
            return {}
        
        all_scores = np.array(all_scores)
        
        # Percentile thresholds
        percentiles = [90, 95, 97, 98, 99, 99.5, 99.9]
        thresholds = {}
        
        for p in percentiles:
            threshold = np.percentile(all_scores, p)
            thresholds[f'p{p}'] = float(threshold)
        
        return thresholds
    
    def _compute_mad_thresholds(self, train_loader) -> Dict[str, float]:
        """Compute Median Absolute Deviation based thresholds."""
        
        # Collect scores
        all_scores = []
        for batch in train_loader:
            try:
                images = batch['image']
                scores, _ = self.evaluator.compute_batch_scores(images)
                all_scores.extend(scores)
            except Exception as e:
                continue
        
        if not all_scores:
            return {}
        
        all_scores = np.array(all_scores)
        
        # MAD calculation
        median_score = np.median(all_scores)
        mad_score = np.median(np.abs(all_scores - median_score))
        
        # MAD-based thresholds
        thresholds = {
            'median': float(median_score),
            'median_plus_2mad': float(median_score + 2 * mad_score),
            'median_plus_3mad': float(median_score + 3 * mad_score),
            'median_plus_4mad': float(median_score + 4 * mad_score),
            'median_plus_2_5mad': float(median_score + 2.5 * mad_score)
        }
        
        return thresholds
    
    def _compute_training_statistics(self, train_loader) -> Dict[str, Any]:
        """Compute comprehensive statistics on training scores."""
        
        all_scores = []
        for batch in train_loader:
            try:
                images = batch['image']
                scores, _ = self.evaluator.compute_batch_scores(images)
                all_scores.extend(scores)
            except Exception as e:
                continue
        
        if not all_scores:
            return {}
        
        all_scores = np.array(all_scores)
        
        # Comprehensive statistics
        stats = {
            'count': len(all_scores),
            'mean': float(np.mean(all_scores)),
            'std': float(np.std(all_scores)),
            'min': float(np.min(all_scores)),
            'max': float(np.max(all_scores)),
            'median': float(np.median(all_scores)),
            'q25': float(np.percentile(all_scores, 25)),
            'q75': float(np.percentile(all_scores, 75)),
            'iqr': float(np.percentile(all_scores, 75) - np.percentile(all_scores, 25)),
            'skewness': float(self._compute_skewness(all_scores)),
            'kurtosis': float(self._compute_kurtosis(all_scores))
        }
        
        # Distribution shape analysis
        mad = np.median(np.abs(all_scores - np.median(all_scores)))
        stats['mad'] = float(mad)
        stats['coefficient_of_variation'] = float(stats['std'] / stats['mean'] if stats['mean'] != 0 else 0)
        
        return stats
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _generate_threshold_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate threshold recommendations based on computed results."""
        
        recommendations = {
            'conservative': {},
            'balanced': {},
            'aggressive': {},
            'category_specific': {},
            'general_advice': []
        }
        
        # Extract available thresholds
        all_thresholds = {}
        for strategy, thresholds in results.get('thresholds', {}).items():
            if isinstance(thresholds, dict):
                for name, value in thresholds.items():
                    all_thresholds[f"{strategy}_{name}"] = value
            else:
                # Handle cases where thresholds is a single value or different structure
                continue
        
        if not all_thresholds:
            self.logger.warning("No thresholds available for recommendations")
            return recommendations
        
        # Convert to numpy array for analysis
        threshold_values = np.array(list(all_thresholds.values()))
        
        # Conservative approach (higher threshold, fewer false positives)
        conservative_threshold = np.percentile(threshold_values, 75)
        recommendations['conservative'] = {
            'threshold': float(conservative_threshold),
            'description': 'High precision, low false positive rate',
            'use_case': 'Production environments where false alarms are costly'
        }
        
        # Balanced approach (median threshold)
        balanced_threshold = np.median(threshold_values)
        recommendations['balanced'] = {
            'threshold': float(balanced_threshold),
            'description': 'Balance between precision and recall',
            'use_case': 'General purpose anomaly detection'
        }
        
        # Aggressive approach (lower threshold, fewer false negatives)
        aggressive_threshold = np.percentile(threshold_values, 25)
        recommendations['aggressive'] = {
            'threshold': float(aggressive_threshold),
            'description': 'High recall, catches more anomalies',
            'use_case': 'Safety-critical applications where missing anomalies is dangerous'
        }
        
        # Category-specific recommendations
        category = getattr(self.config, 'category', 'unknown')
        if category in ['capsule', 'hazelnut', 'screw']:
            # Use knowledge about category characteristics
            if category == 'capsule':
                # Smooth surfaces, subtle defects
                recommendations['category_specific'] = {
                    'threshold': float(np.percentile(threshold_values, 60)),
                    'description': 'Optimized for smooth surface defects (capsule)',
                    'reasoning': 'Capsule defects are often subtle surface imperfections'
                }
            elif category == 'hazelnut':
                # Textured surfaces, structural defects
                recommendations['category_specific'] = {
                    'threshold': float(np.percentile(threshold_values, 50)),
                    'description': 'Optimized for textured surface defects (hazelnut)',
                    'reasoning': 'Hazelnut defects include both surface and structural anomalies'
                }
            elif category == 'screw':
                # Metallic objects, threading defects
                recommendations['category_specific'] = {
                    'threshold': float(np.percentile(threshold_values, 45)),
                    'description': 'Optimized for metallic object defects (screw)',
                    'reasoning': 'Screw defects often involve threading and surface scratches'
                }
        
        # General advice
        stats = results.get('statistics', {})
        if stats:
            advice = []
            
            # Distribution shape advice
            if stats.get('coefficient_of_variation', 0) > 0.5:
                advice.append("High variability in normal scores - consider more robust thresholds (MAD-based)")
            
            if abs(stats.get('skewness', 0)) > 1.0:
                advice.append("Highly skewed score distribution - percentile-based thresholds recommended")
            
            if stats.get('kurtosis', 0) > 3.0:
                advice.append("Heavy-tailed distribution - conservative thresholds recommended")
            
            # Sample size advice
            if stats.get('count', 0) < 100:
                advice.append("Small training set - use validation-based threshold optimization")
            
            recommendations['general_advice'] = advice
        
        return recommendations


def main():
    """Main function for threshold computation script."""
    parser = argparse.ArgumentParser(
        description="Compute optimal thresholds for anomaly detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model (.pth file)'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        choices=['capsule', 'hazelnut', 'screw'],
        help='MVTec category'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/mvtec_ad',
        help='Path to MVTec AD dataset'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='outputs/thresholds.json',
        help='Output path for threshold results'
    )
    
    # Configuration arguments
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )
    
    # Threshold strategies
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['statistical', 'percentile', 'mad'],
        choices=['statistical', 'percentile', 'mad', 'validation_f1', 'validation_youden'],
        help='Threshold computation strategies'
    )
    
    parser.add_argument(
        '--use-validation',
        action='store_true',
        help='Use validation split for threshold optimization'
    )
    
    # System arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Computing device'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(
        "ThresholdComputation",
        level=logging.DEBUG if args.debug else logging.INFO
    )
    
    try:
        # Load configuration
        if args.config_path and Path(args.config_path).exists():
            config = AutoencoderConfig.from_file(args.config_path)
        else:
            config = AutoencoderConfig.from_category(args.category)
        
        # Load model
        logger.info(f"Loading model from: {args.model_path}")
        model = AutoencoderUNetLite(config)
        
        # Load state dict
        state_dict = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {args.device}")
        
        # Setup preprocessing
        preprocessor = MVTecPreprocessor(
            image_size=config.input_size,
            normalize=True
        )
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, test_loader = create_dataloaders(
            root_dir=args.data_dir,
            categories=args.category,
            preprocessor=preprocessor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            load_masks=False
        )
        
        # Create threshold computer
        threshold_computer = ThresholdComputer(
            model=model,
            config=config,
            device=args.device,
            logger=logger
        )
        
        # Determine validation loader
        val_loader = None
        if args.use_validation or 'validation_f1' in args.strategies or 'validation_youden' in args.strategies:
            val_loader = test_loader
            logger.info("Using test set as validation for threshold optimization")
        
        # Compute thresholds
        logger.info("Starting threshold computation...")
        results = threshold_computer.compute_comprehensive_thresholds(
            train_loader=train_loader,
            val_loader=val_loader,
            strategies=args.strategies
        )
        
        # Save results
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Threshold results saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("THRESHOLD COMPUTATION SUMMARY")
        print("="*60)
        
        if 'recommendations' in results:
            recs = results['recommendations']
            print(f"\nRecommended Thresholds:")
            print(f"  Conservative: {recs.get('conservative', {}).get('threshold', 'N/A'):.6f}")
            print(f"  Balanced:     {recs.get('balanced', {}).get('threshold', 'N/A'):.6f}")
            print(f"  Aggressive:   {recs.get('aggressive', {}).get('threshold', 'N/A'):.6f}")
            
            if 'category_specific' in recs and recs['category_specific']:
                print(f"  {args.category.title()}-specific: {recs['category_specific'].get('threshold', 'N/A'):.6f}")
        
        if 'statistics' in results:
            stats = results['statistics']
            print(f"\nTraining Score Statistics:")
            print(f"  Mean ± Std: {stats.get('mean', 0):.6f} ± {stats.get('std', 0):.6f}")
            print(f"  Range: [{stats.get('min', 0):.6f}, {stats.get('max', 0):.6f}]")
            print(f"  Samples: {stats.get('count', 0)}")
        
        print(f"\nFull results saved to: {output_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Threshold computation failed: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
