#!/usr/bin/env python3
"""
Comprehensive model evaluation script.
Evaluates trained anomaly detection models with detailed metrics and visualizations.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

# Import core components
from src.core.model import AutoencoderUNetLite
from src.core.model_config import AutoencoderConfig
from src.data.loaders import create_dataloaders
from src.data.preprocessing import MVTecPreprocessor
from src.evaluation.evaluator import AnomalyEvaluator, evaluate_model
from src.compute_thresholds_advanced import ThresholdComputer
from src.utils.logging_utils import setup_logger


def create_evaluation_plots(results: Dict[str, Any], 
                           output_dir: Path,
                           category: str) -> None:
    """
    Create comprehensive evaluation plots.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots
        category: Category name for titles
    """
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curve
    if 'roc_curve' in results:
        plt.figure(figsize=(8, 6))
        roc_data = results['roc_curve']
        fpr = np.array(roc_data['fpr'])
        tpr = np.array(roc_data['tpr'])
        
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {results.get("auroc_image", 0):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {category.title()} Category')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Precision-Recall Curve
    if 'pr_curve' in results:
        plt.figure(figsize=(8, 6))
        pr_data = results['pr_curve']
        precision = np.array(pr_data['precision'])
        recall = np.array(pr_data['recall'])
        
        plt.plot(recall, precision, linewidth=2, 
                label=f'PR Curve (AP = {results.get("ap_image", 0):.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {category.title()} Category')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Threshold Performance Comparison
    if 'threshold_metrics' in results:
        threshold_data = results['threshold_metrics']
        
        # Extract metrics for plotting
        thresholds = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        names = []
        
        for name, metrics in threshold_data.items():
            names.append(name)
            thresholds.append(metrics['threshold'])
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_score'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Threshold Performance Analysis - {category.title()}', fontsize=16)
        
        # Accuracy
        axes[0, 0].bar(names, accuracies, alpha=0.7)
        axes[0, 0].set_title('Accuracy by Threshold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score
        axes[0, 1].bar(names, f1_scores, alpha=0.7, color='orange')
        axes[0, 1].set_title('F1 Score by Threshold')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        axes[1, 0].scatter(recalls, precisions, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1, 0].annotate(name, (recalls[i], precisions[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Threshold values
        axes[1, 1].bar(names, thresholds, alpha=0.7, color='green')
        axes[1, 1].set_title('Threshold Values')
        axes[1, 1].set_ylabel('Threshold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Score Distribution Analysis
    if 'score_statistics' in results:
        stats = results['score_statistics']
        
        plt.figure(figsize=(10, 6))
        
        # Create box plot data
        labels = ['Normal', 'Anomaly']
        normal_data = [stats['normal_mean']]
        anomaly_data = [stats['anomaly_mean']] if stats['anomaly_mean'] > 0 else [0]
        
        # Bar plot with error bars
        x_pos = np.arange(len(labels))
        means = [stats['normal_mean'], stats['anomaly_mean']]
        errors = [stats['normal_std'], stats['anomaly_std']]
        
        plt.bar(x_pos, means, yerr=errors, alpha=0.7, capsize=5)
        plt.xlabel('Sample Type')
        plt.ylabel('Anomaly Score')
        plt.title(f'Score Distribution Comparison - {category.title()}')
        plt.xticks(x_pos, labels)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for i, (mean, std) in enumerate(zip(means, errors)):
            plt.text(i, mean + std + 0.01 * max(means), 
                    f'{mean:.4f}±{std:.4f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_detailed_report(results: Dict[str, Any], 
                           threshold_results: Optional[Dict[str, Any]],
                           category: str,
                           model_path: str) -> str:
    """Generate a detailed evaluation report."""
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 100)
    report_lines.append("COMPREHENSIVE ANOMALY DETECTION EVALUATION REPORT")
    report_lines.append("=" * 100)
    report_lines.append(f"Category: {category.upper()}")
    report_lines.append(f"Model: {model_path}")
    report_lines.append(f"Evaluation Time: {results.get('dataset_info', {}).get('evaluation_time', 'Unknown')}")
    report_lines.append("")
    
    # Dataset Summary
    if 'dataset_info' in results:
        info = results['dataset_info']
        report_lines.append("DATASET INFORMATION")
        report_lines.append("-" * 50)
        report_lines.append(f"Total Samples: {info.get('total_samples', 0):,}")
        report_lines.append(f"Normal Samples: {info.get('normal_samples', 0):,}")
        report_lines.append(f"Anomaly Samples: {info.get('anomaly_samples', 0):,}")
        if info.get('anomaly_samples', 0) > 0:
            anomaly_rate = info['anomaly_samples'] / info['total_samples'] * 100
            report_lines.append(f"Anomaly Rate: {anomaly_rate:.1f}%")
        report_lines.append(f"Categories: {', '.join(info.get('categories', []))}")
        report_lines.append("")
    
    # Performance Overview
    report_lines.append("PERFORMANCE OVERVIEW")
    report_lines.append("-" * 50)
    report_lines.append(f"Image-Level AUROC: {results.get('auroc_image', 0):.4f}")
    report_lines.append(f"Image-Level AP: {results.get('ap_image', 0):.4f}")
    
    if 'auroc_pixel' in results:
        report_lines.append(f"Pixel-Level AUROC: {results['auroc_pixel']:.4f}")
        report_lines.append(f"Pixel-Level AP: {results.get('ap_pixel', 0):.4f}")
    
    report_lines.append("")
    
    # Performance Interpretation
    report_lines.append("PERFORMANCE INTERPRETATION")
    report_lines.append("-" * 50)
    
    auroc = results.get('auroc_image', 0)
    if auroc >= 0.9:
        performance_level = "Excellent"
    elif auroc >= 0.8:
        performance_level = "Good"
    elif auroc >= 0.7:
        performance_level = "Fair"
    else:
        performance_level = "Poor"
    
    report_lines.append(f"Overall Performance: {performance_level} (AUROC = {auroc:.4f})")
    
    if auroc >= 0.9:
        report_lines.append("✓ Model shows excellent discrimination between normal and anomalous samples")
    elif auroc >= 0.8:
        report_lines.append("✓ Model shows good performance with some room for improvement")
    elif auroc >= 0.7:
        report_lines.append("⚠ Model shows fair performance - consider hyperparameter tuning")
    else:
        report_lines.append("✗ Model shows poor performance - significant improvements needed")
    
    report_lines.append("")
    
    # Threshold Analysis
    if 'threshold_metrics' in results:
        report_lines.append("THRESHOLD PERFORMANCE ANALYSIS")
        report_lines.append("-" * 50)
        
        best_f1 = 0
        best_f1_threshold = None
        best_acc = 0
        best_acc_threshold = None
        
        for name, metrics in results['threshold_metrics'].items():
            f1 = metrics.get('f1_score', 0)
            acc = metrics.get('accuracy', 0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = name
            
            if acc > best_acc:
                best_acc = acc
                best_acc_threshold = name
            
            report_lines.append(f"{name.upper()}:")
            report_lines.append(f"  Threshold: {metrics.get('threshold', 0):.6f}")
            report_lines.append(f"  Accuracy: {acc:.4f}")
            report_lines.append(f"  F1 Score: {f1:.4f}")
            report_lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
            report_lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
            report_lines.append("")
        
        report_lines.append(f"Best F1 Score: {best_f1:.4f} ({best_f1_threshold})")
        report_lines.append(f"Best Accuracy: {best_acc:.4f} ({best_acc_threshold})")
        report_lines.append("")
    
    # Threshold Recommendations
    if threshold_results and 'recommendations' in threshold_results:
        recs = threshold_results['recommendations']
        report_lines.append("THRESHOLD RECOMMENDATIONS")
        report_lines.append("-" * 50)
        
        for approach in ['conservative', 'balanced', 'aggressive']:
            if approach in recs and recs[approach]:
                rec = recs[approach]
                report_lines.append(f"{approach.upper()}:")
                report_lines.append(f"  Threshold: {rec.get('threshold', 0):.6f}")
                report_lines.append(f"  Description: {rec.get('description', 'N/A')}")
                report_lines.append(f"  Use Case: {rec.get('use_case', 'N/A')}")
                report_lines.append("")
        
        if 'category_specific' in recs and recs['category_specific']:
            cs_rec = recs['category_specific']
            report_lines.append("CATEGORY-SPECIFIC RECOMMENDATION:")
            report_lines.append(f"  Threshold: {cs_rec.get('threshold', 0):.6f}")
            report_lines.append(f"  Description: {cs_rec.get('description', 'N/A')}")
            report_lines.append(f"  Reasoning: {cs_rec.get('reasoning', 'N/A')}")
            report_lines.append("")
        
        if 'general_advice' in recs and recs['general_advice']:
            report_lines.append("GENERAL ADVICE:")
            for advice in recs['general_advice']:
                report_lines.append(f"  • {advice}")
            report_lines.append("")
    
    # Score Statistics
    if 'score_statistics' in results:
        stats = results['score_statistics']
        report_lines.append("SCORE DISTRIBUTION ANALYSIS")
        report_lines.append("-" * 50)
        
        report_lines.append("Normal Samples:")
        report_lines.append(f"  Mean ± Std: {stats['normal_mean']:.6f} ± {stats['normal_std']:.6f}")
        report_lines.append(f"  Range: [{stats['normal_min']:.6f}, {stats['normal_max']:.6f}]")
        
        if stats['anomaly_mean'] > 0:
            report_lines.append("Anomaly Samples:")
            report_lines.append(f"  Mean ± Std: {stats['anomaly_mean']:.6f} ± {stats['anomaly_std']:.6f}")
            report_lines.append(f"  Range: [{stats['anomaly_min']:.6f}, {stats['anomaly_max']:.6f}]")
            
            # Separation analysis
            separation = stats['anomaly_mean'] - stats['normal_mean']
            combined_std = np.sqrt(stats['normal_std']**2 + stats['anomaly_std']**2)
            separation_ratio = separation / combined_std if combined_std > 0 else 0
            
            report_lines.append(f"Score Separation: {separation:.6f}")
            report_lines.append(f"Separation Ratio: {separation_ratio:.2f}")
            
            if separation_ratio > 2:
                report_lines.append("✓ Excellent score separation between normal and anomaly classes")
            elif separation_ratio > 1:
                report_lines.append("✓ Good score separation")
            else:
                report_lines.append("⚠ Limited score separation - may indicate model issues")
        
        report_lines.append("")
    
    # Training Statistics (if available)
    if threshold_results and 'statistics' in threshold_results:
        train_stats = threshold_results['statistics']
        report_lines.append("TRAINING DATA STATISTICS")
        report_lines.append("-" * 50)
        report_lines.append(f"Training Samples: {train_stats.get('count', 0):,}")
        report_lines.append(f"Score Mean ± Std: {train_stats.get('mean', 0):.6f} ± {train_stats.get('std', 0):.6f}")
        report_lines.append(f"Score Range: [{train_stats.get('min', 0):.6f}, {train_stats.get('max', 0):.6f}]")
        report_lines.append(f"Distribution Shape:")
        report_lines.append(f"  Skewness: {train_stats.get('skewness', 0):.3f}")
        report_lines.append(f"  Kurtosis: {train_stats.get('kurtosis', 0):.3f}")
        report_lines.append(f"  Coefficient of Variation: {train_stats.get('coefficient_of_variation', 0):.3f}")
        report_lines.append("")
    
    # Conclusions and Recommendations
    report_lines.append("CONCLUSIONS AND RECOMMENDATIONS")
    report_lines.append("-" * 50)
    
    # Performance-based recommendations
    if auroc >= 0.9:
        report_lines.append("✓ Model is ready for production deployment")
        report_lines.append("✓ Consider fine-tuning thresholds for specific use cases")
    elif auroc >= 0.8:
        report_lines.append("✓ Model shows good performance")
        report_lines.append("• Consider additional training data or hyperparameter optimization")
        report_lines.append("• Monitor performance on new data")
    else:
        report_lines.append("⚠ Model requires significant improvements")
        report_lines.append("• Investigate data quality and preprocessing")
        report_lines.append("• Consider different model architectures or loss functions")
        report_lines.append("• Increase training data quantity and diversity")
    
    report_lines.append("")
    report_lines.append("=" * 100)
    
    return "\n".join(report_lines)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model'
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
        help='Path to dataset'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Output directory'
    )
    
    # Configuration
    parser.add_argument(
        '--thresholds-path',
        type=str,
        default=None,
        help='Path to pre-computed thresholds JSON'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of workers'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Computing device'
    )
    
    # Options
    parser.add_argument(
        '--compute-thresholds',
        action='store_true',
        help='Compute optimal thresholds'
    )
    
    parser.add_argument(
        '--create-plots',
        action='store_true',
        default=True,
        help='Create evaluation plots'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save individual predictions'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("ModelEvaluation", level=logging.INFO)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration and model
        logger.info(f"Loading model from: {args.model_path}")
        config = AutoencoderConfig.from_category(args.category)
        model = AutoencoderUNetLite(config)
        
        state_dict = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state_dict)
        model.to(args.device)
        
        # Setup data pipeline
        preprocessor = MVTecPreprocessor(
            image_size=config.input_size,
            normalize=True
        )
        
        train_loader, test_loader = create_dataloaders(
            root_dir=args.data_dir,
            categories=args.category,
            preprocessor=preprocessor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            load_masks=True
        )
        
        # Load or compute thresholds
        thresholds = None
        threshold_results = None
        
        if args.thresholds_path and Path(args.thresholds_path).exists():
            logger.info(f"Loading thresholds from: {args.thresholds_path}")
            with open(args.thresholds_path, 'r') as f:
                threshold_data = json.load(f)
                if 'thresholds' in threshold_data:
                    # Extract flat threshold dictionary
                    thresholds = {}
                    for strategy, thresh_dict in threshold_data['thresholds'].items():
                        if isinstance(thresh_dict, dict):
                            for name, value in thresh_dict.items():
                                thresholds[f"{strategy}_{name}"] = value
                threshold_results = threshold_data
        
        elif args.compute_thresholds:
            logger.info("Computing optimal thresholds...")
            threshold_computer = ThresholdComputer(
                model=model,
                config=config,  
                device=args.device,
                logger=logger
            )
            
            threshold_results = threshold_computer.compute_comprehensive_thresholds(
                train_loader=train_loader,
                val_loader=test_loader,
                strategies=['statistical', 'percentile', 'mad', 'validation_f1']
            )
            
            # Extract thresholds for evaluation
            thresholds = {}
            if 'thresholds' in threshold_results:
                for strategy, thresh_dict in threshold_results['thresholds'].items():
                    if isinstance(thresh_dict, dict):
                        for name, value in thresh_dict.items():
                            thresholds[f"{strategy}_{name}"] = value
            
            # Save threshold results
            threshold_path = output_dir / 'computed_thresholds.json'
            with open(threshold_path, 'w') as f:
                json.dump(threshold_results, f, indent=2)
            logger.info(f"Thresholds saved to: {threshold_path}")
        
        # Perform evaluation
        logger.info("Starting model evaluation...")
        results = evaluate_model(
            model=model,
            dataloader=test_loader,
            thresholds=thresholds,
            device=args.device
        )
        
        # Save evaluation results
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # Generate detailed report
        report = generate_detailed_report(
            results=results,
            threshold_results=threshold_results,
            category=args.category,
            model_path=args.model_path
        )
        
        # Save report
        report_path = output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Create visualizations
        if args.create_plots:
            logger.info("Creating evaluation plots...")
            try:
                create_evaluation_plots(results, output_dir, args.category)
                logger.info("Plots created successfully")
            except Exception as e:
                logger.warning(f"Failed to create plots: {e}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Category: {args.category}")
        print(f"Image-Level AUROC: {results.get('auroc_image', 0):.4f}")
        if 'auroc_pixel' in results:
            print(f"Pixel-Level AUROC: {results['auroc_pixel']:.4f}")
        
        if threshold_results and 'recommendations' in threshold_results:
            recs = threshold_results['recommendations']
            print(f"\nRecommended Thresholds:")
            for approach in ['conservative', 'balanced', 'aggressive']:
                if approach in recs and recs[approach]:
                    threshold = recs[approach].get('threshold', 0)
                    print(f"  {approach.title()}: {threshold:.6f}")
        
        print(f"\nResults saved to: {output_dir}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
