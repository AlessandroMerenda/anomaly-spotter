#!/usr/bin/env python3
"""
Batch training script for multiple categories.
Allows training multiple MVTec categories in sequence or parallel.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
import logging

from src.utils.logging_utils import setup_logger


def run_single_training(category: str, args: argparse.Namespace) -> dict:
    """
    Run training for a single category.
    
    Args:
        category: MVTec category name
        args: Command line arguments
    
    Returns:
        Dictionary with training results
    """
    logger = logging.getLogger(__name__)
    
    # Build command
    cmd = [
        sys.executable, 
        str(Path(__file__).parent / 'train_main.py'),
        '--category', category,
        '--data-dir', args.data_dir,
        '--output-dir', args.output_dir,
        '--num-workers', str(args.num_workers)
    ]
    
    # Add optional arguments
    if args.config:
        cmd.extend(['--config', args.config])
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.learning_rate:
        cmd.extend(['--learning-rate', str(args.learning_rate)])
    if args.debug:
        cmd.append('--debug')
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting training for {category}...")
        
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout if hasattr(args, 'timeout') else None
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            logger.info(f"Training completed successfully for {category} ({duration:.1f}s)")
            return {
                'category': category,
                'status': 'success',
                'duration': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            logger.error(f"Training failed for {category}")
            logger.error(f"Error output: {result.stderr}")
            return {
                'category': category,
                'status': 'failed',
                'duration': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
    
    except subprocess.TimeoutExpired:
        logger.error(f"Training timeout for {category}")
        return {
            'category': category,
            'status': 'timeout',
            'duration': (datetime.now() - start_time).total_seconds(),
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Unexpected error training {category}: {e}")
        return {
            'category': category,
            'status': 'error',
            'duration': (datetime.now() - start_time).total_seconds(),
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'error': str(e)
        }


def main():
    """Main batch training function."""
    parser = argparse.ArgumentParser(
        description="Batch train anomaly detection models on multiple MVTec categories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Categories
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['capsule', 'hazelnut', 'screw'],
        help='Categories to train (space-separated)'
    )
    
    parser.add_argument(
        '--all-categories',
        action='store_true',
        help='Train on all available MVTec categories'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/mvtec_ad',
        help='Path to MVTec AD dataset'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Base output directory'
    )
    
    # Training parameters
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config YAML'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size override'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs override'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate override'
    )
    
    # System arguments
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of dataloader workers per job'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel training jobs (1 = sequential)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=7200,  # 2 hours
        help='Timeout per category in seconds'
    )
    
    # Debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("BatchTraining", level=logging.DEBUG if args.debug else logging.INFO)
    
    # Determine categories to train
    if args.all_categories:
        categories = ['capsule', 'hazelnut', 'screw', 'bottle', 'cable', 'carpet',
                     'grid', 'leather', 'metal_nut', 'pill', 'tile', 'toothbrush',
                     'transistor', 'wood', 'zipper']
    else:
        categories = args.categories
    
    logger.info("=" * 80)
    logger.info("BATCH TRAINING STARTED")
    logger.info("=" * 80)
    logger.info(f"Categories: {categories}")
    logger.info(f"Parallel jobs: {args.parallel}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create batch output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_output_dir = Path(args.output_dir) / f"batch_training_{timestamp}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save batch configuration
    batch_config = {
        'categories': categories,
        'parallel_jobs': args.parallel,
        'start_time': datetime.now().isoformat(),
        'arguments': vars(args)
    }
    
    with open(batch_output_dir / 'batch_config.json', 'w') as f:
        json.dump(batch_config, f, indent=2)
    
    # Execute training
    results = []
    
    if args.parallel == 1:
        # Sequential training
        logger.info("Running sequential training...")
        for i, category in enumerate(categories, 1):
            logger.info(f"\n[{i}/{len(categories)}] Training {category}...")
            result = run_single_training(category, args)
            results.append(result)
            
            # Log progress
            if result['status'] == 'success':
                logger.info(f"✓ {category} completed successfully")
            else:
                logger.error(f"✗ {category} failed: {result['status']}")
    
    else:
        # Parallel training
        logger.info(f"Running parallel training with {args.parallel} workers...")
        
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all jobs
            future_to_category = {
                executor.submit(run_single_training, category, args): category
                for category in categories
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        logger.info(f"✓ {category} completed successfully")
                    else:
                        logger.error(f"✗ {category} failed: {result['status']}")
                        
                except Exception as e:
                    logger.error(f"✗ {category} exception: {e}")
                    results.append({
                        'category': category,
                        'status': 'exception',
                        'error': str(e)
                    })
    
    # Generate summary
    end_time = datetime.now()
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total categories: {len(categories)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        logger.info(f"\nSuccessful categories:")
        for result in successful:
            logger.info(f"  ✓ {result['category']} ({result['duration']:.1f}s)")
    
    if failed:
        logger.info(f"\nFailed categories:")
        for result in failed:
            logger.info(f"  ✗ {result['category']} ({result['status']})")
    
    # Save detailed results
    batch_summary = {
        'batch_config': batch_config,
        'end_time': end_time.isoformat(),
        'total_duration': (end_time - datetime.fromisoformat(batch_config['start_time'])).total_seconds(),
        'results': results,
        'summary': {
            'total': len(categories),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(categories) * 100
        }
    }
    
    with open(batch_output_dir / 'batch_results.json', 'w') as f:
        json.dump(batch_summary, f, indent=2, default=str)
    
    logger.info(f"\nBatch results saved to: {batch_output_dir}")
    logger.info(f"Success rate: {batch_summary['summary']['success_rate']:.1f}%")
    
    # Exit with appropriate code
    if failed:
        logger.warning("Some training jobs failed!")
        sys.exit(1)
    else:
        logger.info("All training jobs completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
