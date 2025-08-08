# Anomaly Detection Training Pipeline

This directory contains the complete training pipeline for the anomaly detection system using AutoencoderUNetLite on the MVTec AD dataset.

## 🚀 Quick Start

### 1. Setup and Validation
```bash
# Run the quick setup script to validate everything
./scripts/quick_setup.sh

# Or check manually
make check-data
make stats
```

### 2. Train Single Category
```bash
# Basic training
make train CATEGORY=capsule

# With custom parameters
make train CATEGORY=hazelnut EPOCHS=50 BATCH_SIZE=16

# Debug mode (fast test)
make train-debug CATEGORY=screw
```

### 3. Batch Training
```bash
# Train core categories sequentially
make train-core

# Train core categories in parallel (faster)
make train-parallel

# Train all MVTec categories
make train-all
```

### 4. Monitor Training
```bash
# Start Tensorboard
make monitor
# Then open http://localhost:6006
```

## 📁 Architecture Overview

```
src/
├── core/                      # Core model components
│   ├── model.py              # AutoencoderUNetLite architecture
│   ├── model_config.py       # Configuration management
│   └── losses.py             # Advanced loss functions
├── data/                      # Data handling
│   ├── loaders.py            # MVTec dataset and dataloaders
│   └── preprocessing.py      # Image preprocessing pipeline
├── training/                  # Training components
│   └── trainer.py            # Advanced trainer with AMP, scheduling
├── train_main.py             # Main training script
└── train_batch.py            # Batch training script
```

## ⚙️ Configuration

### Base Configuration
The system uses category-specific configurations defined in `src/core/model_config.py`:

- **Capsule**: Optimized for smooth surfaces, higher SSIM weight
- **Hazelnut**: Balanced for textured objects with structural defects  
- **Screw**: Metallic objects, higher perceptual loss weight

### Custom Configuration
Create custom YAML configs in `config/`:

```yaml
# config/custom_training.yaml
epochs: 150
batch_size: 24
learning_rate: 0.0005

loss_weights:
  mse: 1.0
  ssim: 0.4
  perceptual: 0.15
  edge: 0.08

enable_ssim_loss: true
use_amp: true
gradient_clip: 1.0
```

Use with:
```bash
make train CATEGORY=capsule CONFIG=config/custom_training.yaml
```

## 🎯 Training Features

### Advanced Trainer (`src/training/trainer.py`)
- **Mixed Precision Training** (AMP) for faster GPU training
- **Advanced Optimizers**: AdamW, Adam, SGD with proper scheduling
- **Smart Scheduling**: ReduceLROnPlateau, CosineAnnealing, StepLR
- **Early Stopping** with configurable patience
- **Comprehensive Logging**: Tensorboard with image visualization
- **Robust Checkpointing**: Best model tracking, resume capability

### Loss Functions (`src/core/losses.py`)
- **Combined Loss**: MSE + SSIM + Perceptual + Edge preservation
- **Custom SSIM**: Fallback implementation when pytorch_msssim unavailable
- **VGG Perceptual Loss**: Deep feature matching for realistic reconstructions
- **Edge Preservation**: Sobel operator-based edge loss
- **Configurable Weights**: Per-category loss weighting

### Data Pipeline (`src/data/`)
- **Advanced Preprocessing**: Albumentations-based augmentation
- **Multi-category Support**: Train on single or multiple categories
- **Ground Truth Masks**: Support for pixel-level anomaly masks
- **Robust Loading**: Fallback mechanisms (OpenCV → PIL)
- **Memory Efficient**: Pin memory, persistent workers

## 📊 Output Structure

Training produces comprehensive outputs:

```
outputs/
├── capsule_20250805_143022/          # Timestamped run directory
│   ├── model.pth                     # Production model (state dict)
│   ├── checkpoint_best.pth           # Best checkpoint (full state)
│   ├── checkpoint_latest.pth         # Latest checkpoint
│   ├── thresholds.json               # Anomaly detection thresholds
│   ├── training_history.json         # Complete training metrics
│   ├── training_summary.json         # Final results summary
│   ├── config.yaml                   # Training configuration used
│   ├── system_info.json              # System/environment info
│   └── tensorboard/                  # Tensorboard logs
└── batch_training_20250805_143500/   # Batch training results
    ├── batch_config.json             # Batch configuration
    ├── batch_results.json            # Detailed results per category
    └── individual_runs/               # Individual category outputs
```

## 🔧 Command Reference

### Training Commands
```bash
make train              # Train single category
make train-core         # Train capsule, hazelnut, screw
make train-parallel     # Train core categories in parallel
make train-all          # Train all 15 MVTec categories
make train-debug        # Debug training (2 epochs)
make train-quick        # Quick test (2 epochs, small batch)
make resume             # Resume from checkpoint
```

### Dataset Commands  
```bash
make check-data         # Validate dataset structure
make stats              # Show dataset statistics
```

### Utility Commands
```bash
make monitor            # Start Tensorboard
make config             # Show current configuration
make help-train         # Show training help
make clean              # Clean outputs
```

### Configuration Variables
```bash
CATEGORY=capsule        # Target category
BATCH_SIZE=32          # Batch size
EPOCHS=100             # Number of epochs
DATA_DIR=data/mvtec_ad # Dataset location
OUTPUT_DIR=outputs     # Output directory
CONFIG=config/file.yaml # Custom config
NUM_WORKERS=4          # Dataloader workers
```

### Examples
```bash
# Basic training
make train CATEGORY=hazelnut

# Custom configuration
make train CATEGORY=screw EPOCHS=75 BATCH_SIZE=24

# Resume training
make resume CHECKPOINT=outputs/capsule_123/checkpoint_latest.pth CATEGORY=capsule

# Parallel batch training
make train-parallel EPOCHS=50

# Debug mode
make train-debug CATEGORY=hazelnut
```

## 🎭 Advanced Usage

### Resume Training
```bash
# Find checkpoint
ls outputs/capsule_*/checkpoint_*.pth

# Resume from best checkpoint
make resume CHECKPOINT=outputs/capsule_20250805_143022/checkpoint_best.pth CATEGORY=capsule

# Resume from latest
make resume CHECKPOINT=outputs/capsule_20250805_143022/checkpoint_latest.pth CATEGORY=capsule
```

### Custom Loss Configuration
Create specialized configs for different defect types:

```yaml
# Surface defects (capsule)
loss_weights:
  mse: 1.0
  ssim: 0.6      # High SSIM for surface quality
  perceptual: 0.1
  edge: 0.02     # Low edge for smooth surfaces

# Structural defects (hazelnut)  
loss_weights:
  mse: 1.0
  ssim: 0.2
  perceptual: 0.15
  edge: 0.12     # High edge for structural integrity

# Metallic objects (screw)
loss_weights:
  mse: 1.0
  ssim: 0.3
  perceptual: 0.25  # High perceptual for metallic appearance
  edge: 0.08
```

### Batch Training Strategies
```bash
# Sequential (safe for limited GPU memory)
make train-core BATCH_SIZE=16

# Parallel (faster with multiple GPUs or high-end GPU)
make train-parallel BATCH_SIZE=32

# All categories with resource management
make train-all BATCH_SIZE=16 NUM_WORKERS=2
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   make train CATEGORY=capsule BATCH_SIZE=16
   
   # Reduce workers
   make train CATEGORY=capsule NUM_WORKERS=2
   
   # Disable mixed precision
   # Edit config: use_amp: false
   ```

2. **Import Errors**
   ```bash
   # Check environment
   ./scripts/quick_setup.sh
   
   # Verify Python environment
   which python3
   python3 -c "import torch; print(torch.__version__)"
   ```

3. **Dataset Not Found**
   ```bash
   # Check dataset structure
   make check-data
   
   # Verify paths
   ls -la data/mvtec_ad/capsule/train/good/
   ```

4. **Training Stalls**
   ```bash
   # Check system resources
   nvidia-smi  # GPU usage
   htop        # CPU/RAM usage
   
   # Reduce complexity
   make train-debug CATEGORY=capsule
   ```

### Performance Tuning

1. **GPU Optimization**
   - Use `use_amp: true` for mixed precision
   - Set `pin_memory: true` for faster GPU transfers
   - Optimize `num_workers` (typically 2-4 per GPU)

2. **Memory Management**
   - Adjust `batch_size` based on GPU memory
   - Use `gradient_clip` to prevent exploding gradients
   - Monitor with `nvidia-smi`

3. **Training Speed**
   - Enable `torch.backends.cudnn.benchmark = True`
   - Use `persistent_workers: true` for dataloaders
   - Balance `batch_size` vs `num_workers`

## 📈 Monitoring and Analysis

### Tensorboard Metrics
- **Training/Validation Loss**: Overall reconstruction quality
- **Loss Components**: MSE, SSIM, Perceptual, Edge contributions  
- **Learning Rate**: Optimizer scheduling
- **Reconstruction Images**: Visual quality assessment
- **Error Maps**: Pixel-level difference visualization

### Key Metrics to Watch
- **Validation Loss**: Should decrease steadily
- **Loss Components**: Should be balanced (no single component dominating)
- **SSIM Values**: Should increase (closer to 1.0)
- **Reconstruction Quality**: Visual similarity to inputs

### Success Indicators
- Smooth convergence without oscillation
- Balanced loss component contributions
- High-quality visual reconstructions
- Appropriate threshold values for anomaly detection

## 🎯 Next Steps

After successful training:

1. **Model Evaluation**: Use evaluation scripts to assess performance
2. **Threshold Tuning**: Optimize detection thresholds for your use case
3. **Production Deployment**: Use saved `model.pth` for inference
4. **Airflow Integration**: Set up automated retraining pipelines

The training pipeline is designed for production use and provides all necessary components for robust anomaly detection model development.
