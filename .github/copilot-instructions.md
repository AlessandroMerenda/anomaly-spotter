# PROJECT CONTEXT - Anomaly Spotter

> **Complete Architecture and Implementation Reference**  
> Generated on 2025-08-09 by Claude Code Analysis

---

## 🎯 PROJECT OVERVIEW

**Anomaly Spotter** is an advanced PyTorch-based anomaly detection system designed for industrial defect detection using autoencoder U-Net architectures. The project implements state-of-the-art computer vision techniques for detecting anomalies in manufacturing processes, specifically targeting the MVTec AD dataset.

### Key Features
- **Modern U-Net Autoencoder**: Custom AutoencoderUNetLite with skip connections
- **Advanced Loss Functions**: Combined MSE, L1, SSIM, perceptual, and edge preservation losses
- **Multi-category Support**: Handles multiple defect categories (capsule, hazelnut, screw, etc.)
- **Production-Ready**: Comprehensive testing, error handling, and monitoring
- **Containerization Support**: Docker-ready with CI/CD pipeline integration
- **Airflow Orchestration**: Planned workflow management for enterprise deployment

---

## 🏗️ ARCHITECTURE OVERVIEW

### Core Components

```
anomaly-spotter/
├── src/                          # Core source code
│   ├── core/                     # Core ML components
│   │   ├── model.py              # AutoencoderUNetLite implementation
│   │   ├── losses.py             # Advanced loss functions
│   │   └── model_config.py       # Configuration management
│   ├── data/                     # Data handling pipeline
│   │   ├── loaders.py            # MVTec dataset loading
│   │   └── preprocessing.py      # Image preprocessing pipeline
│   ├── training/                 # Training infrastructure
│   │   └── trainer.py            # Advanced training pipeline
│   └── utils/                    # Utilities and helpers
│       └── logging_utils.py      # Structured logging system
├── tests/                        # Comprehensive test suite
├── notebooks/                    # Jupyter notebooks for validation
├── requirements/                 # Dependency management
├── config/                       # Configuration files
└── outputs/                      # Training outputs and models
```

### Data Flow Architecture

```
Raw Images (MVTec) 
    ↓
Preprocessing Pipeline (Albumentations)
    ↓
DataLoader (PyTorch)
    ↓
AutoencoderUNetLite Model
    ↓
Combined Loss Functions
    ↓
Training Loop with AMP
    ↓
Anomaly Detection & Thresholding
```

---

## 🧠 MODEL ARCHITECTURE

### AutoencoderUNetLite (`src/core/model.py`)

**Architecture**: U-Net style autoencoder with skip connections
- **Input**: RGB images (3, 128, 128) normalized to [-1, 1]
- **Output**: Reconstructed images (3, 128, 128) via Tanh activation
- **Encoder**: 3 downsampling blocks (128→64→32→16)
- **Bottleneck**: 512 channels at 16x16 resolution
- **Decoder**: 3 upsampling blocks with skip connections
- **Parameters**: ~1.9M trainable parameters

### Key Features
- **Skip Connections**: Preserve spatial details during reconstruction
- **Double Convolutions**: Enhanced feature extraction per block
- **ConvTranspose2d**: High-quality upsampling
- **Input Validation**: Comprehensive shape and channel validation
- **Range Guarantee**: Output guaranteed in [-1, 1] range

---

## 🔬 LOSS FUNCTIONS (`src/core/losses.py`)

### CombinedLoss Architecture
Advanced multi-component loss system optimized for anomaly detection:

#### Components
1. **Reconstruction Losses**
   - MSE Loss: L2 pixel-wise reconstruction error
   - L1 Loss: L1 pixel-wise reconstruction error
   - Focal Loss: Addresses class imbalance

2. **Structural Similarity (SSIM)**
   - PyTorch-MSSSIM integration with fallback
   - Custom SSIM implementation following original paper
   - Robust error handling and numerical stability

3. **Perceptual Loss**
   - VGG16-based feature extraction
   - ImageNet pretrained features (relu3_3)
   - Automatic normalization handling

4. **Edge Preservation Loss**
   - Sobel operator-based edge detection
   - Maintains fine structural details

### Loss Configuration
```python
# Example configuration
loss_weights = {
    'reconstruction': 1.0,
    'ssim': 0.5,
    'perceptual': 0.1,
    'edge': 0.0
}
```

---

## 📊 DATA PIPELINE (`src/data/`)

### MVTecDataset (`src/data/loaders.py`)
Comprehensive dataset implementation supporting:
- **Multi-category Loading**: Single or multiple defect categories
- **Automatic Labeling**: Normal (0) vs Anomaly (1)
- **Ground Truth Masks**: Optional mask loading for evaluation
- **Robust Error Handling**: Detailed error reporting with context
- **Memory Efficient**: Lazy loading with proper resource management

### MVTecPreprocessor (`src/data/preprocessing.py`)
Advanced preprocessing pipeline featuring:
- **Albumentations Integration**: State-of-the-art augmentations
- **Normalization Options**: ImageNet or Tanh [-1, 1] normalization
- **Fallback Support**: Torchvision transforms as backup
- **Training/Test Splits**: Separate augmentation strategies

#### Supported Augmentations
- Geometric: Rotation, flips, elastic transforms
- Photometric: Brightness, contrast, Gaussian noise
- Advanced: Grid distortion, optical distortion

---

## 🚂 TRAINING SYSTEM (`src/training/trainer.py`)

### AnomalyDetectorTrainer
Production-grade training infrastructure with:

#### Core Features
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support
- **Advanced Scheduling**: Cosine, plateau, and step schedulers
- **Robust Checkpointing**: Best model and latest checkpoint saving
- **Tensorboard Integration**: Real-time metrics and image logging
- **Early Stopping**: Configurable patience and monitoring

#### Optimization Support
- **Optimizers**: Adam, AdamW, SGD with configurable parameters
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Multiple strategies available
- **Threshold Calculation**: Automatic anomaly detection thresholds

#### Monitoring & Logging
- **Real-time Metrics**: Loss tracking and learning rate monitoring
- **Image Visualization**: Original/reconstructed image comparisons
- **Training History**: Comprehensive training statistics
- **Resource Monitoring**: GPU/CPU usage tracking

---

## ⚙️ CONFIGURATION SYSTEM (`src/core/model_config.py`)

### AutoencoderConfig
Dataclass-based configuration management supporting:

#### Architecture Configuration
- Model dimensions and channels
- Encoder/decoder depths
- Dropout and attention settings

#### Training Configuration
- Learning rates and scheduling
- Batch sizes and epochs
- Optimization parameters

#### Loss Configuration
- Multi-component loss weights
- SSIM and perceptual loss settings
- Edge preservation parameters

#### Category-Specific Presets
```python
# Predefined configurations
config = AutoencoderConfig.from_category('capsule')
config = AutoencoderConfig.from_category('hazelnut')
config = AutoencoderConfig.from_category('screw')
config = AutoencoderConfig.from_category('multi')
```

---

## 🔧 UTILITIES & INFRASTRUCTURE

### Logging System (`src/utils/logging_utils.py`)
Structured logging with:
- **Colored Console Output**: Enhanced readability
- **File Logging**: Persistent log files with timestamps
- **Error Handling**: Custom exception hierarchy
- **Resource Monitoring**: System resource checking

### Custom Exceptions
- `AnomalySpotterError`: Base project exception
- `ModelError`: Model-related errors
- `DataError`: Data loading/processing errors
- `ConfigError`: Configuration errors
- `ResourceError`: Resource availability errors

---

## 🧪 TESTING FRAMEWORK

### Comprehensive Test Suite (`tests/`)
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Fixtures**: Reusable test components and mock data
- **Coverage**: 80%+ code coverage target
- **CI/CD Ready**: Automated testing pipeline

#### Test Structure
```
tests/
├── conftest.py                   # Shared fixtures
├── unit/                        # Unit tests
│   ├── test_model.py            # Model architecture tests
│   ├── test_losses.py           # Loss function tests
│   └── test_preprocessing.py    # Data preprocessing tests
└── integration/                 # Integration tests
    └── test_training_pipeline.py # End-to-end training tests
```

---

## 📋 WORKFLOW & ENTRY POINTS

### Training Pipeline
1. **Data Loading**: MVTec dataset with preprocessing
2. **Model Initialization**: AutoencoderUNetLite setup
3. **Training Loop**: AMP-enabled training with monitoring
4. **Threshold Calculation**: Automatic anomaly detection thresholds
5. **Model Saving**: Best model and checkpoint persistence

### Evaluation Pipeline
1. **Model Loading**: Trained model restoration
2. **Test Data Processing**: Batch evaluation on test set
3. **Anomaly Scoring**: Reconstruction error calculation
4. **Threshold Application**: Binary anomaly classification
5. **Metrics Calculation**: ROC, PR curves, F1 scores

### Makefile Commands
The project provides comprehensive automation via Makefile:

#### Training Commands
```bash
make train CATEGORY=capsule          # Train single category
make train-core                      # Train core categories
make train-parallel                  # Parallel training
make train-all                       # All categories
make resume CHECKPOINT=path          # Resume training
```

#### Evaluation Commands
```bash
make evaluate CATEGORY=capsule       # Evaluate model
make evaluate-all                    # Evaluate all models
make compute-thresholds             # Calculate thresholds
make test-model                     # Test on samples
```

#### Development Commands
```bash
make install-dev                    # Setup development environment
make test                          # Run test suite
make lint                         # Code quality checks
make clean                        # Cleanup temporary files
```

---

## 📦 DEPENDENCY MANAGEMENT

### Core Dependencies (`requirements/main.txt`)
- **PyTorch >= 2.1.0**: Deep learning framework
- **torchvision >= 0.16.0**: Computer vision utilities
- **albumentations >= 1.3.0**: Advanced image augmentations
- **opencv-python >= 4.8.0**: Image processing
- **scikit-image >= 0.22.0**: Scientific image processing
- **numpy >= 1.24.0**: Numerical computing
- **matplotlib >= 3.8.0**: Plotting and visualization
- **tqdm >= 4.66.0**: Progress bars
- **PyYAML >= 6.0.0**: Configuration file support

### Optional Dependencies
- **pytorch-msssim**: Advanced SSIM loss functions
- **wandb**: Experiment tracking
- **tensorboard**: Training visualization

---

## 🐳 CONTAINERIZATION & DEPLOYMENT

### Docker Support
- **Multi-stage Builds**: Optimized production images
- **GPU Support**: CUDA-enabled containers
- **Volume Mounts**: Data and output persistence
- **Environment Variables**: Configurable runtime parameters

### Planned Orchestration
- **Kubernetes**: Scalable deployment infrastructure
- **Airflow**: Workflow orchestration for training pipelines
- **CI/CD Integration**: Automated testing and deployment

---

## 📊 PERFORMANCE & MONITORING

### Training Performance
- **Mixed Precision**: 30-50% speedup with AMP
- **Batch Processing**: Optimized DataLoader with multiple workers
- **Memory Efficiency**: Gradient accumulation and checkpointing
- **GPU Utilization**: Optimized tensor operations

### Monitoring Capabilities
- **Tensorboard**: Real-time training metrics
- **System Resources**: CPU, GPU, memory tracking
- **Training History**: Comprehensive statistics logging
- **Error Reporting**: Detailed error context and tracebacks

---

## 🔍 ANOMALY DETECTION METHODOLOGY

### Reconstruction-Based Approach
The system employs reconstruction error as the primary anomaly indicator:

1. **Training Phase**: Model learns to reconstruct normal samples
2. **Inference Phase**: Reconstruction error indicates anomalies
3. **Threshold Calculation**: Multiple strategies for optimal thresholds
4. **Post-processing**: Optional Gaussian blur and morphological operations

### Threshold Strategies
- **Percentile-based**: 95th, 99th percentile of training errors
- **Statistical**: Mean + k*std deviation
- **Adaptive**: Dynamic threshold based on validation data
- **IQR-based**: Interquartile range outlier detection

---

## 🚨 ERROR HANDLING & ROBUSTNESS

### Data Pipeline Robustness
- **Graceful Degradation**: Fallback mechanisms for missing dependencies
- **Input Validation**: Comprehensive shape and type checking
- **Resource Management**: Proper cleanup and memory management
- **Error Context**: Detailed error reporting with remediation suggestions

### Training Robustness
- **Checkpoint Recovery**: Automatic training resumption
- **Numerical Stability**: Loss function stability checks
- **Resource Monitoring**: Out-of-memory protection
- **Gradient Monitoring**: Gradient explosion detection

---

## 📚 NOTEBOOKS & VALIDATION

### Jupyter Notebooks (`notebooks/`)
- **model_validation_comprehensive.ipynb**: Complete model validation
- **model_validation_WORKING.ipynb**: Environment-agnostic testing
- **Mock Implementations**: Fallback testing environments

### Validation Features
- **Multi-level Import Strategy**: Robust module importing
- **Environment Detection**: Automatic environment adaptation
- **Visual Validation**: Image reconstruction comparisons
- **Performance Metrics**: Comprehensive evaluation reports

---

## 🔄 DEVELOPMENT WORKFLOW

### Code Quality
- **Black Formatting**: Consistent code style
- **Flake8 Linting**: Code quality enforcement
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings and comments

### Testing Strategy
- **Test-Driven Development**: Tests written alongside code
- **Continuous Integration**: Automated testing on commits
- **Coverage Reporting**: Minimum 80% coverage requirement
- **Performance Testing**: Training and inference benchmarks

### Version Control
- **Git Hooks**: Pre-commit quality checks
- **Branch Strategy**: Feature branches with main integration
- **Release Management**: Tagged releases with changelogs

---

## 🎯 FUTURE ENHANCEMENTS

### Planned Features
1. **Advanced Architectures**: Vision Transformers, Diffusion Models
2. **Multi-Modal Support**: Text descriptions, sensor data integration
3. **Real-time Inference**: Edge deployment optimization
4. **Active Learning**: Human-in-the-loop annotation
5. **Explainable AI**: Attention visualization, gradient-based explanations

### Infrastructure Improvements
1. **Kubernetes Deployment**: Scalable cloud deployment
2. **MLOps Pipeline**: Automated model lifecycle management
3. **A/B Testing**: Model comparison framework
4. **Monitoring Dashboard**: Real-time production monitoring

---

## 📖 API REFERENCE

### Core Classes

#### AutoencoderUNetLite
```python
model = AutoencoderUNetLite(input_channels=3, output_channels=3)
output = model(input_tensor)  # [B, 3, 128, 128] -> [B, 3, 128, 128]
```

#### CombinedLoss
```python
loss_fn = CombinedLoss(config)
total_loss, loss_components = loss_fn(pred, target)
```

#### MVTecDataset
```python
dataset = MVTecDataset(root_dir, categories=['capsule'], split='train')
sample = dataset[0]  # {'image': tensor, 'label': int, 'mask': tensor}
```

#### AnomalyDetectorTrainer
```python
trainer = AnomalyDetectorTrainer(model, config, device='cuda')
results = trainer.train(train_loader, val_loader, save_dir)
```

---

## 🔧 TROUBLESHOOTING

### Common Issues
1. **Import Errors**: Use fallback import strategies in notebooks
2. **Memory Issues**: Reduce batch size or enable gradient checkpointing
3. **CUDA Errors**: Verify GPU availability and driver compatibility
4. **Training Instability**: Check learning rate and gradient clipping

### Debug Commands
```bash
make train-debug CATEGORY=capsule    # Quick debug training
make test-fast                       # Fast test execution
make clean                          # Clear temporary files
make status                         # Check project status
```

---

## 📄 LICENSE & USAGE

This project is designed for educational and research purposes, demonstrating modern anomaly detection techniques in industrial settings. The codebase follows best practices for production ML systems while maintaining flexibility for experimentation and extension.

---

*This document provides a comprehensive overview of the Anomaly Spotter project architecture. For specific implementation details, refer to the individual module documentation and test cases.*