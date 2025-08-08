#!/bin/bash
# Quick setup and test script for Anomaly Spotter training pipeline

set -e  # Exit on error

echo "🚀 Anomaly Spotter - Quick Setup & Test"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "src/train_main.py" ]]; then
    print_error "Not in anomaly-spotter directory. Please run from project root."
    exit 1
fi

print_status "Checking Python environment..."

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Python version: $PYTHON_VERSION"

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "No virtual environment active. Consider using 'conda activate anomaly-spotter' or similar."
else
    print_success "Virtual environment active: $VIRTUAL_ENV"
fi

# Check key dependencies
print_status "Checking dependencies..."

check_dependency() {
    if python3 -c "import $1" 2>/dev/null; then
        VERSION=$(python3 -c "import $1; print($1.__version__)" 2>/dev/null || echo "unknown")
        print_success "$1 available (version: $VERSION)"
        return 0
    else
        print_error "$1 not available"
        return 1
    fi
}

DEPS_OK=true

if ! check_dependency "torch"; then
    DEPS_OK=false
fi

if ! check_dependency "cv2"; then
    DEPS_OK=false
    print_warning "Try: pip install opencv-python"
fi

if ! check_dependency "numpy"; then
    DEPS_OK=false
fi

if ! check_dependency "albumentations"; then
    DEPS_OK=false
    print_warning "Try: pip install albumentations"
fi

if ! check_dependency "tqdm"; then
    DEPS_OK=false
    print_warning "Try: pip install tqdm"
fi

if [[ "$DEPS_OK" == false ]]; then
    print_error "Some dependencies are missing. Install them and try again."
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA availability..."
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    print_success "CUDA available: $GPU_COUNT GPU(s) - $GPU_NAME"
else
    print_warning "CUDA not available. Training will use CPU (slower)."
fi

# Check dataset structure
print_status "Checking dataset structure..."

DATA_DIR="data/mvtec_ad"
CATEGORIES=("capsule" "hazelnut" "screw")

if [[ ! -d "$DATA_DIR" ]]; then
    print_error "Dataset directory not found: $DATA_DIR"
    print_warning "Please download MVTec AD dataset and extract to $DATA_DIR"
    exit 1
fi

for category in "${CATEGORIES[@]}"; do
    if [[ -d "$DATA_DIR/$category/train/good" ]]; then
        TRAIN_COUNT=$(find "$DATA_DIR/$category/train/good" -name "*.png" | wc -l)
        print_success "$category: $TRAIN_COUNT training images"
    else
        print_error "$category: training data missing"
        DEPS_OK=false
    fi
    
    if [[ -d "$DATA_DIR/$category/test" ]]; then
        TEST_COUNT=$(find "$DATA_DIR/$category/test" -name "*.png" | wc -l)
        print_success "$category: $TEST_COUNT test images"
    else
        print_error "$category: test data missing"
        DEPS_OK=false
    fi
done

if [[ "$DEPS_OK" == false ]]; then
    print_error "Dataset structure validation failed."
    exit 1
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p outputs
mkdir -p runs
mkdir -p logs
mkdir -p config
print_success "Directories created"

# Test script syntax
print_status "Checking script syntax..."
if python3 -m py_compile src/train_main.py; then
    print_success "Main training script syntax OK"
else
    print_error "Main training script has syntax errors"
    exit 1
fi

if python3 -m py_compile src/train_batch.py; then
    print_success "Batch training script syntax OK"
else
    print_error "Batch training script has syntax errors"
    exit 1
fi

# Test import functionality
print_status "Testing imports..."
if python3 -c "
from src.core.model import AutoencoderUNetLite
from src.core.model_config import AutoencoderConfig
from src.data.loaders import MVTecDataset, create_dataloaders
from src.data.preprocessing import MVTecPreprocessor
from src.training.trainer import AnomalyDetectorTrainer
print('All imports successful')
" 2>/dev/null; then
    print_success "All imports working"
else
    print_warning "Some imports failed (expected if environment not fully activated)"
fi

# Offer to run quick test
echo ""
print_status "Setup validation complete!"
echo ""

read -p "Do you want to run a quick training test? (2 epochs, small batch) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running quick training test..."
    
    # Use make if available, otherwise direct python
    if command -v make >/dev/null 2>&1; then
        make train-quick CATEGORY=capsule
    else
        python3 src/train_main.py \
            --category capsule \
            --data-dir data/mvtec_ad \
            --output-dir outputs/quick_test \
            --batch-size 4 \
            --epochs 2 \
            --num-workers 1
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "Quick test completed successfully!"
        print_status "Check outputs/quick_test/ for results"
    else
        print_error "Quick test failed. Check error messages above."
        exit 1
    fi
fi

echo ""
print_success "🎉 Setup complete! You're ready to train anomaly detection models."
echo ""
echo "Next steps:"
echo "1. Start training:     make train CATEGORY=capsule"
echo "2. Batch training:     make train-core"
echo "3. Monitor progress:   make monitor"
echo "4. Check dataset:      make stats"
echo "5. Get help:          make help-train"
echo ""
echo "Happy training! 🚀"
