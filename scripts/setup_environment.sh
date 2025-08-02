#!/bin/bash

# Anomaly Spotter - Environment Setup Script
# Supports both CUDA and CPU-only installations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT_NAME="anomaly-spotter"
FORCE_RECREATE=false
CPU_ONLY=false
CONDA_ENV_FILE="envs/environment.yml"

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup conda environment for Anomaly Spotter project.

OPTIONS:
    -h, --help          Show this help message
    -n, --name NAME     Environment name (default: anomaly-spotter)
    -f, --force         Force recreate environment if exists
    -c, --cpu-only      Install CPU-only version (no CUDA)
    -v, --verbose       Verbose output

EXAMPLES:
    $0                  # Create default CUDA environment
    $0 -c               # Create CPU-only environment
    $0 -n my-env -f     # Force recreate with custom name
    $0 -c -n cpu-env    # CPU-only with custom name

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--name)
            ENVIRONMENT_NAME="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_RECREATE=true
            shift
            ;;
        -c|--cpu-only)
            CPU_ONLY=true
            CONDA_ENV_FILE="envs/environment-cpu.yml"
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi

# Print configuration
echo -e "${BLUE}=== Anomaly Spotter Environment Setup ===${NC}"
echo -e "Environment name: ${GREEN}${ENVIRONMENT_NAME}${NC}"
echo -e "Configuration: ${GREEN}$([ "$CPU_ONLY" = true ] && echo "CPU-only" || echo "CUDA-enabled")${NC}"
echo -e "Force recreate: ${GREEN}${FORCE_RECREATE}${NC}"
echo ""

# Check if environment exists
if conda env list | grep -q "^${ENVIRONMENT_NAME} "; then
    if [ "$FORCE_RECREATE" = true ]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n "$ENVIRONMENT_NAME" -y
    else
        echo -e "${YELLOW}Environment '${ENVIRONMENT_NAME}' already exists.${NC}"
        echo "Use --force to recreate or --name to use a different name."
        exit 1
    fi
fi

# Create environment
echo -e "${BLUE}Creating conda environment from ${CONDA_ENV_FILE}...${NC}"
conda env create -f "$CONDA_ENV_FILE" -n "$ENVIRONMENT_NAME"

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
conda activate "$ENVIRONMENT_NAME"

# Test PyTorch installation
python -c "
import torch
import sys

print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  - {torch.cuda.get_device_name(i)}')
else:
    print('Running in CPU-only mode')

# Test basic functionality
try:
    x = torch.randn(10, 10)
    if torch.cuda.is_available() and not '$CPU_ONLY':
        x = x.cuda()
        print('✅ GPU tensor creation successful')
    else:
        print('✅ CPU tensor creation successful')
except Exception as e:
    print(f'❌ Error testing PyTorch: {e}')
    sys.exit(1)
"

echo ""
echo -e "${GREEN}✅ Environment setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}To activate the environment:${NC}"
echo -e "  ${GREEN}conda activate ${ENVIRONMENT_NAME}${NC}"
echo ""
echo -e "${BLUE}To run the project:${NC}"
echo -e "  ${GREEN}cd $(pwd)${NC}"
echo -e "  ${GREEN}conda activate ${ENVIRONMENT_NAME}${NC}"
echo -e "  ${GREEN}python src/train_model.py${NC}"
echo ""

# GPU recommendations
if [ "$CPU_ONLY" = false ]; then
    echo -e "${YELLOW}💡 GPU Tips:${NC}"
    echo "  - Monitor GPU usage: nvidia-smi"
    echo "  - For memory issues: reduce batch size in config/training.yml"
    echo "  - Fallback to CPU: set CUDA_VISIBLE_DEVICES=''"
    echo ""
fi
