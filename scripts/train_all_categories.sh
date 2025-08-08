#!/bin/bash
#
# Script di Training Batch per Anomaly Spotter
# Train automatico su tutte le categorie MVTec AD con valutazione post-training
#
# Usage:
#   ./scripts/train_all_categories.sh [--parallel] [--gpu-ids "0,1"] [--config custom_config.yaml]
#
# Features:
# - Training sequenziale o parallelo
# - Valutazione automatica post-training
# - Gestione errori robusta
# - Logging dettagliato
# - Supporto multi-GPU
#

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default categories (subset of MVTec AD for core training)
CORE_CATEGORIES=("capsule" "hazelnut" "screw")

# Extended categories (full MVTec AD dataset)
ALL_CATEGORIES=("capsule" "hazelnut" "screw" "metal_nut" "pill" 
                "toothbrush" "transistor" "zipper" "cable" "grid"
                "carpet" "leather" "tile" "wood" "bottle")

# Default configuration
DATA_DIR="data/mvtec_ad"
OUTPUT_DIR="outputs"
CONFIG_FILE="config/training_config.yaml"
NUM_WORKERS=4
BATCH_SIZE=32
EPOCHS=100
DEVICE="auto"
PARALLEL_MODE=false
GPU_IDS=""
USE_ALL_CATEGORIES=false
EVALUATE_AFTER_TRAINING=true
FIND_OPTIMAL_THRESHOLDS=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}=========================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}=========================${NC}"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --parallel              Run training in parallel mode
    --gpu-ids "0,1,2"      Specify GPU IDs for parallel training
    --config FILE          Custom configuration file
    --data-dir DIR         Data directory (default: data/mvtec_ad)
    --output-dir DIR       Output directory (default: outputs)
    --batch-size SIZE      Batch size (default: 32)
    --epochs NUM           Number of epochs (default: 100)
    --num-workers NUM      Number of workers (default: 4)
    --all-categories       Train on all 15 MVTec categories
    --core-categories      Train only on core categories (default)
    --no-evaluation        Skip post-training evaluation
    --no-optimal           Skip optimal threshold search
    --device DEVICE        Device: auto/cuda/cpu (default: auto)
    --help                 Show this help message

EXAMPLES:
    # Basic training on core categories
    $0

    # Parallel training with custom config
    $0 --parallel --gpu-ids "0,1" --config custom.yaml

    # Train all categories sequentially
    $0 --all-categories --batch-size 16

    # Quick training without evaluation
    $0 --epochs 10 --no-evaluation
EOF
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check required directories
    if [[ ! -d "$DATA_DIR" ]]; then
        log_error "Data directory not found: $DATA_DIR"
        log_info "Please download and extract MVTec AD dataset"
        exit 1
    fi
    
    # Check training script
    if [[ ! -f "src/train_main.py" ]]; then
        log_error "Training script not found: src/train_main.py"
        exit 1
    fi
    
    # Check evaluation script
    if [[ ! -f "src/evaluate_model_post_training.py" && "$EVALUATE_AFTER_TRAINING" == "true" ]]; then
        log_warning "Evaluation script not found: src/evaluate_model_post_training.py"
        log_warning "Post-training evaluation will be skipped"
        EVALUATE_AFTER_TRAINING=false
    fi
    
    log_success "Dependencies check passed"
}

validate_categories() {
    local categories=("$@")
    
    for category in "${categories[@]}"; do
        local category_path="$DATA_DIR/$category"
        if [[ ! -d "$category_path" ]]; then
            log_warning "Category directory not found: $category_path"
            log_warning "Skipping category: $category"
            continue
        fi
        
        # Check for training data
        if [[ ! -d "$category_path/train/good" ]]; then
            log_warning "Training data not found for category: $category"
            continue
        fi
        
        # Check for test data
        if [[ ! -d "$category_path/test" ]]; then
            log_warning "Test data not found for category: $category"
            continue
        fi
    done
}

train_single_category() {
    local category=$1
    local gpu_id=${2:-"auto"}
    
    log_header "Training Category: $category"
    
    # Prepare output directory
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local category_output_dir="$OUTPUT_DIR/${category}_${timestamp}"
    mkdir -p "$category_output_dir"
    
    # Setup device
    local device_arg="$DEVICE"
    if [[ "$gpu_id" != "auto" ]]; then
        device_arg="cuda:$gpu_id"
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        log_info "Using GPU: $gpu_id"
    fi
    
    # Training command
    local train_cmd=(
        python src/train_main.py
        --category "$category"
        --data-dir "$DATA_DIR"
        --output-dir "$category_output_dir"
        --config "$CONFIG_FILE"
        --batch-size "$BATCH_SIZE"
        --epochs "$EPOCHS"
        --num-workers "$NUM_WORKERS"
        --device "$device_arg"
    )
    
    log_info "Starting training for $category..."
    log_info "Command: ${train_cmd[*]}"
    
    # Execute training
    if "${train_cmd[@]}" 2>&1 | tee "$category_output_dir/training.log"; then
        log_success "Training completed for $category"
        
        # Post-training evaluation
        if [[ "$EVALUATE_AFTER_TRAINING" == "true" ]]; then
            evaluate_category "$category" "$category_output_dir" "$device_arg"
        fi
        
        # Generate training summary
        generate_training_summary "$category" "$category_output_dir"
        
    else
        log_error "Training failed for $category"
        echo "Training failed at $(date)" >> "$category_output_dir/training.log"
        return 1
    fi
}

evaluate_category() {
    local category=$1
    local model_dir=$2
    local device=${3:-"auto"}
    
    log_info "Starting post-training evaluation for $category..."
    
    # Check if model exists
    if [[ ! -f "$model_dir/model.pth" ]]; then
        log_warning "Model not found for evaluation: $model_dir/model.pth"
        return 1
    fi
    
    # Evaluation command
    local eval_cmd=(
        python src/evaluate_model_post_training.py
        --model-dir "$model_dir"
        --category "$category"
        --data-dir "$DATA_DIR"
        --device "$device"
        --batch-size "$BATCH_SIZE"
        --num-samples 15
    )
    
    # Add optimal threshold search if enabled
    if [[ "$FIND_OPTIMAL_THRESHOLDS" == "true" ]]; then
        eval_cmd+=(--find-optimal)
    fi
    
    log_info "Evaluation command: ${eval_cmd[*]}"
    
    # Execute evaluation
    if "${eval_cmd[@]}" 2>&1 | tee "$model_dir/evaluation.log"; then
        log_success "Evaluation completed for $category"
    else
        log_warning "Evaluation failed for $category"
        return 1
    fi
}

generate_training_summary() {
    local category=$1
    local model_dir=$2
    
    local summary_file="$model_dir/training_summary.txt"
    
    cat > "$summary_file" << EOF
Training Summary for $category
==============================

Timestamp: $(date)
Category: $category
Model Directory: $model_dir
Configuration: $CONFIG_FILE

Training Parameters:
- Batch Size: $BATCH_SIZE
- Epochs: $EPOCHS
- Workers: $NUM_WORKERS
- Device: $DEVICE

Files Generated:
$(ls -la "$model_dir" | grep -E '\.(pth|json|yaml|png|log)$' || echo "No model files found")

EOF

    # Add evaluation results if available
    if [[ -f "$model_dir/post_training_evaluation/evaluation_results.json" ]]; then
        echo "" >> "$summary_file"
        echo "Evaluation Results:" >> "$summary_file"
        echo "===================" >> "$summary_file"
        python -c "
import json
try:
    with open('$model_dir/post_training_evaluation/evaluation_results.json', 'r') as f:
        results = json.load(f)
    for key, value in results.items():
        if isinstance(value, (int, float)) and any(metric in key.lower() for metric in ['auroc', 'auprc', 'f1', 'precision', 'recall', 'accuracy']):
            print(f'{key}: {value:.4f}')
except Exception as e:
    print(f'Could not parse evaluation results: {e}')
" >> "$summary_file"
    fi
    
    log_info "Training summary saved: $summary_file"
}

train_parallel() {
    local categories=("$@")
    
    if [[ -z "$GPU_IDS" ]]; then
        log_error "GPU IDs must be specified for parallel training"
        log_info "Use --gpu-ids \"0,1,2\" to specify GPU IDs"
        exit 1
    fi
    
    # Parse GPU IDs
    IFS=',' read -ra GPUS <<< "$GPU_IDS"
    local gpu_count=${#GPUS[@]}
    
    log_header "Parallel Training Mode"
    log_info "Available GPUs: ${GPUS[*]}"
    log_info "Categories to train: ${categories[*]}"
    
    # Start training jobs
    local pids=()
    local gpu_idx=0
    
    for category in "${categories[@]}"; do
        local gpu_id=${GPUS[$gpu_idx]}
        
        log_info "Starting training for $category on GPU $gpu_id"
        
        # Start training in background
        (
            train_single_category "$category" "$gpu_id"
        ) &
        
        pids+=($!)
        gpu_idx=$(((gpu_idx + 1) % gpu_count))
        
        # Limit concurrent jobs to number of GPUs
        if [[ ${#pids[@]} -ge $gpu_count ]]; then
            # Wait for one job to finish
            wait ${pids[0]}
            pids=("${pids[@]:1}")  # Remove first PID
        fi
    done
    
    # Wait for all remaining jobs
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    log_success "All parallel training jobs completed"
}

train_sequential() {
    local categories=("$@")
    
    log_header "Sequential Training Mode"
    log_info "Categories to train: ${categories[*]}"
    
    local success_count=0
    local total_count=${#categories[@]}
    
    for category in "${categories[@]}"; do
        log_info "Training category $((success_count + 1))/$total_count: $category"
        
        if train_single_category "$category"; then
            ((success_count++))
        else
            log_error "Failed to train category: $category"
        fi
        
        # Progress update
        log_info "Progress: $success_count/$total_count completed"
        echo ""
    done
    
    # Final summary
    log_header "Training Summary"
    log_info "Successfully trained: $success_count/$total_count categories"
    
    if [[ $success_count -eq $total_count ]]; then
        log_success "All categories trained successfully!"
    else
        log_warning "Some categories failed. Check logs for details."
    fi
}

generate_final_report() {
    local categories=("$@")
    
    log_info "Generating final training report..."
    
    local report_file="$OUTPUT_DIR/batch_training_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Batch Training Report

**Generated:** $(date)
**Script:** $0
**Mode:** $([ "$PARALLEL_MODE" = true ] && echo "Parallel" || echo "Sequential")
**Categories:** ${categories[*]}

## Configuration

- Data Directory: $DATA_DIR
- Output Directory: $OUTPUT_DIR
- Config File: $CONFIG_FILE
- Batch Size: $BATCH_SIZE
- Epochs: $EPOCHS
- Workers: $NUM_WORKERS
- Device: $DEVICE

## Results Summary

EOF

    # Add results for each category
    for category in "${categories[@]}"; do
        echo "### $category" >> "$report_file"
        echo "" >> "$report_file"
        
        # Find latest model directory for this category
        local latest_dir=$(ls -td "$OUTPUT_DIR"/${category}_* 2>/dev/null | head -1 || echo "")
        
        if [[ -n "$latest_dir" && -d "$latest_dir" ]]; then
            echo "- **Status:** ✅ Completed" >> "$report_file"
            echo "- **Model Directory:** $latest_dir" >> "$report_file"
            
            # Add evaluation results if available
            if [[ -f "$latest_dir/post_training_evaluation/evaluation_results.json" ]]; then
                echo "- **Evaluation:** Available" >> "$report_file"
                echo "- **Results:**" >> "$report_file"
                python -c "
import json
try:
    with open('$latest_dir/post_training_evaluation/evaluation_results.json', 'r') as f:
        results = json.load(f)
    for key, value in results.items():
        if isinstance(value, (int, float)) and any(metric in key.lower() for metric in ['auroc', 'f1']):
            print(f'  - {key}: {value:.4f}')
except:
    pass
" >> "$report_file"
            else
                echo "- **Evaluation:** Not available" >> "$report_file"
            fi
        else
            echo "- **Status:** ❌ Failed or not found" >> "$report_file"
        fi
        
        echo "" >> "$report_file"
    done
    
    # Add system information
    cat >> "$report_file" << EOF

## System Information

- **Hostname:** $(hostname)
- **Python Version:** $(python --version 2>&1)
- **PyTorch Version:** $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not available")
- **CUDA Available:** $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "Unknown")
- **GPU Count:** $(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "Unknown")

## Files Generated

$(find "$OUTPUT_DIR" -name "${categories[0]}_*" -type d 2>/dev/null | head -5 | while read dir; do echo "- $dir"; done)
$([ $(find "$OUTPUT_DIR" -name "${categories[0]}_*" -type d 2>/dev/null | wc -l) -gt 5 ] && echo "- ... and more")

EOF

    log_success "Final report generated: $report_file"
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --parallel)
                PARALLEL_MODE=true
                shift
                ;;
            --gpu-ids)
                GPU_IDS="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --num-workers)
                NUM_WORKERS="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --all-categories)
                USE_ALL_CATEGORIES=true
                shift
                ;;
            --core-categories)
                USE_ALL_CATEGORIES=false
                shift
                ;;
            --no-evaluation)
                EVALUATE_AFTER_TRAINING=false
                shift
                ;;
            --no-optimal)
                FIND_OPTIMAL_THRESHOLDS=false
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Select categories
    if [[ "$USE_ALL_CATEGORIES" == "true" ]]; then
        CATEGORIES=("${ALL_CATEGORIES[@]}")
    else
        CATEGORIES=("${CORE_CATEGORIES[@]}")
    fi
    
    # Print configuration
    log_header "Anomaly Spotter Batch Training"
    log_info "Mode: $([ "$PARALLEL_MODE" = true ] && echo "Parallel" || echo "Sequential")"
    log_info "Categories: ${CATEGORIES[*]}"
    log_info "Data Directory: $DATA_DIR"
    log_info "Output Directory: $OUTPUT_DIR"
    log_info "Configuration: $CONFIG_FILE"
    log_info "Batch Size: $BATCH_SIZE, Epochs: $EPOCHS"
    log_info "Post-training Evaluation: $EVALUATE_AFTER_TRAINING"
    
    # Dependency checks
    check_dependencies
    validate_categories "${CATEGORIES[@]}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Start training
    local start_time=$(date +%s)
    
    if [[ "$PARALLEL_MODE" == "true" ]]; then
        train_parallel "${CATEGORIES[@]}"
    else
        train_sequential "${CATEGORIES[@]}"
    fi
    
    # Calculate elapsed time
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    
    log_success "Batch training completed in ${hours}h ${minutes}m ${seconds}s"
    
    # Generate final report
    generate_final_report "${CATEGORIES[@]}"
    
    log_header "Training Complete!"
    log_info "Check output directory: $OUTPUT_DIR"
    log_info "See final report for detailed results"
}

# Execute main function
main "$@"
