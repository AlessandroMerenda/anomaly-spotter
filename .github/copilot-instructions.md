# Copilot Instructions for Anomaly Spotter

## Overview
Anomaly Spotter is an advanced industrial defect detection system built with PyTorch and U-Net autoencoder architecture. The project is structured for scalability, modularity, and ease of development.

## Architecture
### Key Components:
1. **Model**:
   - Defined in `src/model.py`.
   - AutoencoderUNetLite architecture with skip connections.
   - Input: RGB images (128x128).

2. **Data**:
   - Located in `data/mvtec_ad/`.
   - Categories: `capsule`, `hazelnut`, `screw`.
   - Subfolders: `train/`, `test/`, `ground_truth/`.

3. **Training Pipeline**:
   - Notebook: `notebooks/production_training.ipynb`.
   - Outputs: Saved in `outputs/` (e.g., `model.pth`, `thresholds.json`).

4. **Environment Configuration**:
   - Conda environments in `envs/`.
   - Environment variables in `config/env-vars/`.

## Developer Workflows
### Environment Setup:
1. **CUDA-enabled environment**:
   ```bash
   conda env create -f envs/environment.yml
   conda activate anomaly-spotter
   ```
2. **CPU-only environment**:
   ```bash
   conda env create -f envs/environment-cpu.yml
   conda activate anomaly-spotter-cpu
   ```
3. **Automated setup**:
   ```bash
   ./scripts/setup_environment.sh
   ./scripts/setup_environment.sh --cpu-only
   ```

### Training:
1. Run `notebooks/production_training.ipynb` for training.
2. Save model and thresholds in `outputs/`.

### Testing:
1. Use `src/test_model.py` for model evaluation.
2. Metrics saved in `stats/`.

## Project-Specific Conventions
1. **Data Augmentation**:
   - Use `torchvision.transforms` or `albumentations`.
   - Example: Random rotations, Gaussian noise.

2. **Batch Processing**:
   - Process images in batches (default: 32).
   - Configure batch size via `ANOMALY_SPOTTER_BATCH_SIZE` in `.env`.

3. **Threshold Optimization**:
   - Use `src/compute_thresholds.py`.
   - Outputs saved in `outputs/thresholds.json`.

## Integration Points
1. **External Dependencies**:
   - PyTorch, OpenCV, scikit-learn.
   - Conda environments for dependency management.

2. **Cross-Component Communication**:
   - Data flows from `data/` to `outputs/`.
   - Configurations managed via `.env` and `config/env-vars/`.

## Examples
### Example: Training Workflow
```bash
# Activate environment
conda activate anomaly-spotter

# Run training notebook
jupyter notebook notebooks/production_training.ipynb

# Save model
mv model.pth outputs/
```

### Example: Threshold Optimization
```bash
python src/compute_thresholds.py --input outputs/model.pth --output outputs/thresholds.json
```
