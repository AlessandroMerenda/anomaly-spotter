"""
Shared pytest fixtures and test configuration.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import tempfile
import shutil
from unittest.mock import Mock

# Import project modules
from src.core.model import AutoencoderUNetLite
from src.core.model_config import AutoencoderConfig
from src.data.preprocessing import MVTecPreprocessor
from src.utils.logging_utils import setup_logger


@pytest.fixture(scope="session")
def device():
    """Determine device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")  
def test_config():
    """Create test configuration."""
    config = AutoencoderConfig()
    # Override with test-friendly settings
    config.batch_size = 2
    config.num_epochs = 2
    config.learning_rate = 1e-3
    config.input_size = (64, 64)  # Smaller for faster tests
    config.val_split = 0.2
    config.use_amp = False  # Disable AMP for consistent testing
    config.num_workers = 0  # No multiprocessing in tests
    config.log_level = "ERROR"  # Reduce test noise
    return config


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return AutoencoderUNetLite(input_channels=3, output_channels=3)


@pytest.fixture
def sample_input_tensor():
    """Create sample input tensor."""
    return torch.randn(2, 3, 128, 128)  # [batch_size, channels, height, width]


@pytest.fixture
def sample_normalized_tensor():
    """Create sample normalized tensor in [-1, 1] range."""
    return torch.tanh(torch.randn(2, 3, 128, 128))  # Ensure [-1, 1] range


@pytest.fixture
def preprocessor():
    """Create preprocessor instance for testing."""
    return MVTecPreprocessor(
        image_size=(128, 128),
        normalize=True,
        normalization_type="tanh"
    )


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="anomaly_test_")
    
    # Create basic MVTec-like structure
    categories = ["test_category"]
    splits = ["train", "test"]
    defect_types = ["good", "defect1"]
    
    for category in categories:
        for split in splits:
            for defect_type in defect_types:
                dir_path = Path(temp_dir) / category / split / defect_type
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Create dummy ground truth directory for test split anomalies
                if split == "test" and defect_type != "good":
                    gt_path = Path(temp_dir) / category / "ground_truth" / defect_type
                    gt_path.mkdir(parents=True, exist_ok=True)
    
    yield Path(temp_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_image_array():
    """Create sample image as numpy array."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask_array():
    """Create sample binary mask."""
    mask = np.zeros((128, 128), dtype=np.float32)
    # Add some anomaly regions
    mask[30:60, 30:60] = 1.0
    mask[80:100, 80:100] = 1.0
    return mask


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()  
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data for testing."""
    return {
        'labels': np.array([0, 0, 0, 1, 1, 1]),  # 3 normal, 3 anomaly
        'scores': np.array([0.01, 0.02, 0.015, 0.08, 0.09, 0.085]),
        'pixel_labels': np.array([0, 0, 0, 1, 1, 0, 1, 1]),
        'pixel_scores': np.array([0.005, 0.008, 0.006, 0.095, 0.088, 0.007, 0.092, 0.099])
    }


@pytest.fixture
def sample_training_batch():
    """Create sample training batch."""
    return {
        'image': torch.randn(4, 3, 128, 128),
        'label': torch.tensor([0, 0, 1, 1]),
        'mask': torch.zeros(4, 128, 128),
        'path': ['img1.png', 'img2.png', 'img3.png', 'img4.png'],
        'category': ['test_cat', 'test_cat', 'test_cat', 'test_cat']
    }


@pytest.fixture(params=[
    (1, 1),    # Grayscale
    (3, 3),    # RGB  
    (3, 1),    # RGB to grayscale
])
def channel_combinations(request):
    """Parametrize tests with different channel combinations."""
    return request.param


@pytest.fixture(params=[
    (64, 64),
    (128, 128),
    (256, 256),
])
def image_sizes(request):
    """Parametrize tests with different image sizes.""" 
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def batch_sizes(request):
    """Parametrize tests with different batch sizes."""
    return request.param


# Helper functions for test assertions
def assert_tensor_properties(tensor: torch.Tensor, 
                           expected_shape: Tuple[int, ...],
                           expected_dtype: torch.dtype = torch.float32,
                           expected_range: Tuple[float, float] = None):
    """Assert tensor has expected properties."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if expected_range is not None:
        min_val, max_val = expected_range
        assert tensor.min().item() >= min_val, f"Tensor minimum {tensor.min().item()} < {min_val}"
        assert tensor.max().item() <= max_val, f"Tensor maximum {tensor.max().item()} > {max_val}"


def assert_model_output_valid(model: torch.nn.Module, 
                            input_tensor: torch.Tensor,
                            expected_output_shape: Tuple[int, ...] = None):
    """Assert model produces valid output."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # Check output is tensor
    assert isinstance(output, torch.Tensor), "Model output must be tensor"
    
    # Check shape matches input if not specified
    if expected_output_shape is None:
        expected_output_shape = input_tensor.shape
        
    assert output.shape == expected_output_shape, f"Output shape {output.shape} != expected {expected_output_shape}"
    
    # Check no NaN or inf values
    assert torch.isfinite(output).all(), "Model output contains NaN or Inf values"
    
    return output


# Test categories for organizing tests
class TestCategories:
    UNIT = "unit"
    INTEGRATION = "integration" 
    SLOW = "slow"
    GPU = "gpu"
    DATA = "data"
    MODEL = "model"
    PREPROCESSING = "preprocessing"
    EVALUATION = "evaluation"