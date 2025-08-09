"""
Unit tests for data preprocessing pipeline.
Tests image loading, transformations, and normalization.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import cv2
from unittest.mock import Mock, patch, MagicMock

from src.data.preprocessing import MVTecPreprocessor
from tests.conftest import assert_tensor_properties, TestCategories


class TestMVTecPreprocessor:
    """Test suite for MVTecPreprocessor."""
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocessor_initialization_default(self):
        """Test preprocessor initialization with default parameters."""
        preprocessor = MVTecPreprocessor()
        
        assert preprocessor.image_size == (128, 128)
        assert preprocessor.normalize is True
        assert preprocessor.normalization_type == "tanh"
        assert preprocessor.SUPPORTED_EXTENSIONS == ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    @pytest.mark.parametrize("image_size", [(64, 64), (128, 128), (256, 256)])
    def test_preprocessor_initialization_custom_size(self, image_size):
        """Test preprocessor initialization with custom image sizes."""
        preprocessor = MVTecPreprocessor(image_size=image_size)
        assert preprocessor.image_size == image_size
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    @pytest.mark.parametrize("norm_type", ["tanh", "imagenet"])
    def test_preprocessor_initialization_normalization_types(self, norm_type):
        """Test preprocessor initialization with different normalization types."""
        preprocessor = MVTecPreprocessor(normalization_type=norm_type)
        assert preprocessor.normalization_type == norm_type
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocessor_initialization_invalid_norm_type(self):
        """Test preprocessor initialization with invalid normalization type."""
        with pytest.raises(ValueError, match="Unknown normalization type"):
            MVTecPreprocessor(normalization_type="invalid")
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_load_image_safely_valid_formats(self, temp_data_dir, sample_image_array):
        """Test loading images with various valid formats."""
        preprocessor = MVTecPreprocessor()
        
        # Test different formats
        formats = ['.png', '.jpg', '.bmp']
        
        for fmt in formats:
            # Create test image
            image_path = temp_data_dir / f"test_image{fmt}"
            if fmt == '.png':
                Image.fromarray(sample_image_array).save(image_path, 'PNG')
            elif fmt == '.jpg':
                Image.fromarray(sample_image_array).save(image_path, 'JPEG')
            elif fmt == '.bmp':
                Image.fromarray(sample_image_array).save(image_path, 'BMP')
            
            # Test loading
            loaded_image = preprocessor._load_image_safely(image_path)
            
            assert loaded_image is not None
            assert isinstance(loaded_image, np.ndarray)
            assert loaded_image.shape == (128, 128, 3)  # RGB format
            assert loaded_image.dtype == np.uint8
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_load_image_safely_nonexistent_file(self):
        """Test loading non-existent image file."""
        preprocessor = MVTecPreprocessor()
        
        result = preprocessor._load_image_safely("/non/existent/path.png")
        assert result is None
    
    @pytest.mark.unit 
    @pytest.mark.preprocessing
    def test_load_image_safely_unsupported_format(self, temp_data_dir):
        """Test loading image with unsupported format."""
        preprocessor = MVTecPreprocessor()
        
        # Create file with unsupported extension
        unsupported_file = temp_data_dir / "test.xyz"
        unsupported_file.write_text("dummy content")
        
        result = preprocessor._load_image_safely(unsupported_file)
        assert result is None
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_load_image_safely_fallback_to_pil(self, temp_data_dir, sample_image_array):
        """Test fallback from OpenCV to PIL when OpenCV fails."""
        preprocessor = MVTecPreprocessor()
        
        # Create valid image
        image_path = temp_data_dir / "test.png"
        Image.fromarray(sample_image_array).save(image_path, 'PNG')
        
        # Mock OpenCV to fail, forcing PIL fallback
        with patch('cv2.imread', return_value=None):
            loaded_image = preprocessor._load_image_safely(image_path)
            
            assert loaded_image is not None
            assert isinstance(loaded_image, np.ndarray)
            assert loaded_image.shape == (128, 128, 3)
    
    @pytest.mark.unit
    @pytest.mark.preprocessing  
    def test_preprocess_single_training_mode(self, temp_data_dir, sample_image_array):
        """Test single image preprocessing in training mode."""
        preprocessor = MVTecPreprocessor(
            image_size=(64, 64),
            normalization_type="tanh"
        )
        
        # Create test image
        image_path = temp_data_dir / "test.png"
        Image.fromarray(sample_image_array).save(image_path, 'PNG')
        
        # Preprocess in training mode (with augmentations)
        result = preprocessor.preprocess_single(image_path, is_training=True)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert_tensor_properties(
            result,
            expected_shape=(3, 64, 64),  # C, H, W
            expected_dtype=torch.float32,
            expected_range=(-1.0, 1.0)  # Tanh normalization
        )
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocess_single_inference_mode(self, temp_data_dir, sample_image_array):
        """Test single image preprocessing in inference mode."""
        preprocessor = MVTecPreprocessor(
            image_size=(64, 64),
            normalization_type="tanh"
        )
        
        # Create test image
        image_path = temp_data_dir / "test.png"
        Image.fromarray(sample_image_array).save(image_path, 'PNG')
        
        # Preprocess in inference mode (no augmentations)
        result = preprocessor.preprocess_single(image_path, is_training=False)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert_tensor_properties(
            result,
            expected_shape=(3, 64, 64),
            expected_dtype=torch.float32,
            expected_range=(-1.0, 1.0)
        )
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocess_single_imagenet_normalization(self, temp_data_dir, sample_image_array):
        """Test preprocessing with ImageNet normalization."""
        preprocessor = MVTecPreprocessor(
            image_size=(64, 64),
            normalization_type="imagenet"
        )
        
        # Create test image
        image_path = temp_data_dir / "test.png"
        Image.fromarray(sample_image_array).save(image_path, 'PNG')
        
        result = preprocessor.preprocess_single(image_path, is_training=False)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
        # ImageNet normalization can produce values outside [-1, 1]
        assert_tensor_properties(
            result,
            expected_shape=(3, 64, 64),
            expected_dtype=torch.float32
        )
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocess_single_invalid_image(self):
        """Test preprocessing with invalid image path."""
        preprocessor = MVTecPreprocessor()
        
        result = preprocessor.preprocess_single("/invalid/path.png", is_training=False)
        assert result is None
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocess_batch_success(self, temp_data_dir, sample_image_array):
        """Test batch preprocessing with valid images."""
        preprocessor = MVTecPreprocessor(image_size=(32, 32))
        
        # Create multiple test images
        image_paths = []
        for i in range(3):
            image_path = temp_data_dir / f"test_{i}.png"
            Image.fromarray(sample_image_array).save(image_path, 'PNG')
            image_paths.append(image_path)
        
        # Process batch
        result = preprocessor.preprocess_batch(image_paths, is_training=False)
        
        assert isinstance(result, torch.Tensor)
        assert_tensor_properties(
            result,
            expected_shape=(3, 3, 32, 32),  # B, C, H, W
            expected_dtype=torch.float32
        )
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocess_batch_some_failures_skip(self, temp_data_dir, sample_image_array):
        """Test batch preprocessing with some failed images (skip mode)."""
        preprocessor = MVTecPreprocessor(image_size=(32, 32))
        
        # Create mix of valid and invalid paths
        valid_path = temp_data_dir / "valid.png"
        Image.fromarray(sample_image_array).save(valid_path, 'PNG')
        
        image_paths = [
            valid_path,
            "/invalid/path1.png",
            "/invalid/path2.png"
        ]
        
        # Should skip failed images
        result = preprocessor.preprocess_batch(
            image_paths, 
            is_training=False, 
            skip_failed=True
        )
        
        assert isinstance(result, torch.Tensor)
        # Only 1 valid image should be processed
        assert_tensor_properties(
            result,
            expected_shape=(1, 3, 32, 32),
            expected_dtype=torch.float32
        )
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocess_batch_failures_no_skip(self, temp_data_dir):
        """Test batch preprocessing with failures and skip_failed=False."""
        preprocessor = MVTecPreprocessor()
        
        # All invalid paths
        image_paths = ["/invalid/path1.png", "/invalid/path2.png"]
        
        # Should raise error when skip_failed=False
        with pytest.raises(Exception):  # DataError from utils
            preprocessor.preprocess_batch(
                image_paths,
                is_training=False,
                skip_failed=False
            )
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocess_batch_all_failures(self):
        """Test batch preprocessing when all images fail."""
        preprocessor = MVTecPreprocessor()
        
        # All invalid paths
        image_paths = ["/invalid/path1.png", "/invalid/path2.png"]
        
        # Should raise error even with skip_failed=True when no images succeed
        with pytest.raises(Exception):  # DataError
            preprocessor.preprocess_batch(
                image_paths,
                is_training=False,
                skip_failed=True
            )
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_get_transform_training_vs_inference(self):
        """Test that training and inference transforms are different."""
        preprocessor = MVTecPreprocessor()
        
        train_transform = preprocessor.get_transform(is_training=True)
        test_transform = preprocessor.get_transform(is_training=False)
        
        # Should return different transform objects
        assert train_transform is not test_transform
        
        # Training transform should have more steps (augmentations)
        train_steps = len(train_transform.transforms)
        test_steps = len(test_transform.transforms)
        assert train_steps > test_steps
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocessing_deterministic_in_inference(self, temp_data_dir, sample_image_array):
        """Test that inference preprocessing is deterministic."""
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        # Create test image
        image_path = temp_data_dir / "test.png"
        Image.fromarray(sample_image_array).save(image_path, 'PNG')
        
        # Process same image twice in inference mode
        result1 = preprocessor.preprocess_single(image_path, is_training=False)
        result2 = preprocessor.preprocess_single(image_path, is_training=False)
        
        # Should be identical
        torch.testing.assert_close(result1, result2, atol=1e-6, rtol=1e-6)
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_preprocessing_stochastic_in_training(self, temp_data_dir, sample_image_array):
        """Test that training preprocessing introduces stochasticity."""
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        # Create test image
        image_path = temp_data_dir / "test.png"
        Image.fromarray(sample_image_array).save(image_path, 'PNG')
        
        # Process same image multiple times in training mode
        results = []
        for _ in range(5):
            result = preprocessor.preprocess_single(image_path, is_training=True)
            results.append(result)
        
        # Should have some variation due to augmentations
        # Check that not all results are identical
        all_identical = True
        for i in range(1, len(results)):
            if not torch.allclose(results[0], results[i], atol=1e-6):
                all_identical = False
                break
        
        assert not all_identical, "Training preprocessing should introduce variation"
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_image_size_consistency(self, temp_data_dir):
        """Test that output size matches configured size regardless of input size."""
        preprocessor = MVTecPreprocessor(image_size=(96, 96))
        
        # Create images of different sizes
        sizes = [(64, 64), (128, 128), (256, 256)]
        
        for size in sizes:
            # Create test image with specific size
            image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            image_path = temp_data_dir / f"test_{size[0]}x{size[1]}.png"
            Image.fromarray(image_array).save(image_path, 'PNG')
            
            # Process image
            result = preprocessor.preprocess_single(image_path, is_training=False)
            
            # Should always output configured size
            assert result.shape == (3, 96, 96), f"Wrong output shape for input size {size}"


class TestPreprocessingErrorHandling:
    """Test error handling and edge cases in preprocessing."""
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_corrupt_image_handling(self, temp_data_dir):
        """Test handling of corrupted image files."""
        preprocessor = MVTecPreprocessor()
        
        # Create corrupted image file
        corrupt_path = temp_data_dir / "corrupt.png"
        corrupt_path.write_bytes(b"not an image file")
        
        result = preprocessor._load_image_safely(corrupt_path)
        assert result is None
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_empty_image_file(self, temp_data_dir):
        """Test handling of empty image files."""
        preprocessor = MVTecPreprocessor()
        
        # Create empty file
        empty_path = temp_data_dir / "empty.png"
        empty_path.touch()
        
        result = preprocessor._load_image_safely(empty_path)
        assert result is None
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    @patch('src.data.preprocessing.ALBUMENTATIONS_AVAILABLE', False)
    def test_fallback_when_albumentations_unavailable(self):
        """Test fallback behavior when albumentations is not available."""
        # This test checks that the code gracefully handles missing albumentations
        # Note: This would require refactoring the current code to support fallback
        
        # For now, just test that the import check works
        from src.data.preprocessing import ALBUMENTATIONS_AVAILABLE
        
        # If albumentations is available in test env, we can't easily test this
        # This test serves as documentation of the intended behavior
        if ALBUMENTATIONS_AVAILABLE:
            pytest.skip("Albumentations is available, cannot test fallback")
    
    @pytest.mark.unit
    @pytest.mark.preprocessing
    def test_memory_efficiency_large_batch(self, temp_data_dir, sample_image_array):
        """Test memory efficiency with larger batches."""
        preprocessor = MVTecPreprocessor(image_size=(128, 128))
        
        # Create multiple test images
        num_images = 10
        image_paths = []
        for i in range(num_images):
            image_path = temp_data_dir / f"large_batch_{i}.png"
            Image.fromarray(sample_image_array).save(image_path, 'PNG')
            image_paths.append(image_path)
        
        # Process batch - should not run out of memory
        result = preprocessor.preprocess_batch(image_paths, is_training=False)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (num_images, 3, 128, 128)
        
        # Check memory usage is reasonable
        memory_mb = result.numel() * result.element_size() / (1024 * 1024)
        assert memory_mb < 100, f"Batch uses too much memory: {memory_mb:.2f}MB"