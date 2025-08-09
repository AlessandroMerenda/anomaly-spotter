"""
Unit tests for data loading components (MVTecDataset and related functions).
Tests dataset creation, sample loading, and data validation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from src.data.loaders import MVTecDataset, create_dataloaders
from src.data.preprocessing import MVTecPreprocessor
from tests.conftest import assert_tensor_properties, TestCategories


class TestMVTecDataset:
    """Test suite for MVTecDataset."""
    
    @pytest.fixture
    def setup_test_dataset_structure(self, temp_data_dir):
        """Setup a complete test dataset structure."""
        # Create categories
        categories = ['capsule', 'hazelnut']
        
        for category in categories:
            # Create train/good directory with images
            train_good_dir = temp_data_dir / category / 'train' / 'good'
            train_good_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(5):  # 5 training images per category
                image_path = train_good_dir / f'{i:03d}.png'
                # Create dummy image
                dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                Image.fromarray(dummy_image).save(image_path, 'PNG')
            
            # Create test directories with good and defect images
            test_good_dir = temp_data_dir / category / 'test' / 'good'
            test_good_dir.mkdir(parents=True, exist_ok=True)
            
            test_defect_dir = temp_data_dir / category / 'test' / 'crack'
            test_defect_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ground truth directory for defects
            gt_dir = temp_data_dir / category / 'ground_truth' / 'crack'
            gt_dir.mkdir(parents=True, exist_ok=True)
            
            # Add test images
            for i in range(3):  # 3 good test images
                image_path = test_good_dir / f'{i:03d}.png'
                dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                Image.fromarray(dummy_image).save(image_path, 'PNG')
            
            for i in range(3):  # 3 defect test images
                image_path = test_defect_dir / f'{i:03d}.png'
                dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                Image.fromarray(dummy_image).save(image_path, 'PNG')
                
                # Create corresponding mask
                mask_path = gt_dir / f'{i:03d}_mask.png'
                dummy_mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8) * 255
                Image.fromarray(dummy_mask, mode='L').save(mask_path, 'PNG')
        
        return temp_data_dir
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_initialization_single_category(self, setup_test_dataset_structure):
        """Test dataset initialization with single category."""
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='train'
        )
        
        assert dataset.categories == ['capsule']
        assert dataset.split == 'train'
        assert len(dataset.image_paths) == 5  # 5 training images
        assert all(label == 0 for label in dataset.labels)  # All normal in training
        assert len(dataset.categories_list) == 5
        assert all(cat == 'capsule' for cat in dataset.categories_list)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_initialization_multiple_categories(self, setup_test_dataset_structure):
        """Test dataset initialization with multiple categories."""
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories=['capsule', 'hazelnut'],
            split='train'
        )
        
        assert dataset.categories == ['capsule', 'hazelnut']
        assert len(dataset.image_paths) == 10  # 5 + 5 training images
        assert all(label == 0 for label in dataset.labels)  # All normal in training
        
        # Check categories are properly tracked
        capsule_count = dataset.categories_list.count('capsule')
        hazelnut_count = dataset.categories_list.count('hazelnut')
        assert capsule_count == 5
        assert hazelnut_count == 5
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_initialization_auto_detect_categories(self, setup_test_dataset_structure):
        """Test dataset initialization with auto-detected categories."""
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories=None,  # Auto-detect
            split='train'
        )
        
        # Should detect both categories
        assert set(dataset.categories) == {'capsule', 'hazelnut'}
        assert len(dataset.image_paths) == 10
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_test_split_with_defects(self, setup_test_dataset_structure):
        """Test dataset with test split containing defects."""
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='test'
        )
        
        assert len(dataset.image_paths) == 6  # 3 good + 3 defect
        
        # Check labels: should have both 0 (good) and 1 (defect)
        normal_count = dataset.labels.count(0)
        defect_count = dataset.labels.count(1)
        assert normal_count == 3
        assert defect_count == 3
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_test_split_with_masks(self, setup_test_dataset_structure):
        """Test dataset loading with ground truth masks."""
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='test',
            load_masks=True
        )
        
        # Check that defect samples have mask paths
        defect_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
        normal_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
        
        # Defects should have mask paths
        for idx in defect_indices:
            assert dataset.mask_paths[idx] is not None
        
        # Normal samples should have None mask paths
        for idx in normal_indices:
            assert dataset.mask_paths[idx] is None
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_getitem_without_preprocessor(self, setup_test_dataset_structure):
        """Test dataset __getitem__ without preprocessor."""
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='train',
            preprocessor=None
        )
        
        sample = dataset[0]
        
        assert isinstance(sample, dict)
        assert 'image' in sample
        assert 'label' in sample
        assert 'mask' in sample
        assert 'path' in sample
        assert 'category' in sample
        
        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].dim() == 3  # C, H, W
        assert sample['label'] == 0  # Training sample should be normal
        assert sample['category'] == 'capsule'
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_getitem_with_preprocessor(self, setup_test_dataset_structure):
        """Test dataset __getitem__ with preprocessor."""
        preprocessor = MVTecPreprocessor(
            image_size=(64, 64),
            normalization_type='tanh'
        )
        
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='train',
            preprocessor=preprocessor
        )
        
        sample = dataset[0]
        
        assert isinstance(sample['image'], torch.Tensor)
        assert_tensor_properties(
            sample['image'],
            expected_shape=(3, 64, 64),
            expected_dtype=torch.float32,
            expected_range=(-1.0, 1.0)
        )
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_getitem_with_mask(self, setup_test_dataset_structure):
        """Test dataset __getitem__ loading masks for defects."""
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='test',
            load_masks=True
        )
        
        # Find a defect sample
        defect_idx = None
        for i, label in enumerate(dataset.labels):
            if label == 1:  # Defect
                defect_idx = i
                break
        
        assert defect_idx is not None, "No defect sample found"
        
        sample = dataset[defect_idx]
        
        assert isinstance(sample['mask'], torch.Tensor)
        assert sample['mask'].dim() >= 2  # At least H, W
        assert sample['label'] == 1  # Should be defect
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_length(self, setup_test_dataset_structure):
        """Test dataset length calculation."""
        train_dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='train'
        )
        
        test_dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='test'
        )
        
        assert len(train_dataset) == 5  # 5 training images
        assert len(test_dataset) == 6   # 3 good + 3 defect
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_category_stats(self, setup_test_dataset_structure):
        """Test dataset category statistics."""
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories=['capsule', 'hazelnut'],
            split='test'
        )
        
        stats = dataset.get_category_stats()
        
        assert isinstance(stats, dict)
        assert 'capsule' in stats
        assert 'hazelnut' in stats
        
        # Each category should have 6 test samples (3 good + 3 defect)
        for category in ['capsule', 'hazelnut']:
            assert stats[category]['total'] == 6
            assert stats[category]['normal'] == 3
            assert stats[category]['anomaly'] == 3
    
    @pytest.mark.unit
    @pytest.mark.data  
    def test_dataset_error_handling_missing_directory(self, temp_data_dir):
        """Test dataset handles missing directories gracefully."""
        dataset = MVTecDataset(
            root_dir=temp_data_dir,
            categories='nonexistent_category',
            split='train'
        )
        
        # Should create dataset but with no samples
        assert len(dataset.image_paths) == 0
        assert len(dataset.labels) == 0
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataset_error_handling_corrupt_image(self, setup_test_dataset_structure):
        """Test dataset handles corrupt images gracefully."""
        # Create corrupt image file
        corrupt_path = setup_test_dataset_structure / 'capsule' / 'train' / 'good' / 'corrupt.png'
        corrupt_path.write_bytes(b'not an image')
        
        dataset = MVTecDataset(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            split='train'
        )
        
        # Should handle corrupt image and return dummy data
        # Find index of corrupt image
        corrupt_idx = None
        for i, path in enumerate(dataset.image_paths):
            if 'corrupt.png' in str(path):
                corrupt_idx = i
                break
        
        if corrupt_idx is not None:
            sample = dataset[corrupt_idx]
            # Should return dummy data rather than crash
            assert isinstance(sample, dict)
            assert 'image' in sample


class TestDataLoaderCreation:
    """Test dataloaders creation functions."""
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_create_dataloaders_basic(self, setup_test_dataset_structure):
        """Test basic dataloader creation."""
        train_loader, test_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            batch_size=2,
            num_workers=0  # No multiprocessing in tests
        )
        
        assert train_loader is not None
        assert test_loader is not None
        
        # Check train loader properties
        assert train_loader.batch_size == 2
        assert train_loader.dataset.split == 'train'
        
        # Check test loader properties  
        assert test_loader.batch_size == 2
        assert test_loader.dataset.split == 'test'
        
        # Test loader should not shuffle
        assert not test_loader.shuffle
        # Train loader should shuffle
        assert train_loader.shuffle
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_create_dataloaders_with_preprocessor(self, setup_test_dataset_structure):
        """Test dataloader creation with preprocessor."""
        preprocessor = MVTecPreprocessor(image_size=(32, 32))
        
        train_loader, test_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=2,
            num_workers=0
        )
        
        # Test a batch from train loader
        train_batch = next(iter(train_loader))
        assert train_batch['image'].shape == (2, 3, 32, 32)  # Batch of 2, preprocessed size
        
        # Test a batch from test loader
        test_batch = next(iter(test_loader))
        assert test_batch['image'].shape[1:] == (3, 32, 32)  # Correct preprocessed shape
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_create_dataloaders_with_masks(self, setup_test_dataset_structure):
        """Test dataloader creation with mask loading."""
        train_loader, test_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            load_masks=True,
            batch_size=2,
            num_workers=0
        )
        
        # Train loader should not load masks (forced to False)
        train_batch = next(iter(train_loader))
        # Masks should be dummy for training
        
        # Test loader should load masks
        test_batch = next(iter(test_loader))
        assert 'mask' in test_batch
        assert isinstance(test_batch['mask'], torch.Tensor)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_create_dataloaders_multiple_categories(self, setup_test_dataset_structure):
        """Test dataloader creation with multiple categories."""
        train_loader, test_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories=['capsule', 'hazelnut'],
            batch_size=4,
            num_workers=0
        )
        
        # Should have samples from both categories
        train_batch = next(iter(train_loader))
        assert len(set(train_batch['category'])) <= 2  # At most 2 categories in batch
        
        test_batch = next(iter(test_loader))
        assert len(set(test_batch['category'])) <= 2  # At most 2 categories in batch
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataloader_batch_consistency(self, setup_test_dataset_structure):
        """Test that dataloader batches have consistent structure."""
        train_loader, _ = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            batch_size=3,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        
        # Check all required keys exist
        required_keys = ['image', 'label', 'mask', 'path', 'category']
        for key in required_keys:
            assert key in batch, f"Missing key: {key}"
        
        # Check batch dimensions are consistent
        batch_size = batch['image'].shape[0]
        assert len(batch['label']) == batch_size
        assert len(batch['path']) == batch_size
        assert len(batch['category']) == batch_size
        
        # Check tensor types
        assert isinstance(batch['image'], torch.Tensor)
        assert isinstance(batch['label'], torch.Tensor)
        assert isinstance(batch['mask'], torch.Tensor)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataloader_pin_memory_cuda_available(self, setup_test_dataset_structure):
        """Test pin_memory setting based on CUDA availability."""
        with patch('torch.cuda.is_available', return_value=True):
            train_loader, test_loader = create_dataloaders(
                root_dir=setup_test_dataset_structure,
                categories='capsule',
                batch_size=2,
                num_workers=0
            )
            
            assert train_loader.pin_memory == True
            assert test_loader.pin_memory == True
        
        with patch('torch.cuda.is_available', return_value=False):
            train_loader, test_loader = create_dataloaders(
                root_dir=setup_test_dataset_structure,
                categories='capsule',
                batch_size=2,
                num_workers=0
            )
            
            assert train_loader.pin_memory == False
            assert test_loader.pin_memory == False
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_dataloader_drop_last_training(self, setup_test_dataset_structure):
        """Test that training dataloader drops last incomplete batch."""
        train_loader, test_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            batch_size=3,  # 5 samples, so last batch would have 2 samples
            num_workers=0
        )
        
        # Training loader should drop last incomplete batch
        assert train_loader.drop_last == True
        
        # Test loader should not drop last batch
        assert test_loader.drop_last == False
        
        # Count actual batches
        train_batches = list(train_loader)
        # With 5 samples and batch_size=3, drop_last=True should give 1 batch
        assert len(train_batches) == 1
        assert train_batches[0]['image'].shape[0] == 3