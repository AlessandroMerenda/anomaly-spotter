"""
Integration tests for training pipeline.
Tests end-to-end training workflow, component interactions, and training stability.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from src.core.model import AutoencoderUNetLite
from src.core.model_config import AutoencoderConfig
from src.core.losses import CombinedLoss, create_loss_function
from src.training.trainer import AnomalyDetectorTrainer, create_trainer
from src.data.loaders import create_dataloaders
from src.data.preprocessing import MVTecPreprocessor
from tests.conftest import assert_tensor_properties, TestCategories


class TestTrainingPipelineIntegration:
    """Test suite for complete training pipeline integration."""
    
    @pytest.fixture
    def minimal_config(self):
        """Create minimal config for fast integration tests."""
        config = AutoencoderConfig()
        # Minimal settings for fast tests
        config.batch_size = 2
        config.num_epochs = 2
        config.learning_rate = 1e-3
        config.input_size = (64, 64)  # Smaller for speed
        config.val_split = 0.5
        config.use_amp = False  # Disable AMP for consistent testing
        config.num_workers = 0  # No multiprocessing in tests
        config.log_level = "ERROR"  # Reduce test noise
        config.early_stopping = False  # Don't stop early in tests
        config.save_checkpoint_every = 1  # Save every epoch for testing
        return config
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp(prefix="training_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_training_workflow(self, setup_test_dataset_structure, minimal_config, temp_output_dir, device):
        """Test complete training workflow from data loading to model saving."""
        # Create model
        model = AutoencoderUNetLite(input_channels=3, output_channels=3)
        
        # Create preprocessor
        preprocessor = MVTecPreprocessor(
            image_size=minimal_config.input_size,
            normalization_type='tanh'
        )
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=minimal_config.batch_size,
            num_workers=0
        )
        
        # Create trainer
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=minimal_config,
            device=device,
            log_dir=temp_output_dir / "logs"
        )
        
        # Run training
        results = trainer.train(train_loader, val_loader, temp_output_dir)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'best_loss' in results
        assert 'final_epoch' in results
        assert 'thresholds' in results
        assert 'training_history' in results
        
        # Verify training history
        history = results['training_history']
        assert len(history['train_loss']) == minimal_config.num_epochs
        assert len(history['val_loss']) == minimal_config.num_epochs
        
        # Verify model was saved
        assert (temp_output_dir / 'model.pth').exists()
        assert (temp_output_dir / 'thresholds.json').exists()
        assert (temp_output_dir / 'training_history.json').exists()
    
    @pytest.mark.integration
    def test_trainer_initialization(self, minimal_config, device):
        """Test trainer initialization with various configurations."""
        model = AutoencoderUNetLite()
        
        # Test successful initialization
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=minimal_config,
            device=device
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None
        assert trainer.device == device
    
    @pytest.mark.integration
    def test_training_step_integration(self, setup_test_dataset_structure, minimal_config, device):
        """Test single training step integration."""
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        train_loader, _ = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=2,
            num_workers=0
        )
        
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=minimal_config,
            device=device
        )
        
        # Test single training step
        avg_loss, loss_components = trainer.train_epoch(train_loader, epoch=0)
        
        assert isinstance(avg_loss, float)
        assert avg_loss > 0  # Loss should be positive
        assert isinstance(loss_components, dict)
        assert 'total' in loss_components
    
    @pytest.mark.integration
    def test_validation_step_integration(self, setup_test_dataset_structure, minimal_config, device):
        """Test validation step integration."""
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        _, val_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=2,
            num_workers=0
        )
        
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=minimal_config,
            device=device
        )
        
        # Test validation step
        avg_loss, reconstruction_errors = trainer.validate(val_loader, epoch=0)
        
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert isinstance(reconstruction_errors, list)
        assert len(reconstruction_errors) > 0
    
    @pytest.mark.integration
    def test_threshold_calculation(self, setup_test_dataset_structure, minimal_config, device):
        """Test threshold calculation integration."""
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        train_loader, _ = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=2,
            num_workers=0
        )
        
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=minimal_config,
            device=device
        )
        
        # Calculate thresholds
        thresholds = trainer.calculate_threshold(train_loader)
        
        assert isinstance(thresholds, dict)
        assert len(thresholds) > 0
        
        # Check expected threshold types
        expected_keys = ['percentile_99', 'percentile_95', 'mean', 'std']
        for key in expected_keys:
            assert key in thresholds
            assert isinstance(thresholds[key], float)
            assert thresholds[key] > 0
    
    @pytest.mark.integration
    def test_checkpoint_save_load(self, setup_test_dataset_structure, minimal_config, temp_output_dir, device):
        """Test checkpoint saving and loading."""
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        train_loader, val_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=2,
            num_workers=0
        )
        
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=minimal_config,
            device=device
        )
        
        # Train for one epoch to get some state
        trainer.train_epoch(train_loader, epoch=0)
        val_loss, _ = trainer.validate(val_loader, epoch=0)
        
        # Save checkpoint
        trainer.save_checkpoint(epoch=0, loss=val_loss, save_dir=temp_output_dir)
        
        # Verify checkpoint files exist
        assert (temp_output_dir / 'checkpoint_latest.pth').exists()
        
        # Create new trainer and load checkpoint
        model2 = AutoencoderUNetLite()
        trainer2 = AnomalyDetectorTrainer(
            model=model2,
            config=minimal_config,
            device=device
        )
        
        # Load checkpoint
        loaded_epoch = trainer2.load_checkpoint(temp_output_dir / 'checkpoint_latest.pth')
        assert loaded_epoch == 0
    
    @pytest.mark.integration
    def test_mixed_precision_training(self, setup_test_dataset_structure, device):
        """Test training with mixed precision (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")
            
        config = AutoencoderConfig()
        config.batch_size = 2
        config.num_epochs = 1
        config.use_amp = True  # Enable AMP
        config.num_workers = 0
        
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        train_loader, _ = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=config.batch_size,
            num_workers=0
        )
        
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=config,
            device=device
        )
        
        # Should initialize scaler for AMP
        assert trainer.scaler is not None
        assert trainer.use_amp is True
        
        # Test training step with AMP
        avg_loss, _ = trainer.train_epoch(train_loader, epoch=0)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
    
    @pytest.mark.integration
    def test_loss_function_integration(self, minimal_config):
        """Test different loss function configurations."""
        loss_types = ['mse', 'l1', 'combined']
        
        for loss_type in loss_types:
            config = minimal_config
            config.loss_type = loss_type
            
            # Test loss function creation
            loss_fn = create_loss_function(config)
            assert loss_fn is not None
            
            # Test loss computation
            pred = torch.randn(2, 3, 64, 64)
            target = torch.randn(2, 3, 64, 64)
            
            result = loss_fn(pred, target)
            
            # Check result format
            if isinstance(result, tuple):
                loss_value, loss_dict = result
                assert isinstance(loss_value, torch.Tensor)
                assert isinstance(loss_dict, dict)
                assert loss_value.dim() == 0  # Scalar
            else:
                assert isinstance(result, torch.Tensor)
                assert result.dim() == 0  # Scalar
    
    @pytest.mark.integration
    def test_data_preprocessing_integration(self, setup_test_dataset_structure):
        """Test data preprocessing pipeline integration."""
        # Test different preprocessing configurations
        configs = [
            {'image_size': (64, 64), 'normalization_type': 'tanh'},
            {'image_size': (128, 128), 'normalization_type': 'imagenet'},
        ]
        
        for config in configs:
            preprocessor = MVTecPreprocessor(**config)
            
            train_loader, test_loader = create_dataloaders(
                root_dir=setup_test_dataset_structure,
                categories='capsule',
                preprocessor=preprocessor,
                batch_size=2,
                num_workers=0
            )
            
            # Test train loader batch
            train_batch = next(iter(train_loader))
            assert_tensor_properties(
                train_batch['image'],
                expected_shape=(2, 3, config['image_size'][0], config['image_size'][1]),
                expected_dtype=torch.float32
            )
            
            # Test test loader batch
            test_batch = next(iter(test_loader))
            assert test_batch['image'].shape[1:] == (3, config['image_size'][0], config['image_size'][1])
    
    @pytest.mark.integration
    def test_training_stability(self, setup_test_dataset_structure, minimal_config, device):
        """Test training stability and convergence."""
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        train_loader, val_loader = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=2,
            num_workers=0
        )
        
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=minimal_config,
            device=device
        )
        
        # Record initial state
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Train for multiple epochs
        train_losses = []
        for epoch in range(3):
            avg_loss, _ = trainer.train_epoch(train_loader, epoch)
            train_losses.append(avg_loss)
        
        # Check that parameters have changed (model is learning)
        for name, param in model.named_parameters():
            if name in initial_params:
                assert not torch.allclose(param, initial_params[name], atol=1e-6), \
                    f"Parameter {name} hasn't changed during training"
        
        # Check that loss is finite and positive
        for loss in train_losses:
            assert torch.isfinite(torch.tensor(loss)), "Training loss is not finite"
            assert loss > 0, "Training loss is not positive"
        
        # Loss should generally decrease (though not strictly required for few epochs)
        assert train_losses[-1] < train_losses[0] * 2, "Training loss increased too much"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_overfitting_behavior(self, setup_test_dataset_structure, device):
        """Test that model can overfit to small dataset (sanity check)."""
        # Create config that should allow overfitting
        config = AutoencoderConfig()
        config.batch_size = 1  # Small batch
        config.num_epochs = 10  # More epochs
        config.learning_rate = 1e-3  # Higher learning rate
        config.input_size = (64, 64)
        config.num_workers = 0
        config.early_stopping = False
        
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        # Use very small dataset (single category, small batch)
        train_loader, _ = create_dataloaders(
            root_dir=setup_test_dataset_structure,
            categories='capsule',
            preprocessor=preprocessor,
            batch_size=1,
            num_workers=0
        )
        
        trainer = AnomalyDetectorTrainer(
            model=model,
            config=config,
            device=device
        )
        
        # Train and check loss decreases significantly
        initial_loss, _ = trainer.train_epoch(train_loader, epoch=0)
        
        # Train for several epochs
        for epoch in range(1, 5):
            trainer.train_epoch(train_loader, epoch)
        
        final_loss, _ = trainer.train_epoch(train_loader, epoch=5)
        
        # Loss should decrease significantly if model can learn
        assert final_loss < initial_loss * 0.8, \
            f"Model failed to overfit: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"


class TestTrainingErrorHandling:
    """Test error handling in training pipeline."""
    
    @pytest.mark.integration
    def test_corrupted_data_handling(self, temp_data_dir, minimal_config, device):
        """Test handling of corrupted data during training."""
        # Create minimal dataset structure with corrupted file
        category_dir = temp_data_dir / 'test_category' / 'train' / 'good'
        category_dir.mkdir(parents=True)
        
        # Create one valid image and one corrupted file
        from PIL import Image
        import numpy as np
        
        valid_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(valid_img).save(category_dir / 'valid.png')
        
        # Create corrupted file
        (category_dir / 'corrupted.png').write_bytes(b'not an image')
        
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=(64, 64))
        
        # This should fail during data loading
        with pytest.raises(RuntimeError):
            train_loader, _ = create_dataloaders(
                root_dir=temp_data_dir,
                categories='test_category',
                preprocessor=preprocessor,
                batch_size=2,
                num_workers=0
            )
            
            trainer = AnomalyDetectorTrainer(
                model=model,
                config=minimal_config,
                device=device
            )
            
            # This should raise error when encountering corrupted data
            trainer.train_epoch(train_loader, epoch=0)
    
    @pytest.mark.integration
    def test_out_of_memory_handling(self, setup_test_dataset_structure, device):
        """Test handling of out-of-memory conditions."""
        if device == 'cpu':
            pytest.skip("OOM testing only relevant for GPU")
        
        # Create config with unreasonably large batch size to trigger OOM
        config = AutoencoderConfig()
        config.batch_size = 1000  # Very large batch
        config.input_size = (512, 512)  # Large images
        config.num_epochs = 1
        config.num_workers = 0
        
        model = AutoencoderUNetLite()
        preprocessor = MVTecPreprocessor(image_size=config.input_size)
        
        # This might trigger OOM
        try:
            train_loader, _ = create_dataloaders(
                root_dir=setup_test_dataset_structure,
                categories='capsule',
                preprocessor=preprocessor,
                batch_size=config.batch_size,
                num_workers=0
            )
            
            trainer = AnomalyDetectorTrainer(
                model=model,
                config=config,
                device=device
            )
            
            trainer.train_epoch(train_loader, epoch=0)
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Expected for large batch sizes
            assert "out of memory" in str(e).lower() or "cuda" in str(e).lower()
    
    @pytest.mark.integration
    def test_invalid_config_handling(self, device):
        """Test handling of invalid configurations."""
        model = AutoencoderUNetLite()
        
        # Test invalid learning rate
        config = AutoencoderConfig()
        config.learning_rate = -1  # Invalid
        
        with pytest.raises((ValueError, AssertionError)):
            config._validate_config()
        
        # Test invalid batch size
        config = AutoencoderConfig()
        config.batch_size = 0  # Invalid
        
        with pytest.raises((ValueError, AssertionError)):
            config._validate_config()