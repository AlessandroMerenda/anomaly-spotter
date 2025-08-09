"""
Unit tests for core model (AutoencoderUNetLite).
Tests architecture, forward pass, input validation, and output properties.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, Mock

from src.core.model import AutoencoderUNetLite, CONV_KERNEL_SIZE, POOL_KERNEL_SIZE
from tests.conftest import assert_tensor_properties, assert_model_output_valid, TestCategories


class TestAutorencoderUNetLite:
    """Test suite for AutoencoderUNetLite model."""
    
    @pytest.mark.unit
    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = AutoencoderUNetLite()
        
        assert model.input_channels == 3
        assert model.output_channels == 3
        assert isinstance(model.encoder_block_1, nn.Sequential)
        assert isinstance(model.bottleneck, nn.Sequential)
        assert isinstance(model.final_conv, nn.Conv2d)
        assert isinstance(model.output_activation, nn.Tanh)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_ch, output_ch", [
        (1, 1),  # Grayscale
        (3, 3),  # RGB
        (3, 1),  # RGB to grayscale
        (4, 3),  # RGBA to RGB
    ])
    def test_model_initialization_custom_channels(self, input_ch, output_ch):
        """Test model initialization with custom channel configurations."""
        model = AutoencoderUNetLite(input_channels=input_ch, output_channels=output_ch)
        
        assert model.input_channels == input_ch
        assert model.output_channels == output_ch
        
        # Check first conv layer has correct input channels
        first_conv = model.encoder_block_1[0]
        assert first_conv.in_channels == input_ch
        
        # Check final conv layer has correct output channels
        assert model.final_conv.out_channels == output_ch
    
    @pytest.mark.unit
    def test_model_initialization_invalid_channels(self):
        """Test model initialization with invalid channel configurations."""
        with pytest.raises(ValueError, match="must be positive"):
            AutoencoderUNetLite(input_channels=0, output_channels=3)
            
        with pytest.raises(ValueError, match="must be positive"):
            AutoencoderUNetLite(input_channels=3, output_channels=-1)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_forward_pass_shape_consistency(self, batch_size):
        """Test forward pass maintains input-output shape consistency."""
        model = AutoencoderUNetLite()
        input_tensor = torch.randn(batch_size, 3, 128, 128)
        
        output = assert_model_output_valid(model, input_tensor)
        
        # Output shape should match input shape exactly
        assert output.shape == input_tensor.shape
    
    @pytest.mark.unit
    def test_forward_pass_output_range(self):
        """Test forward pass output is in correct range [-1, 1]."""
        model = AutoencoderUNetLite()
        input_tensor = torch.randn(2, 3, 128, 128)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output range is [-1, 1] due to Tanh activation
        assert_tensor_properties(
            output, 
            expected_shape=(2, 3, 128, 128),
            expected_range=(-1.0, 1.0)
        )
        
        # More strict range check for Tanh
        assert output.min().item() >= -1.0
        assert output.max().item() <= 1.0
    
    @pytest.mark.unit
    def test_forward_pass_input_validation(self):
        """Test forward pass validates input correctly."""
        model = AutoencoderUNetLite(input_channels=3)
        
        # Test wrong number of dimensions
        with pytest.raises(RuntimeError, match="Expected 4D input tensor"):
            model(torch.randn(3, 128, 128))  # Missing batch dimension
            
        # Test wrong number of channels
        with pytest.raises(RuntimeError, match="Expected 3 input channels"):
            model(torch.randn(1, 1, 128, 128))  # Wrong channels
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_size", [
        (64, 64),
        (128, 128),  
        (256, 256),
    ])
    def test_forward_pass_different_sizes(self, input_size):
        """Test forward pass with different input sizes."""
        model = AutoencoderUNetLite()
        h, w = input_size
        input_tensor = torch.randn(1, 3, h, w)
        
        output = assert_model_output_valid(model, input_tensor)
        assert output.shape == (1, 3, h, w)
    
    @pytest.mark.unit
    def test_forward_pass_gradient_flow(self):
        """Test gradients flow properly through the model."""
        model = AutoencoderUNetLite()
        input_tensor = torch.randn(1, 3, 128, 128, requires_grad=True)
        
        # Forward pass
        output = model(input_tensor)
        loss = output.sum()  # Simple loss for gradient computation
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None, "Gradient is None for model parameter"
            assert not torch.isnan(param.grad).any(), "NaN gradient detected"
    
    @pytest.mark.unit
    def test_forward_pass_no_gradient_in_eval(self):
        """Test no gradients are computed in evaluation mode."""
        model = AutoencoderUNetLite()
        model.eval()
        
        input_tensor = torch.randn(1, 3, 128, 128, requires_grad=True)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Output should not require gradients
        assert not output.requires_grad
    
    @pytest.mark.unit
    def test_model_parameters_count(self):
        """Test model has reasonable number of parameters."""
        model = AutoencoderUNetLite()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have substantial number of parameters (U-Net is not tiny)
        assert total_params > 1_000_000, f"Model has only {total_params} parameters, seems too small"
        assert total_params < 100_000_000, f"Model has {total_params} parameters, seems too large"
        
        # All parameters should be trainable by default
        assert trainable_params == total_params
    
    @pytest.mark.unit
    def test_model_device_consistency(self, device):
        """Test model works correctly on different devices."""
        model = AutoencoderUNetLite().to(device)
        input_tensor = torch.randn(1, 3, 128, 128).to(device)
        
        output = model(input_tensor)
        
        assert output.device == torch.device(device)
        assert output.shape == (1, 3, 128, 128)
    
    @pytest.mark.unit
    def test_model_info_method(self):
        """Test get_model_info() method returns correct information."""
        model = AutoencoderUNetLite(input_channels=3, output_channels=3)
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert info['model_name'] == 'AutoencoderUNetLite'
        assert info['input_channels'] == 3
        assert info['output_channels'] == 3
        assert info['input_size'] == '128x128'
        assert info['output_range'] == '[-1, 1]'
        assert isinstance(info['parameters'], int)
        assert isinstance(info['trainable_parameters'], int)
        assert info['parameters'] > 0
        assert info['trainable_parameters'] == info['parameters']  # All should be trainable
    
    @pytest.mark.unit
    def test_skip_connections_preserve_information(self):
        """Test that skip connections preserve spatial information."""
        model = AutoencoderUNetLite()
        
        # Create input with distinct patterns at different scales
        input_tensor = torch.zeros(1, 3, 128, 128)
        
        # Add patterns at different locations that should be preserved by skip connections
        input_tensor[0, :, 10:20, 10:20] = 1.0  # High-res detail
        input_tensor[0, :, 60:80, 60:80] = -1.0  # Mid-res detail
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # Output should preserve some spatial structure (not be completely uniform)
        output_std = output.std().item()
        assert output_std > 0.01, "Output has too little variation, skip connections may not be working"
    
    @pytest.mark.unit
    def test_model_deterministic_output(self):
        """Test model produces consistent output for same input."""
        model = AutoencoderUNetLite()
        model.eval()
        
        input_tensor = torch.randn(1, 3, 128, 128)
        
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
        
        # Outputs should be identical for same input
        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6)
    
    @pytest.mark.unit
    def test_model_state_dict_serialization(self):
        """Test model state dict can be saved and loaded."""
        model1 = AutoencoderUNetLite()
        
        # Get state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state
        model2 = AutoencoderUNetLite()
        model2.load_state_dict(state_dict)
        
        # Models should produce identical outputs
        input_tensor = torch.randn(1, 3, 128, 128)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(input_tensor)
            output2 = model2(input_tensor)
        
        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.slow
    def test_model_memory_usage(self):
        """Test model memory usage is reasonable."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
            
        model = AutoencoderUNetLite().cuda()
        input_tensor = torch.randn(4, 3, 128, 128).cuda()  # Reasonable batch size
        
        # Measure memory before forward pass
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
        
        # Forward pass
        output = model(input_tensor)
        
        memory_after = torch.cuda.memory_allocated()
        memory_used = memory_after - memory_before
        
        # Should use reasonable amount of memory (less than 2GB for this input)
        memory_gb = memory_used / (1024**3)
        assert memory_gb < 2.0, f"Model uses too much memory: {memory_gb:.2f}GB"
        
        torch.cuda.empty_cache()
    
    @pytest.mark.unit  
    def test_architecture_constants(self):
        """Test that architecture constants are used correctly."""
        model = AutoencoderUNetLite()
        
        # Check first encoder conv uses correct kernel size
        first_conv = model.encoder_block_1[0]
        assert first_conv.kernel_size == (CONV_KERNEL_SIZE, CONV_KERNEL_SIZE)
        
        # Check pooling uses correct kernel size
        assert model.pool1.kernel_size == POOL_KERNEL_SIZE
        
        # Check upconv uses correct kernel size
        assert model.upconv1.kernel_size == (2, 2)  # UPSAMPLE_KERNEL