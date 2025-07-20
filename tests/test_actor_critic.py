#!/usr/bin/env python3
"""
Tests for the ActorCritic class in cartpole.py
"""
import pytest
import torch
import numpy as np
import os

from src.cartpole import ActorCritic


class TestActorCriticInit:
    """Test ActorCritic initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim)
        
        # Check dimensions
        assert ac.W_a.shape == (act_dim, obs_dim)
        assert ac.W_c.shape == (1, obs_dim)
        assert ac.log_std.shape == (act_dim,)
        
        # Check eligibility traces
        assert ac.e_a.shape == ac.W_a.shape
        assert ac.e_c.shape == ac.W_c.shape
        assert torch.allclose(ac.e_a, torch.zeros_like(ac.W_a))
        assert torch.allclose(ac.e_c, torch.zeros_like(ac.W_c))
        
        # Check hyperparameters
        assert ac.lr_a == 8e-5
        assert ac.lr_c == 2.5e-4
        assert ac.lambda_a == 0.88
        assert ac.lambda_c == 0.92
        assert ac.training_mode == True
        
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        obs_dim = 8
        act_dim = 2
        custom_params = {
            'lr_a': 1e-4,
            'lr_c': 5e-4,
            'lambda_a': 0.9,
            'lambda_c': 0.95,
            'noise_std': 0.1,
            'device': 'cpu'
        }
        
        ac = ActorCritic(obs_dim, act_dim, **custom_params)
        
        assert ac.W_a.shape == (act_dim, obs_dim)
        assert ac.lr_a == custom_params['lr_a']
        assert ac.lr_c == custom_params['lr_c']
        assert ac.lambda_a == custom_params['lambda_a']
        assert ac.lambda_c == custom_params['lambda_c']
        assert torch.allclose(ac.log_std, torch.full((act_dim,), np.log(custom_params['noise_std'])))
        
    @pytest.mark.gpu
    def test_gpu_initialization(self, gpu_available):
        """Test initialization on GPU if available."""
        if not gpu_available:
            pytest.skip("GPU not available")
            
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cuda:0')
        
        assert ac.W_a.is_cuda
        assert ac.W_c.is_cuda
        assert ac.log_std.is_cuda
        assert ac.e_a.is_cuda
        assert ac.e_c.is_cuda


class TestActorCriticPolicy:
    """Test policy and action sampling."""
    
    def test_policy_output(self):
        """Test policy returns correct mean and std."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        
        # Create a test state
        state = torch.randn(obs_dim)
        mean, std = ac.policy(state)
        
        # Check shapes
        assert mean.shape == (act_dim,)
        assert std.shape == (act_dim,)
        
        # Check mean is bounded by tanh
        assert torch.all(mean >= -1.0)
        assert torch.all(mean <= 1.0)
        
        # Check std is positive
        assert torch.all(std > 0)
        
    def test_sample_action(self):
        """Test action sampling is bounded."""
        obs_dim = 4
        act_dim = 2
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        
        # Sample many actions
        state = torch.randn(obs_dim)
        mean, std = ac.policy(state)
        
        for _ in range(100):
            action = ac.sample_action(mean, std)
            assert action.shape == (act_dim,)
            assert torch.all(action >= -1.0)
            assert torch.all(action <= 1.0)
            
    def test_deterministic_policy(self):
        """Test policy with very low noise."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, noise_std=1e-6, device='cpu')
        
        state = torch.randn(obs_dim)
        mean, std = ac.policy(state)
        
        # Multiple samples should be very close to mean
        actions = [ac.sample_action(mean, std) for _ in range(10)]
        actions = torch.stack(actions)
        
        assert torch.allclose(actions, mean.unsqueeze(0).repeat(10, 1), atol=1e-4)


class TestActorCriticValue:
    """Test value function."""
    
    def test_value_output(self):
        """Test value function returns scalar."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        
        state = torch.randn(obs_dim)
        value = ac.value(state)
        
        assert value.shape == ()  # Scalar
        assert value.dtype == torch.float32
        
    def test_value_consistency(self):
        """Test value function is consistent for same input."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        
        state = torch.randn(obs_dim)
        value1 = ac.value(state)
        value2 = ac.value(state)
        
        assert torch.allclose(value1, value2)


class TestActorCriticTraces:
    """Test eligibility traces."""
    
    def test_reset_traces(self):
        """Test trace resetting."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        
        # Add some values to traces
        ac.e_a += torch.ones_like(ac.e_a)
        ac.e_c += torch.ones_like(ac.e_c)
        
        # Reset
        ac.reset_traces()
        
        assert torch.allclose(ac.e_a, torch.zeros_like(ac.e_a))
        assert torch.allclose(ac.e_c, torch.zeros_like(ac.e_c))
        
    def test_update_traces_training_mode(self):
        """Test trace updates in training mode."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        ac.training_mode = True
        
        state = torch.randn(obs_dim)
        action = torch.randn(act_dim)
        
        # Store original traces
        e_a_before = ac.e_a.clone()
        e_c_before = ac.e_c.clone()
        
        ac.update_traces(state, action)
        
        # Traces should have changed
        assert not torch.allclose(ac.e_a, e_a_before)
        assert not torch.allclose(ac.e_c, e_c_before)
        
    def test_update_traces_eval_mode(self):
        """Test trace updates in evaluation mode."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        ac.training_mode = False
        
        state = torch.randn(obs_dim)
        action = torch.randn(act_dim)
        
        # Store original traces
        e_a_before = ac.e_a.clone()
        e_c_before = ac.e_c.clone()
        
        ac.update_traces(state, action)
        
        # Traces should NOT have changed
        assert torch.allclose(ac.e_a, e_a_before)
        assert torch.allclose(ac.e_c, e_c_before)
        
    def test_trace_decay(self):
        """Test that traces decay correctly."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, lambda_a=0.9, lambda_c=0.95, device='cpu')
        ac.training_mode = True
        
        # Initialize traces with some values
        ac.e_a = torch.ones_like(ac.e_a)
        ac.e_c = torch.ones_like(ac.e_c)
        
        state = torch.zeros(obs_dim)
        action = torch.zeros(act_dim)
        
        # Update should decay traces
        ac.update_traces(state, action)
        
        # Check decay happened (traces should be less than 1 but not 0)
        assert torch.all(ac.e_a < 1.0)
        assert torch.all(ac.e_a >= 0.9)  # Should be at least lambda_a * 1
        assert torch.all(ac.e_c < 1.0)
        assert torch.all(ac.e_c >= 0.95)  # Should be at least lambda_c * 1


class TestActorCriticUpdates:
    """Test weight updates."""
    
    def test_apply_updates_training_mode(self):
        """Test weight updates in training mode."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        ac.training_mode = True
        
        # Store original weights
        W_a_before = ac.W_a.clone()
        W_c_before = ac.W_c.clone()
        
        # Set up some traces
        ac.e_a = torch.ones_like(ac.e_a) * 0.1
        ac.e_c = torch.ones_like(ac.e_c) * 0.1
        
        # Apply update with positive TD error
        delta = torch.tensor(1.0)
        ac.apply_updates(delta)
        
        # Weights should have changed
        assert not torch.allclose(ac.W_a, W_a_before)
        assert not torch.allclose(ac.W_c, W_c_before)
        
    def test_apply_updates_eval_mode(self):
        """Test weight updates in evaluation mode."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        ac.training_mode = False
        
        # Store original weights
        W_a_before = ac.W_a.clone()
        W_c_before = ac.W_c.clone()
        
        # Set up some traces
        ac.e_a = torch.ones_like(ac.e_a) * 0.1
        ac.e_c = torch.ones_like(ac.e_c) * 0.1
        
        # Apply update
        delta = torch.tensor(1.0)
        ac.apply_updates(delta)
        
        # Weights should NOT have changed
        assert torch.allclose(ac.W_a, W_a_before)
        assert torch.allclose(ac.W_c, W_c_before)
        
    def test_td_clipping(self):
        """Test TD error clipping."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        ac.training_mode = True
        
        # Set up traces
        ac.e_a = torch.ones_like(ac.e_a)
        ac.e_c = torch.ones_like(ac.e_c)
        
        # Store weights before
        W_a_before = ac.W_a.clone()
        
        # Apply update with very large TD error
        delta = torch.tensor(100.0)
        td_clip = 5.0
        ac.apply_updates(delta, td_clip=td_clip)
        
        # Calculate expected update (should be clipped)
        expected_update = ac.lr_a * td_clip * ac.e_a
        actual_update = ac.W_a - W_a_before
        
        assert torch.allclose(actual_update, expected_update, rtol=1e-5, atol=1e-6)


class TestActorCriticCheckpoint:
    """Test checkpoint save/load functionality."""
    
    def test_save_checkpoint(self, tmp_path):
        """Test saving checkpoint."""
        obs_dim = 4
        act_dim = 1
        ac = ActorCritic(obs_dim, act_dim, device='cpu')
        
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        ac.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Load and verify contents
        checkpoint = torch.load(checkpoint_path)
        assert 'W_a' in checkpoint
        assert 'W_c' in checkpoint
        assert 'log_std' in checkpoint
        assert 'hyperparams' in checkpoint
        
        assert checkpoint['W_a'].shape == ac.W_a.shape
        assert checkpoint['hyperparams']['obs_dim'] == obs_dim
        assert checkpoint['hyperparams']['act_dim'] == act_dim
        
    def test_load_checkpoint(self, tmp_path):
        """Test loading checkpoint."""
        obs_dim = 4
        act_dim = 1
        
        # Create and save from one instance
        ac1 = ActorCritic(obs_dim, act_dim, lr_a=1e-3, device='cpu')
        ac1.training_mode = True
        
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        ac1.save_checkpoint(str(checkpoint_path))
        
        # Load into another instance
        ac2 = ActorCritic(obs_dim, act_dim, device='cpu')
        loaded_params = ac2.load_checkpoint(str(checkpoint_path))
        
        # Verify weights are loaded correctly
        assert torch.allclose(ac2.W_a, ac1.W_a)
        assert torch.allclose(ac2.W_c, ac1.W_c)
        assert torch.allclose(ac2.log_std, ac1.log_std)
        
        # Verify traces are reset
        assert torch.allclose(ac2.e_a, torch.zeros_like(ac2.e_a))
        assert torch.allclose(ac2.e_c, torch.zeros_like(ac2.e_c))
        
        # Verify training mode is set to False
        assert ac2.training_mode == False
        
        # Verify returned hyperparams
        assert loaded_params['lr_a'] == 1e-3
        assert loaded_params['obs_dim'] == obs_dim
        assert loaded_params['act_dim'] == act_dim
        
    def test_checkpoint_device_transfer(self, tmp_path):
        """Test loading checkpoint across devices."""
        obs_dim = 4
        act_dim = 1
        
        # Save on CPU
        ac_cpu = ActorCritic(obs_dim, act_dim, device='cpu')
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        ac_cpu.save_checkpoint(str(checkpoint_path))
        
        # Load on CPU (even if GPU requested, should work)
        ac_load = ActorCritic(obs_dim, act_dim, device='cpu')
        ac_load.load_checkpoint(str(checkpoint_path))
        
        assert ac_load.W_a.device.type == 'cpu'
        assert torch.allclose(ac_load.W_a, ac_cpu.W_a)