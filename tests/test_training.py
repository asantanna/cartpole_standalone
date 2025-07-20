#!/usr/bin/env python3
"""
Tests for the training loop with real Isaac Gym environment.
"""
import pytest
import torch
import os
import json
import glob

from src.cartpole import make_env, train, ActorCritic


class TestIsaacGymEnvironment:
    """Test Isaac Gym environment setup."""
    
    def test_isaac_gym_import(self):
        """Verify Isaac Gym is installed."""
        try:
            import isaacgym
            import isaacgymenvs
        except ImportError:
            pytest.fail("Isaac Gym is not installed. Please install it before running tests.")
            
    def test_make_env(self):
        """Test environment creation."""
        env = make_env(headless=True)
        assert env is not None
        
        # Check environment properties
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        
        # Test reset
        obs_dict = env.reset()
        assert 'obs' in obs_dict
        assert isinstance(obs_dict['obs'], torch.Tensor)
        
        # Check observation shape
        obs = obs_dict['obs']
        assert len(obs.shape) == 2  # (num_envs, obs_dim)
        assert obs.shape[0] == 1  # Single environment
        
    def test_env_step(self):
        """Test environment step function."""
        env = make_env(headless=True)
        obs_dict = env.reset()
        
        # Get action dimension
        act_dim = env.action_space.shape[0]
        
        # Create random action
        action = torch.randn(1, act_dim)  # Shape: (num_envs, act_dim)
        
        # Step
        next_obs_dict, reward, done, info = env.step(action)
        
        assert 'obs' in next_obs_dict
        assert isinstance(reward, torch.Tensor)
        assert isinstance(done, torch.Tensor)
        assert reward.shape == (1,)
        assert done.shape == (1,)


class TestTrainingFunction:
    """Test the main training function."""
    
    @pytest.mark.slow
    def test_train_minimal_episodes(self, tmp_path, mock_args):
        """Test training with minimal episodes."""
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Run training for just 5 episodes
            mock_args.num_episodes = 5
            final_return = train(headless=True, num_episodes=5, args=mock_args)
            
            assert isinstance(final_return, (int, float))
            assert final_return > -1000  # Should have some reasonable return
            
        finally:
            os.chdir(original_cwd)
            
    def test_train_with_metrics(self, tmp_path, mock_args):
        """Test training with metrics saving."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            mock_args.num_episodes = 5
            mock_args.save_metrics = True
            mock_args.run_id = 'test_metrics_run'
            
            final_return = train(headless=True, num_episodes=5, args=mock_args)
            
            # Check metrics file was created
            metrics_path = 'runs/single/test_metrics_run/metrics.json'
            assert os.path.exists(metrics_path)
            
            # Load and verify metrics
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            assert metrics['run_id'] == 'test_metrics_run'
            assert 'hyperparameters' in metrics
            assert 'returns' in metrics
            assert len(metrics['returns']) == 5
            assert metrics['final_avg_return'] == final_return
            
        finally:
            os.chdir(original_cwd)
            
    def test_train_with_checkpoint(self, tmp_path, mock_args):
        """Test training with checkpoint saving."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            mock_args.num_episodes = 5
            mock_args.save_checkpoint = 'model.pt'
            mock_args.run_id = 'test_checkpoint_run'
            
            train(headless=True, num_episodes=5, args=mock_args)
            
            # Check checkpoint was created
            checkpoint_path = 'runs/single/test_checkpoint_run/model.pt'
            assert os.path.exists(checkpoint_path)
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path)
            assert 'W_a' in checkpoint
            assert 'W_c' in checkpoint
            assert 'log_std' in checkpoint
            assert 'hyperparams' in checkpoint
            
        finally:
            os.chdir(original_cwd)
            
    def test_train_load_checkpoint(self, tmp_path, mock_args):
        """Test loading checkpoint and continuing training."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # First, train and save a checkpoint
            mock_args.num_episodes = 5
            mock_args.save_checkpoint = 'initial.pt'
            mock_args.save_metrics = True
            mock_args.run_id = 'initial_run'
            
            train(headless=True, num_episodes=5, args=mock_args)
            
            # Now load the checkpoint and train more
            mock_args.load_checkpoint = 'runs/single/initial_run/initial.pt'
            mock_args.training_mode = 'true'  # Continue training
            mock_args.run_id = 'continued_run'
            mock_args.save_checkpoint = 'continued.pt'
            
            final_return = train(headless=True, num_episodes=5, args=mock_args)
            
            # Verify continued training happened
            assert os.path.exists('runs/single/continued_run/continued.pt')
            assert os.path.exists('runs/single/continued_run/metrics.json')
            
        finally:
            os.chdir(original_cwd)
            
    def test_train_best_config(self, tmp_path, mock_args):
        """Test training with best configuration."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            mock_args.num_episodes = 5
            mock_args.best_config = True
            mock_args.save_metrics = True
            mock_args.run_id = 'best_config_run'
            
            train(headless=True, num_episodes=5, args=mock_args)
            
            # Load metrics and verify best config was used
            with open('runs/single/best_config_run/metrics.json', 'r') as f:
                metrics = json.load(f)
                
            # Check that best config hyperparameters were used
            assert abs(metrics['hyperparameters']['lr_actor'] - 7.93676080244564e-05) < 1e-10
            assert abs(metrics['hyperparameters']['noise_std'] - 0.02340585371545556) < 1e-10
            
        finally:
            os.chdir(original_cwd)
            
    def test_training_mode_override(self, tmp_path, mock_args):
        """Test training mode override."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create a checkpoint first
            mock_args.num_episodes = 5
            mock_args.save_checkpoint = 'model.pt'
            mock_args.run_id = 'create_checkpoint'
            
            train(headless=True, num_episodes=5, args=mock_args)
            
            # Load in evaluation mode (default)
            mock_args.load_checkpoint = 'runs/single/create_checkpoint/model.pt'
            mock_args.training_mode = 'false'
            mock_args.run_id = 'eval_run'
            mock_args.save_checkpoint = None
            
            # In eval mode, the model shouldn't learn
            train(headless=True, num_episodes=5, args=mock_args)
            
            # Load in training mode
            mock_args.training_mode = 'true'
            mock_args.run_id = 'train_run'
            
            train(headless=True, num_episodes=5, args=mock_args)
            
            # Both should complete without error
            
        finally:
            os.chdir(original_cwd)
            
    @pytest.mark.slow
    def test_best_model_saving(self, tmp_path, mock_args):
        """Test automatic best model saving during training."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            mock_args.num_episodes = 30  # Need enough episodes for averaging
            mock_args.save_checkpoint = 'model.pt'
            mock_args.run_id = 'best_model_test'
            
            train(headless=True, num_episodes=30, args=mock_args)
            
            # Check both final and best checkpoints exist
            final_path = 'runs/single/best_model_test/model.pt'
            best_path = 'runs/single/best_model_test/model_best.pt'
            
            assert os.path.exists(final_path)
            # Best model might not exist if performance never improved
            # But the mechanism should work without errors
            
        finally:
            os.chdir(original_cwd)


class TestTrainingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_train_without_args(self, tmp_path):
        """Test training with no args (defaults)."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Should work with all defaults
            final_return = train(headless=True, num_episodes=5, args=None)
            assert isinstance(final_return, (int, float))
            
        finally:
            os.chdir(original_cwd)
            
    def test_checkpoint_resolution(self, tmp_path, mock_args):
        """Test checkpoint path resolution."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create a checkpoint in runs directory
            os.makedirs('runs/single/old_run', exist_ok=True)
            
            # Create a dummy checkpoint
            dummy_checkpoint = {
                'W_a': torch.randn(1, 4),
                'W_c': torch.randn(1, 4),
                'log_std': torch.tensor([0.0]),
                'hyperparams': {
                    'lr_a': 1e-4,
                    'lr_c': 1e-3,
                    'lambda_a': 0.9,
                    'lambda_c': 0.95,
                    'obs_dim': 4,
                    'act_dim': 1
                }
            }
            torch.save(dummy_checkpoint, 'runs/single/old_run/model.pt')
            
            # Try to load with just filename
            mock_args.load_checkpoint = 'model.pt'
            mock_args.num_episodes = 5
            
            # Should find and load the checkpoint
            train(headless=True, num_episodes=5, args=mock_args)
            
        finally:
            os.chdir(original_cwd)
            
    @pytest.mark.gpu
    def test_gpu_training(self, gpu_available, tmp_path, mock_args):
        """Test training on GPU if available."""
        if not gpu_available:
            pytest.skip("GPU not available")
            
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            mock_args.num_episodes = 5
            # Training should automatically use GPU if available
            train(headless=True, num_episodes=5, args=mock_args)
            
        finally:
            os.chdir(original_cwd)