#!/usr/bin/env python3
"""
Tests for hyperparameter search functionality.
"""
import pytest
import os
import json
import shutil
import subprocess
from unittest.mock import patch, MagicMock

from src.hyperparam_search import run_experiment, grid_search, random_search
from src.refined_search_config import get_search_config


class TestRunExperiment:
    """Test single experiment execution."""
    
    def test_run_experiment_success(self, tmp_path):
        """Test successful experiment run."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create minimal runs directory structure
            os.makedirs('runs/single/test_exp', exist_ok=True)
            
            # Create a mock metrics file that would be created by cartpole.py
            metrics = {
                'run_id': 'test_exp',
                'hyperparameters': {
                    'lr_actor': 1e-4,
                    'lr_critic': 1e-3,
                    'lambda_actor': 0.9,
                    'lambda_critic': 0.95,
                    'noise_std': 0.1,
                    'gamma': 0.99,
                    'reward_scale': 5.0,
                    'td_clip': 5.0
                },
                'returns': [10.0, 15.0, 20.0, 25.0, 30.0],
                'final_avg_return': 20.0
            }
            
            with open('runs/single/test_exp/metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            # Mock subprocess to avoid actual training
            with patch('subprocess.Popen') as mock_popen:
                mock_process = MagicMock()
                mock_process.stdout = iter(['Episode 1\tReturn 10.0\n', 'Episode 2\tReturn 15.0\n'])
                mock_process.wait.return_value = None
                mock_process.returncode = 0
                mock_popen.return_value = mock_process
                
                params = {
                    'lr_actor': 1e-4,
                    'lr_critic': 1e-3,
                    'lambda_actor': 0.9,
                    'lambda_critic': 0.95,
                    'noise_std': 0.1,
                    'gamma': 0.99,
                    'reward_scale': 5.0,
                    'td_clip': 5.0
                }
                
                score, metrics_data, metrics_file = run_experiment(
                    params, 'test_exp', num_episodes=5, show_output=True
                )
                
                assert score == 20.0
                assert metrics_data is not None
                assert metrics_file == 'runs/single/test_exp/metrics.json'
                
        finally:
            os.chdir(original_cwd)
            
    def test_run_experiment_silent(self, tmp_path):
        """Test experiment run in silent mode."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            os.makedirs('runs/single/test_silent', exist_ok=True)
            
            # Create mock metrics
            metrics = {
                'run_id': 'test_silent',
                'hyperparameters': {},
                'returns': [10.0],
                'final_avg_return': 10.0
            }
            
            with open('runs/single/test_silent/metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            # Mock subprocess.run for silent mode
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                
                params = {'lr_actor': 1e-4, 'lr_critic': 1e-3, 'lambda_actor': 0.9,
                         'lambda_critic': 0.95, 'noise_std': 0.1, 'gamma': 0.99,
                         'reward_scale': 5.0, 'td_clip': 5.0}
                
                score, _, _ = run_experiment(
                    params, 'test_silent', num_episodes=5, show_output=False
                )
                
                assert score == 10.0
                
        finally:
            os.chdir(original_cwd)


class TestGridSearch:
    """Test grid search functionality."""
    
    def test_grid_search_small(self, tmp_path):
        """Test grid search with small parameter grid."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Small parameter grid for testing
            param_grid = {
                'lr_actor': [1e-4, 2e-4],
                'lr_critic': [1e-3],
                'lambda_actor': [0.9],
                'lambda_critic': [0.95],
                'noise_std': [0.1],
                'gamma': [0.99],
                'reward_scale': [5.0, 10.0],
                'td_clip': [5.0]
            }
            
            # This would create 2 * 1 * 1 * 1 * 1 * 1 * 2 * 1 = 4 combinations
            
            # Mock run_experiment to avoid actual training
            with patch('src.hyperparam_search.run_experiment') as mock_run:
                mock_run.return_value = (50.0, {'final_avg_return': 50.0}, 'dummy_path')
                
                results, best_params, best_score, search_dir = grid_search(
                    param_grid, num_episodes=5, show_output=False, search_id='test_grid'
                )
                
                assert len(results) == 4  # All combinations
                assert best_score == 50.0
                assert best_params is not None
                assert search_dir == 'runs/search/test_grid'
                assert os.path.exists(search_dir)
                
                # Check search results file
                search_results_path = os.path.join(search_dir, 'search_results.json')
                assert os.path.exists(search_results_path)
                
                with open(search_results_path, 'r') as f:
                    saved_results = json.load(f)
                    assert saved_results['method'] == 'grid'
                    assert len(saved_results['results']) == 4
                    
        finally:
            os.chdir(original_cwd)
            
    def test_grid_search_directory_structure(self, tmp_path):
        """Test that grid search creates correct directory structure."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            param_grid = {
                'lr_actor': [1e-4],
                'lr_critic': [1e-3],
                'lambda_actor': [0.9],
                'lambda_critic': [0.95],
                'noise_std': [0.1],
                'gamma': [0.99],
                'reward_scale': [5.0],
                'td_clip': [5.0]
            }
            
            # Mock to simulate successful runs with metrics files
            def mock_run_side_effect(params, run_id, num_episodes, show_output):
                # Create a fake metrics file
                metrics_path = f'runs/single/{run_id}/metrics.json'
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                with open(metrics_path, 'w') as f:
                    json.dump({'final_avg_return': 25.0}, f)
                return 25.0, {'final_avg_return': 25.0}, metrics_path
            
            with patch('src.hyperparam_search.run_experiment', side_effect=mock_run_side_effect):
                results, _, _, search_dir = grid_search(
                    param_grid, num_episodes=5, show_output=False
                )
                
                # Check directory structure
                assert os.path.exists(search_dir)
                
                # Check individual trial directories were created
                for result in results:
                    trial_dir = os.path.join(search_dir, result['run_id'])
                    assert os.path.exists(trial_dir)
                    assert os.path.exists(os.path.join(trial_dir, 'metrics.json'))
                    
        finally:
            os.chdir(original_cwd)


class TestRandomSearch:
    """Test random search functionality."""
    
    def test_random_search_small(self, tmp_path):
        """Test random search with few trials."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            param_ranges = {
                'lr_actor': (1e-5, 1e-3),
                'lr_critic': (1e-4, 1e-2),
                'lambda_actor': (0.8, 0.99),
                'lambda_critic': (0.9, 0.99),
                'noise_std': (0.01, 0.5),
                'gamma': (0.95, 0.999),
                'reward_scale': (1.0, 20.0),
                'td_clip': (1.0, 10.0)
            }
            
            with patch('src.hyperparam_search.run_experiment') as mock_run:
                # Make different scores for each trial
                mock_run.side_effect = [
                    (30.0, {'final_avg_return': 30.0}, 'path1'),
                    (50.0, {'final_avg_return': 50.0}, 'path2'),
                    (40.0, {'final_avg_return': 40.0}, 'path3')
                ]
                
                results, best_params, best_score, search_dir = random_search(
                    param_ranges, n_trials=3, num_episodes=5, show_output=False
                )
                
                assert len(results) == 3
                assert best_score == 50.0
                assert best_params is not None
                
                # Check parameters are within ranges
                for param, (low, high) in param_ranges.items():
                    assert low <= best_params[param] <= high
                    
        finally:
            os.chdir(original_cwd)
            
    def test_random_search_log_uniform_sampling(self, tmp_path):
        """Test that random search uses log-uniform sampling for appropriate parameters."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            param_ranges = {
                'lr_actor': (1e-5, 1e-2),
                'lr_critic': (1e-4, 1e-2),
                'noise_std': (0.01, 1.0),
                'reward_scale': (0.1, 20.0),
                'td_clip': (1.0, 10.0),
                'lambda_actor': (0.8, 0.99),
                'lambda_critic': (0.9, 0.99),
                'gamma': (0.95, 0.999)
            }
            
            sampled_params = []
            
            def capture_params(params, run_id, num_episodes, show_output):
                sampled_params.append(params.copy())
                return 10.0, {}, None
            
            with patch('src.hyperparam_search.run_experiment', side_effect=capture_params):
                random_search(param_ranges, n_trials=10, num_episodes=5, show_output=False)
                
                # Check that log-uniform parameters have good distribution
                lr_actors = [p['lr_actor'] for p in sampled_params]
                
                # Should have some values in each order of magnitude
                assert any(v < 1e-4 for v in lr_actors)
                assert any(v > 1e-3 for v in lr_actors)
                
        finally:
            os.chdir(original_cwd)


class TestRefinedSearchConfig:
    """Test refined search configuration."""
    
    def test_get_search_config_refined_grid(self):
        """Test getting refined grid search config."""
        config = get_search_config('refined_grid')
        
        assert isinstance(config, dict)
        assert 'lr_actor' in config
        assert isinstance(config['lr_actor'], list)
        assert all(5e-5 <= v <= 2e-4 for v in config['lr_actor'])
        
    def test_get_search_config_refined_random(self):
        """Test getting refined random search config."""
        config = get_search_config('refined_random')
        
        assert isinstance(config, dict)
        assert 'lr_actor' in config
        assert isinstance(config['lr_actor'], tuple)
        assert len(config['lr_actor']) == 2
        assert config['lr_actor'][0] < config['lr_actor'][1]
        
    def test_get_search_config_top_variations(self):
        """Test getting top performer variations config."""
        config = get_search_config('top_variations')
        
        assert isinstance(config, dict)
        assert all(isinstance(v, list) for v in config.values())
        
    def test_get_search_config_invalid(self):
        """Test invalid search type raises error."""
        with pytest.raises(ValueError, match="Unknown search type"):
            get_search_config('invalid_type')


class TestSearchIntegration:
    """Integration tests for search functionality."""
    
    @pytest.mark.slow
    def test_real_small_search(self, tmp_path):
        """Test actual small search with real subprocess calls."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Very small search for real testing
            param_ranges = {
                'lr_actor': (8e-5, 9e-5),
                'lr_critic': (2e-4, 3e-4),
                'lambda_actor': (0.88, 0.89),
                'lambda_critic': (0.92, 0.93),
                'noise_std': (0.05, 0.06),
                'gamma': (0.96, 0.97),
                'reward_scale': (10.0, 11.0),
                'td_clip': (5.0, 6.0)
            }
            
            results, best_params, best_score, search_dir = random_search(
                param_ranges, n_trials=2, num_episodes=5, show_output=False
            )
            
            # Should complete even if some experiments fail
            assert isinstance(results, list)
            assert os.path.exists(search_dir)
            
        finally:
            os.chdir(original_cwd)
            
    def test_search_with_refined_config(self, tmp_path):
        """Test search using refined configuration."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Get refined config
            param_ranges = get_search_config('refined_random')
            
            with patch('src.hyperparam_search.run_experiment') as mock_run:
                mock_run.return_value = (100.0, {}, None)
                
                results, best_params, best_score, search_dir = random_search(
                    param_ranges, n_trials=1, num_episodes=5, show_output=False
                )
                
                # Check parameters are within refined ranges
                assert param_ranges['lr_actor'][0] <= best_params['lr_actor'] <= param_ranges['lr_actor'][1]
                assert param_ranges['noise_std'][0] <= best_params['noise_std'] <= param_ranges['noise_std'][1]
                
        finally:
            os.chdir(original_cwd)