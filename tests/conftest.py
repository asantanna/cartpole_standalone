#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for CartPole tests.
"""
import pytest
import sys
import os
import json
import numpy as np
from datetime import datetime
import shutil

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Isaac Gym before PyTorch (required by Isaac Gym)
import isaacgym
import isaacgymenvs
import torch


def pytest_configure(config):
    """Check for Isaac Gym before running any tests."""
    # Isaac Gym imports are now at the top of the file
    # If we got here, they imported successfully
    
    # Configure matplotlib to use non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_run"
    test_dir.mkdir()
    original_cwd = os.getcwd()
    os.chdir(test_dir)
    yield test_dir
    os.chdir(original_cwd)


@pytest.fixture
def sample_metrics():
    """Generate sample metrics data."""
    returns = [10.5 + i + np.random.normal(0, 5) for i in range(100)]
    return {
        'run_id': 'test_run_123',
        'hyperparameters': {
            'lr_actor': 8e-5,
            'lr_critic': 2.5e-4,
            'lambda_actor': 0.88,
            'lambda_critic': 0.92,
            'noise_std': 0.05,
            'gamma': 0.96,
            'reward_scale': 10.0,
            'td_clip': 5.0
        },
        'returns': returns,
        'final_avg_return': np.mean(returns[-50:])
    }


@pytest.fixture
def sample_checkpoint(tmp_path):
    """Create a sample checkpoint file."""
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    
    obs_dim = 4
    act_dim = 1
    device = 'cpu'
    
    checkpoint = {
        'W_a': torch.randn(act_dim, obs_dim),
        'W_c': torch.randn(1, obs_dim),
        'log_std': torch.full((act_dim,), np.log(0.05)),
        'hyperparams': {
            'lr_a': 8e-5,
            'lr_c': 2.5e-4,
            'lambda_a': 0.88,
            'lambda_c': 0.92,
            'obs_dim': obs_dim,
            'act_dim': act_dim
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_search_results():
    """Generate sample hyperparameter search results."""
    results = []
    for i in range(5):
        params = {
            'lr_actor': 10 ** np.random.uniform(-5, -3),
            'lr_critic': 10 ** np.random.uniform(-4, -2),
            'lambda_actor': np.random.uniform(0.8, 0.99),
            'lambda_critic': np.random.uniform(0.9, 0.99),
            'noise_std': 10 ** np.random.uniform(-2, -0.5),
            'gamma': np.random.uniform(0.95, 0.99),
            'reward_scale': np.random.uniform(1, 20),
            'td_clip': np.random.uniform(1, 10)
        }
        score = np.random.uniform(0, 100) + (100 if i == 2 else 0)  # Make one clearly best
        results.append({
            'params': params,
            'score': score,
            'run_id': f'test_{i}'
        })
    
    # Find best
    best_idx = max(range(len(results)), key=lambda i: results[i]['score'])
    
    return {
        'method': 'random',
        'num_episodes': 10,
        'results': results,
        'best_params': results[best_idx]['params'],
        'best_score': results[best_idx]['score']
    }


@pytest.fixture
def run_directory_setup(tmp_path):
    """Set up a runs directory structure for testing."""
    runs_dir = tmp_path / "runs"
    single_dir = runs_dir / "single"
    search_dir = runs_dir / "search"
    
    single_dir.mkdir(parents=True)
    search_dir.mkdir(parents=True)
    
    # Create some sample run directories
    run1 = single_dir / "train_20250720_120000"
    run1.mkdir()
    
    run2 = single_dir / "train_20250720_130000"
    run2.mkdir()
    
    search1 = search_dir / "random_20250720_140000"
    search1.mkdir()
    
    return runs_dir


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture
def mock_args():
    """Create a mock args object with default values."""
    class Args:
        def __init__(self):
            self.visual = False
            self.num_episodes = 10
            self.lr_actor = 8e-5
            self.lr_critic = 2.5e-4
            self.lambda_actor = 0.88
            self.lambda_critic = 0.92
            self.noise_std = 0.05
            self.gamma = 0.96
            self.reward_scale = 10.0
            self.td_clip = 5.0
            self.save_metrics = False
            self.run_id = 'test_run'
            self.best_config = False
            self.save_checkpoint = None
            self.load_checkpoint = None
            self.training_mode = None
    
    return Args()


# Skip markers for tests that require specific conditions
def pytest_collection_modifyitems(config, items):
    """Add skip markers based on available resources."""
    gpu_available = torch.cuda.is_available()
    
    for item in items:
        # Skip GPU tests if no GPU available
        if "gpu" in item.keywords and not gpu_available:
            item.add_marker(
                pytest.mark.skip(reason="Test requires GPU but none available")
            )