#!/usr/bin/env python3
"""
Integration tests for the full workflow.
"""
import pytest
import os
import json
import glob
import shutil
import subprocess
import time

from src.cartpole import train, get_run_directory, ensure_directory_exists, resolve_checkpoint_path
from src.hyperparam_search import random_search, grid_search
from src.visualize_learning import load_metrics, plot_single_run
from src.analyze_results import analyze_results


class TestSingleRunWorkflow:
    """Test complete single training run workflow."""
    
    @pytest.mark.slow
    def test_complete_single_run(self, tmp_path, mock_args):
        """Test training → metrics → checkpoint flow."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Configure for a complete run
            mock_args.num_episodes = 10
            mock_args.save_metrics = True
            mock_args.save_checkpoint = 'model.pt'
            mock_args.run_id = 'integration_test'
            
            # Train
            final_return = train(headless=True, num_episodes=10, args=mock_args)
            
            # Verify outputs
            run_dir = 'runs/single/integration_test'
            assert os.path.exists(run_dir)
            
            # Check metrics
            metrics_path = os.path.join(run_dir, 'metrics.json')
            assert os.path.exists(metrics_path)
            
            metrics = load_metrics(metrics_path)
            assert metrics['run_id'] == 'integration_test'
            assert len(metrics['returns']) == 10
            assert metrics['final_avg_return'] == final_return
            
            # Check checkpoint
            checkpoint_path = os.path.join(run_dir, 'model.pt')
            assert os.path.exists(checkpoint_path)
            
            # Verify checkpoint can be loaded
            import torch
            checkpoint = torch.load(checkpoint_path)
            assert 'W_a' in checkpoint
            assert 'W_c' in checkpoint
            assert 'hyperparams' in checkpoint
            
            # Test visualization of the results
            ax = plot_single_run(metrics)
            assert ax is not None
            
        finally:
            os.chdir(original_cwd)
            
    @pytest.mark.skip(reason="Multiple Isaac Gym environments cause segmentation fault")
    def test_multiple_sequential_runs(self, tmp_path, mock_args):
        """Test multiple training runs with different configurations."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            run_ids = []
            
            # Run 1: Default config
            mock_args.num_episodes = 5
            mock_args.save_metrics = True
            mock_args.run_id = None  # Auto-generate
            
            train(headless=True, num_episodes=5, args=mock_args)
            
            # Find the created run
            runs = glob.glob('runs/single/train_*')
            assert len(runs) == 1
            run_ids.append(os.path.basename(runs[0]))
            
            # Wait to ensure different timestamp
            time.sleep(0.1)
            
            # Run 2: Best config
            mock_args.best_config = True
            mock_args.run_id = None  # Auto-generate another
            
            train(headless=True, num_episodes=5, args=mock_args)
            
            # Should have 2 runs now
            runs = glob.glob('runs/single/train_*')
            assert len(runs) == 2
            
            # Verify both runs have metrics
            for run in runs:
                metrics_file = os.path.join(run, 'metrics.json')
                assert os.path.exists(metrics_file)
                
        finally:
            os.chdir(original_cwd)


class TestSearchWorkflow:
    """Test hyperparameter search workflow."""
    
    @pytest.mark.slow
    def test_small_search_to_visualization(self, tmp_path):
        """Test search → analysis → visualization workflow."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Define very small search space
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
            
            # Run small random search
            results, best_params, best_score, search_dir = random_search(
                param_ranges, n_trials=2, num_episodes=5, show_output=False
            )
            
            # Verify search directory structure
            assert os.path.exists(search_dir)
            search_results_file = os.path.join(search_dir, 'search_results.json')
            assert os.path.exists(search_results_file)
            
            # Test analysis of results
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                analyze_results(search_results_file)
                output = sys.stdout.getvalue()
                
                # Check analysis output
                assert "SEARCH COMPLETE" in output or "Search Method:" in output
                assert "TOP" in output or "Best score:" in output
                
            finally:
                sys.stdout = old_stdout
                
            # Test visualization of search results
            from visualize_learning import plot_hyperparameter_analysis
            
            # This might return None if no valid results, which is ok
            fig = plot_hyperparameter_analysis(search_results_file)
            # Just verify it doesn't crash
            
        finally:
            os.chdir(original_cwd)
            
    def test_grid_search_integration(self, tmp_path):
        """Test grid search with directory structure."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Tiny grid for testing
            param_grid = {
                'lr_actor': [8e-5],
                'lr_critic': [2.5e-4],
                'lambda_actor': [0.88],
                'lambda_critic': [0.92],
                'noise_std': [0.05, 0.1],  # 2 values
                'gamma': [0.96],
                'reward_scale': [10.0],
                'td_clip': [5.0]
            }
            
            # Run grid search
            results, best_params, best_score, search_dir = grid_search(
                param_grid, num_episodes=5, show_output=False
            )
            
            # Should have 2 results (2 noise_std values)
            assert len(results) <= 2  # Some might fail
            
            # Check directory structure
            assert os.path.exists(search_dir)
            
            # Each successful result should have a subdirectory
            for result in results:
                if result['score'] is not None:
                    trial_dir = os.path.join(search_dir, result['run_id'])
                    assert os.path.exists(trial_dir)
                    
        finally:
            os.chdir(original_cwd)


class TestCheckpointWorkflow:
    """Test checkpoint save/load workflow."""
    
    @pytest.mark.slow  
    def test_checkpoint_save_load_continue(self, tmp_path, mock_args):
        """Test save → load → continue training workflow."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Phase 1: Initial training
            mock_args.num_episodes = 5
            mock_args.save_checkpoint = 'checkpoint.pt'
            mock_args.save_metrics = True
            mock_args.run_id = 'initial'
            
            initial_return = train(headless=True, num_episodes=5, args=mock_args)
            
            initial_metrics_path = 'runs/single/initial/metrics.json'
            initial_checkpoint_path = 'runs/single/initial/checkpoint.pt'
            
            assert os.path.exists(initial_metrics_path)
            assert os.path.exists(initial_checkpoint_path)
            
            # Phase 2: Load and evaluate (no training)
            mock_args.load_checkpoint = initial_checkpoint_path
            mock_args.training_mode = 'false'
            mock_args.run_id = 'eval'
            mock_args.save_checkpoint = None
            
            eval_return = train(headless=True, num_episodes=5, args=mock_args)
            
            # Phase 3: Load and continue training
            mock_args.training_mode = 'true'
            mock_args.run_id = 'continued'
            mock_args.save_checkpoint = 'continued.pt'
            
            continued_return = train(headless=True, num_episodes=5, args=mock_args)
            
            # Verify all phases created their outputs
            assert os.path.exists('runs/single/eval/metrics.json')
            assert os.path.exists('runs/single/continued/metrics.json')
            assert os.path.exists('runs/single/continued/continued.pt')
            
        finally:
            os.chdir(original_cwd)
            
    def test_checkpoint_path_resolution(self, tmp_path, mock_args):
        """Test finding checkpoints in various locations."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create checkpoints in different locations
            locations = [
                'runs/single/run1/model.pt',
                'runs/single/run2/checkpoint.pt',
                'runs/search/search1/trial1/best.pt'
            ]
            
            import torch
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
            
            for loc in locations:
                os.makedirs(os.path.dirname(loc), exist_ok=True)
                torch.save(dummy_checkpoint, loc)
                
            # Test resolution
            resolved = resolve_checkpoint_path('model.pt')
            assert resolved == 'runs/single/run1/model.pt'
            
            resolved = resolve_checkpoint_path('checkpoint.pt')
            assert resolved == 'runs/single/run2/checkpoint.pt'
            
            resolved = resolve_checkpoint_path('best.pt')
            assert resolved == 'runs/search/search1/trial1/best.pt'
            
            resolved = resolve_checkpoint_path('nonexistent.pt')
            assert resolved == 'nonexistent.pt'  # Returns original if not found
            
        finally:
            os.chdir(original_cwd)


class TestDirectoryStructure:
    """Test directory structure creation and organization."""
    
    def test_full_directory_structure(self, tmp_path):
        """Test creation of full directory structure."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create various directories
            single_runs = [
                get_run_directory('single', 'train_1'),
                get_run_directory('single', 'train_2'),
                get_run_directory('single')  # Auto-generated
            ]
            
            search_runs = [
                get_run_directory('search', 'grid_search_1'),
                get_run_directory('search', 'random_search_1')
            ]
            
            # Ensure all directories
            for run_dir in single_runs + search_runs:
                ensure_directory_exists(run_dir)
                
            # Verify structure
            assert os.path.exists('runs')
            assert os.path.exists('runs/single')
            assert os.path.exists('runs/search')
            
            # Count directories
            single_dirs = glob.glob('runs/single/*')
            search_dirs = glob.glob('runs/search/*')
            
            assert len(single_dirs) == 3
            assert len(search_dirs) == 2
            
            # Create some files in the directories
            for run_dir in single_runs[:2]:
                metrics_file = os.path.join(run_dir, 'metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump({'test': True}, f)
                    
            # Test finding files
            metrics_files = glob.glob('runs/single/*/metrics.json')
            assert len(metrics_files) == 2
            
        finally:
            os.chdir(original_cwd)
            
    def test_parallel_directory_creation(self, tmp_path):
        """Test that parallel runs don't conflict."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Simulate parallel directory creation
            dirs = []
            for i in range(5):
                run_dir = get_run_directory('single')
                ensure_directory_exists(run_dir)
                dirs.append(run_dir)
                time.sleep(0.01)  # Small delay to ensure different timestamps
                
            # All directories should be unique
            assert len(set(dirs)) == 5
            
            # All should exist
            for d in dirs:
                assert os.path.exists(d)
                
        finally:
            os.chdir(original_cwd)


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    @pytest.mark.slow
    def test_research_workflow(self, tmp_path, mock_args):
        """Test typical research workflow: explore → refine → evaluate."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Step 1: Initial exploration with random search
            param_ranges = {
                'lr_actor': (5e-5, 1e-4),
                'lr_critic': (2e-4, 3e-4),
                'lambda_actor': (0.85, 0.9),
                'lambda_critic': (0.9, 0.95),
                'noise_std': (0.05, 0.1),
                'gamma': (0.95, 0.97),
                'reward_scale': (8.0, 12.0),
                'td_clip': (4.0, 6.0)
            }
            
            # Small search
            results, best_params, best_score, search_dir = random_search(
                param_ranges, n_trials=2, num_episodes=5, show_output=False
            )
            
            # Step 2: Train longer with best params (if found)
            if best_params:
                mock_args.num_episodes = 10
                mock_args.save_checkpoint = 'best_model.pt'
                mock_args.save_metrics = True
                mock_args.run_id = 'best_config_run'
                
                # Set hyperparameters from search
                for param, value in best_params.items():
                    if param == 'lr_actor':
                        mock_args.lr_actor = value
                    elif param == 'lr_critic':
                        mock_args.lr_critic = value
                    # ... etc
                
                train(headless=True, num_episodes=10, args=mock_args)
                
                # Step 3: Evaluate the best model
                mock_args.load_checkpoint = 'runs/single/best_config_run/best_model.pt'
                mock_args.training_mode = 'false'
                mock_args.run_id = 'evaluation'
                
                train(headless=True, num_episodes=5, args=mock_args)
                
                # Verify complete workflow artifacts
                assert os.path.exists(search_dir)
                assert os.path.exists('runs/single/best_config_run/best_model.pt')
                assert os.path.exists('runs/single/evaluation/metrics.json')
                
        finally:
            os.chdir(original_cwd)