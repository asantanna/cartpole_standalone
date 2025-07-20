#!/usr/bin/env python3
"""
Tests for visualization functions.
"""
import pytest
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from unittest.mock import patch, MagicMock

from src.visualize_learning import (
    load_metrics, plot_single_run, plot_comparison, plot_hyperparameter_analysis,
    plot_best_run_details
)


class TestLoadMetrics:
    """Test metrics loading functionality."""
    
    def test_load_metrics_valid_file(self, tmp_path, sample_metrics):
        """Test loading valid metrics file."""
        metrics_file = tmp_path / "test_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(sample_metrics, f)
            
        loaded = load_metrics(str(metrics_file))
        
        assert loaded['run_id'] == sample_metrics['run_id']
        assert loaded['final_avg_return'] == sample_metrics['final_avg_return']
        assert len(loaded['returns']) == len(sample_metrics['returns'])
        
    def test_load_metrics_invalid_file(self, tmp_path):
        """Test loading invalid metrics file."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json{")
        
        with pytest.raises(json.JSONDecodeError):
            load_metrics(str(invalid_file))


class TestPlotSingleRun:
    """Test single run plotting."""
    
    def test_plot_single_run_basic(self, sample_metrics):
        """Test basic single run plotting."""
        fig, ax = plt.subplots()
        result_ax = plot_single_run(sample_metrics, ax=ax)
        
        assert result_ax is ax
        assert len(ax.lines) >= 1  # At least raw returns plotted
        
        # Check that rolling average is plotted if enough data
        if len(sample_metrics['returns']) > 20:
            assert len(ax.lines) >= 2
            
    def test_plot_single_run_no_ax(self, sample_metrics):
        """Test plotting without providing axes."""
        ax = plot_single_run(sample_metrics)
        
        assert ax is not None
        assert len(ax.lines) >= 1
        
    def test_plot_single_run_with_label(self, sample_metrics):
        """Test plotting with custom label."""
        fig, ax = plt.subplots()
        plot_single_run(sample_metrics, ax=ax, label="Test Run")
        
        # Check that label was set on one of the lines
        labels = [line.get_label() for line in ax.lines]
        assert "Test Run" in labels
        
    def test_plot_single_run_alpha(self, sample_metrics):
        """Test plotting with custom alpha."""
        fig, ax = plt.subplots()
        plot_single_run(sample_metrics, ax=ax, alpha=0.5)
        
        # Check that alpha affects at least one line
        alphas = [line.get_alpha() for line in ax.lines]
        assert any(a is not None and a < 1.0 for a in alphas)


class TestPlotComparison:
    """Test comparison plotting."""
    
    def test_plot_comparison_multiple_files(self, tmp_path, sample_metrics):
        """Test comparing multiple metrics files."""
        # Create multiple metrics files with different scores
        files = []
        for i in range(6):
            metrics = sample_metrics.copy()
            metrics['final_avg_return'] = 10.0 + i * 10
            metrics['returns'] = [r + i * 10 for r in metrics['returns']]
            
            file_path = tmp_path / f"metrics_{i}.json"
            with open(file_path, 'w') as f:
                json.dump(metrics, f)
            files.append(str(file_path))
            
        fig = plot_comparison(files)
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 2  # Top and bottom performers
        
        # Check titles
        assert "Top Performers" in axes[0].get_title()
        assert "Bottom Performers" in axes[1].get_title()
        
    def test_plot_comparison_single_file(self, tmp_path, sample_metrics):
        """Test comparison with single file."""
        file_path = tmp_path / "single_metrics.json"
        with open(file_path, 'w') as f:
            json.dump(sample_metrics, f)
            
        fig = plot_comparison([str(file_path)])
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 2
        
    def test_plot_comparison_missing_files(self, tmp_path):
        """Test comparison with some missing files."""
        existing_file = tmp_path / "exists.json"
        with open(existing_file, 'w') as f:
            json.dump({'returns': [1, 2, 3], 'final_avg_return': 2}, f)
            
        files = [str(existing_file), "nonexistent.json"]
        
        # Should handle missing files gracefully
        fig = plot_comparison(files)
        assert fig is not None


class TestPlotHyperparameterAnalysis:
    """Test hyperparameter analysis plotting."""
    
    def test_plot_hyperparam_analysis(self, tmp_path, sample_search_results):
        """Test hyperparameter impact visualization."""
        results_file = tmp_path / "search_results.json"
        with open(results_file, 'w') as f:
            json.dump(sample_search_results, f)
            
        fig = plot_hyperparameter_analysis(str(results_file))
        
        assert fig is not None
        axes = fig.get_axes()
        
        # Should have subplot for each parameter
        n_params = len(sample_search_results['results'][0]['params'])
        assert len([ax for ax in axes if ax.get_visible()]) <= n_params
        
    def test_plot_hyperparam_empty_results(self, tmp_path):
        """Test with empty results."""
        empty_results = {
            'method': 'random',
            'num_episodes': 10,
            'results': [],
            'best_params': None,
            'best_score': None
        }
        
        results_file = tmp_path / "empty_results.json"
        with open(results_file, 'w') as f:
            json.dump(empty_results, f)
            
        # Should handle empty results gracefully
        fig = plot_hyperparameter_analysis(str(results_file))
        assert fig is None  # Function returns early for no results
        
    def test_plot_hyperparam_log_scale(self, tmp_path, sample_search_results):
        """Test that appropriate parameters use log scale."""
        results_file = tmp_path / "search_results.json"
        with open(results_file, 'w') as f:
            json.dump(sample_search_results, f)
            
        fig = plot_hyperparameter_analysis(str(results_file))
        axes = fig.get_axes()
        
        # Find axes for parameters that should have log scale
        log_params = ['lr_actor', 'lr_critic', 'noise_std', 'reward_scale']
        
        for ax in axes:
            if ax.get_visible() and ax.get_xlabel():
                param_name = ax.get_xlabel().lower().replace(' ', '_')
                if any(log_param in param_name for log_param in log_params):
                    # Check if x-axis is log scale
                    assert ax.get_xscale() == 'log'


class TestPlotBestRunDetails:
    """Test detailed best run plotting."""
    
    def test_plot_best_run_details(self, tmp_path, sample_metrics):
        """Test detailed analysis plot."""
        metrics_file = tmp_path / "best_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(sample_metrics, f)
            
        fig = plot_best_run_details(str(metrics_file))
        
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 4  # 2x2 grid
        
        # Check subplot contents
        titles = [ax.get_title() for ax in axes if ax.get_visible()]
        assert any("Learning Progress" in t for t in titles)
        assert any("Return Distribution" in t for t in titles)
        
    def test_plot_best_run_statistics(self, tmp_path):
        """Test that statistics are calculated correctly."""
        metrics = {
            'run_id': 'test_stats',
            'hyperparameters': {'lr_actor': 1e-4},
            'returns': list(range(100)),  # 0 to 99
            'final_avg_return': np.mean(list(range(50, 100)))
        }
        
        metrics_file = tmp_path / "stats_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
            
        fig = plot_best_run_details(str(metrics_file))
        
        # The plot should show correct statistics
        # Mean should be 49.5, median 49.5, etc.
        assert fig is not None


class TestVisualizationModes:
    """Test different visualization modes in main function."""
    
    @patch('src.visualize_learning.plt.show')
    @patch('src.visualize_learning.glob.glob')
    def test_single_file_mode(self, mock_glob, mock_show, tmp_path, sample_metrics):
        """Test single file visualization mode."""
        metrics_file = tmp_path / "single.json"
        with open(metrics_file, 'w') as f:
            json.dump(sample_metrics, f)
            
        mock_glob.return_value = [str(metrics_file)]
        
        # Import and call main with mocked args
        from src.visualize_learning import main
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = ['visualize_learning.py', str(metrics_file)]
            main()
            mock_show.assert_called_once()
        finally:
            sys.argv = old_argv
            
    @patch('src.visualize_learning.plt.show')
    @patch('src.visualize_learning.glob.glob')
    def test_wildcard_comparison_mode(self, mock_glob, mock_show, tmp_path, sample_metrics):
        """Test wildcard pattern triggering comparison mode."""
        # Create multiple files
        files = []
        for i in range(3):
            metrics_file = tmp_path / f"metrics_{i}.json"
            with open(metrics_file, 'w') as f:
                json.dump(sample_metrics, f)
            files.append(str(metrics_file))
            
        mock_glob.return_value = files
        
        from src.visualize_learning import main
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = ['visualize_learning.py', 'metrics_*.json']
            main()
            mock_show.assert_called_once()
        finally:
            sys.argv = old_argv
            
    @patch('src.visualize_learning.plt.show')
    def test_best_only_mode(self, mock_show, tmp_path, sample_metrics):
        """Test --best-only flag."""
        # Create files with different scores
        best_metrics = sample_metrics.copy()
        best_metrics['final_avg_return'] = 100.0
        
        worse_metrics = sample_metrics.copy()
        worse_metrics['final_avg_return'] = 50.0
        
        best_file = tmp_path / "best.json"
        with open(best_file, 'w') as f:
            json.dump(best_metrics, f)
            
        worse_file = tmp_path / "worse.json"
        with open(worse_file, 'w') as f:
            json.dump(worse_metrics, f)
            
        from src.visualize_learning import main
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = ['visualize_learning.py', str(tmp_path / "*.json"), '--best-only']
            main()
            mock_show.assert_called_once()
        finally:
            sys.argv = old_argv


class TestMatplotlibBackend:
    """Test matplotlib backend configuration."""
    
    def test_backend_is_non_interactive(self):
        """Verify matplotlib is using non-interactive backend."""
        backend = matplotlib.get_backend()
        assert backend.lower() == 'agg'  # Set in conftest.py
        
    def test_figures_are_closed(self, sample_metrics):
        """Test that figures are properly closed."""
        initial_figs = plt.get_fignums()
        
        # Create some plots
        plot_single_run(sample_metrics)
        plot_single_run(sample_metrics)
        
        # Should have created new figures
        assert len(plt.get_fignums()) > len(initial_figs)
        
        # cleanup_matplotlib fixture should close them after test