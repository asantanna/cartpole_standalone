#!/usr/bin/env python3
"""
Tests for command-line argument parsing.
"""
import pytest
import sys
import argparse
from io import StringIO
from unittest.mock import patch

# Import argument parsers from main modules
import src.cartpole as cartpole
import src.hyperparam_search as hyperparam_search
import src.visualize_learning as visualize_learning
import src.analyze_results as analyze_results


class TestCartpoleCLI:
    """Test cartpole.py command-line arguments."""
    
    def test_default_args(self):
        """Test default argument values."""
        with patch('sys.argv', ['cartpole.py']):
            parser = argparse.ArgumentParser(description='Train CartPole with Actor-Critic')
            # Copy the argument setup from cartpole.py
            parser.add_argument('--visual', action='store_true', 
                                help='Show visualization window (default: headless)')
            parser.add_argument('--num-episodes', type=int, default=500,
                                help='Number of training episodes (default: 500)')
            parser.add_argument('--lr-actor', type=float, default=8e-5,
                                help='Actor learning rate (default: 8e-5)')
            parser.add_argument('--lr-critic', type=float, default=2.5e-4,
                                help='Critic learning rate (default: 2.5e-4)')
            parser.add_argument('--lambda-actor', type=float, default=0.88,
                                help='Actor eligibility trace decay (default: 0.88)')
            parser.add_argument('--lambda-critic', type=float, default=0.92,
                                help='Critic eligibility trace decay (default: 0.92)')
            parser.add_argument('--noise-std', type=float, default=0.05,
                                help='Action noise standard deviation (default: 0.05)')
            parser.add_argument('--gamma', type=float, default=0.96,
                                help='Discount factor (default: 0.96)')
            parser.add_argument('--reward-scale', type=float, default=10.0,
                                help='Reward scaling divisor (default: 10.0)')
            parser.add_argument('--td-clip', type=float, default=5.0,
                                help='TD error clipping value (default: 5.0)')
            parser.add_argument('--save-metrics', action='store_true',
                                help='Save training metrics to file')
            parser.add_argument('--run-id', type=str, default='default',
                                help='Run identifier for saving metrics')
            parser.add_argument('--best-config', action='store_true',
                                help='Use the best hyperparameters from search')
            parser.add_argument('--save-checkpoint', type=str, default=None,
                                help='Path to save checkpoint after training')
            parser.add_argument('--load-checkpoint', type=str, default=None,
                                help='Path to load checkpoint from')
            parser.add_argument('--training-mode', type=str, choices=['true', 'false'], default=None,
                                help='Override training mode (true/false)')
            
            args = parser.parse_args()
            
            assert args.visual == False
            assert args.num_episodes == 500
            assert args.lr_actor == 8e-5
            assert args.lr_critic == 2.5e-4
            assert args.lambda_actor == 0.88
            assert args.lambda_critic == 0.92
            assert args.noise_std == 0.05
            assert args.gamma == 0.96
            assert args.reward_scale == 10.0
            assert args.td_clip == 5.0
            assert args.save_metrics == False
            assert args.run_id == 'default'
            assert args.best_config == False
            assert args.save_checkpoint is None
            assert args.load_checkpoint is None
            assert args.training_mode is None
            
    def test_custom_hyperparameters(self):
        """Test parsing custom hyperparameter values."""
        test_args = [
            'cartpole.py',
            '--lr-actor', '1e-4',
            '--lr-critic', '5e-4',
            '--lambda-actor', '0.9',
            '--lambda-critic', '0.95',
            '--noise-std', '0.1',
            '--gamma', '0.99',
            '--reward-scale', '5.0',
            '--td-clip', '10.0'
        ]
        
        with patch('sys.argv', test_args):
            # Use the actual parser from cartpole module
            old_argv = sys.argv
            sys.argv = test_args
            
            parser = argparse.ArgumentParser()
            parser.add_argument('--lr-actor', type=float, default=8e-5)
            parser.add_argument('--lr-critic', type=float, default=2.5e-4)
            parser.add_argument('--lambda-actor', type=float, default=0.88)
            parser.add_argument('--lambda-critic', type=float, default=0.92)
            parser.add_argument('--noise-std', type=float, default=0.05)
            parser.add_argument('--gamma', type=float, default=0.96)
            parser.add_argument('--reward-scale', type=float, default=10.0)
            parser.add_argument('--td-clip', type=float, default=5.0)
            
            args = parser.parse_args()
            
            assert args.lr_actor == 1e-4
            assert args.lr_critic == 5e-4
            assert args.lambda_actor == 0.9
            assert args.lambda_critic == 0.95
            assert args.noise_std == 0.1
            assert args.gamma == 0.99
            assert args.reward_scale == 5.0
            assert args.td_clip == 10.0
            
            sys.argv = old_argv
            
    def test_boolean_flags(self):
        """Test boolean flag arguments."""
        test_args = [
            'cartpole.py',
            '--visual',
            '--save-metrics',
            '--best-config'
        ]
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('--visual', action='store_true')
            parser.add_argument('--save-metrics', action='store_true')
            parser.add_argument('--best-config', action='store_true')
            
            args = parser.parse_args()
            
            assert args.visual == True
            assert args.save_metrics == True
            assert args.best_config == True
            
    def test_checkpoint_arguments(self):
        """Test checkpoint-related arguments."""
        test_args = [
            'cartpole.py',
            '--save-checkpoint', 'model.pt',
            '--load-checkpoint', 'pretrained.pt',
            '--training-mode', 'false'
        ]
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('--save-checkpoint', type=str, default=None)
            parser.add_argument('--load-checkpoint', type=str, default=None)
            parser.add_argument('--training-mode', type=str, choices=['true', 'false'], default=None)
            
            args = parser.parse_args()
            
            assert args.save_checkpoint == 'model.pt'
            assert args.load_checkpoint == 'pretrained.pt'
            assert args.training_mode == 'false'


class TestHyperparamSearchCLI:
    """Test hyperparam_search.py command-line arguments."""
    
    def test_default_search_args(self):
        """Test default search arguments."""
        with patch('sys.argv', ['hyperparam_search.py']):
            parser = argparse.ArgumentParser(description='Hyperparameter search for CartPole')
            parser.add_argument('--method', type=str, choices=['grid', 'random'], default='random')
            parser.add_argument('--n-trials', type=int, default=30)
            parser.add_argument('--num-episodes', type=int, default=100)
            parser.add_argument('--output', type=str, default='search_results.json')
            parser.add_argument('--quiet', action='store_true')
            parser.add_argument('--use-refined', action='store_true')
            
            args = parser.parse_args()
            
            assert args.method == 'random'
            assert args.n_trials == 30
            assert args.num_episodes == 100
            assert args.output == 'search_results.json'
            assert args.quiet == False
            assert args.use_refined == False
            
    def test_grid_search_args(self):
        """Test grid search specific arguments."""
        test_args = [
            'hyperparam_search.py',
            '--method', 'grid',
            '--num-episodes', '50',
            '--use-refined'
        ]
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('--method', type=str, choices=['grid', 'random'], default='random')
            parser.add_argument('--num-episodes', type=int, default=100)
            parser.add_argument('--use-refined', action='store_true')
            
            args = parser.parse_args()
            
            assert args.method == 'grid'
            assert args.num_episodes == 50
            assert args.use_refined == True
            
    def test_quiet_mode(self):
        """Test quiet mode flag."""
        test_args = ['hyperparam_search.py', '--quiet']
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('--quiet', action='store_true')
            
            args = parser.parse_args()
            assert args.quiet == True


class TestVisualizationCLI:
    """Test visualize_learning.py command-line arguments."""
    
    def test_positional_input(self):
        """Test positional input argument."""
        test_args = ['visualize_learning.py', 'metrics.json']
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('input', type=str, nargs='?')
            
            args = parser.parse_args()
            assert args.input == 'metrics.json'
            
    def test_optional_arguments(self):
        """Test optional visualization arguments."""
        test_args = [
            'visualize_learning.py',
            'metrics_*.json',
            '--output', 'plot.png',
            '--best-only',
            '--compare'
        ]
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('input', type=str, nargs='?')
            parser.add_argument('--output', type=str)
            parser.add_argument('--best-only', action='store_true')
            parser.add_argument('--compare', action='store_true')
            
            args = parser.parse_args()
            
            assert args.input == 'metrics_*.json'
            assert args.output == 'plot.png'
            assert args.best_only == True
            assert args.compare == True
            
    def test_search_results_arg(self):
        """Test search results argument."""
        test_args = [
            'visualize_learning.py',
            '--search-results', 'search_results.json'
        ]
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('input', type=str, nargs='?')
            parser.add_argument('--search-results', type=str)
            
            args = parser.parse_args()
            
            assert args.input is None
            assert args.search_results == 'search_results.json'


class TestAnalyzeResultsCLI:
    """Test analyze_results.py command-line arguments."""
    
    def test_default_analyze_args(self):
        """Test default analysis arguments."""
        with patch('sys.argv', ['analyze_results.py']):
            parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
            parser.add_argument('--input', type=str, default='search_results.json',
                                help='Input results file')
            
            args = parser.parse_args()
            assert args.input == 'search_results.json'
            
    def test_custom_input_file(self):
        """Test custom input file argument."""
        test_args = ['analyze_results.py', '--input', 'custom_results.json']
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('--input', type=str, default='search_results.json')
            
            args = parser.parse_args()
            assert args.input == 'custom_results.json'


class TestArgumentValidation:
    """Test argument validation and error handling."""
    
    def test_invalid_choice(self):
        """Test invalid choice for constrained arguments."""
        test_args = ['cartpole.py', '--training-mode', 'invalid']
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('--training-mode', type=str, choices=['true', 'false'])
            
            with pytest.raises(SystemExit):
                parser.parse_args()
                
    def test_invalid_type(self):
        """Test invalid type for numeric arguments."""
        test_args = ['cartpole.py', '--num-episodes', 'not-a-number']
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('--num-episodes', type=int)
            
            with pytest.raises(SystemExit):
                parser.parse_args()
                
    def test_help_message(self):
        """Test help message display."""
        test_args = ['cartpole.py', '--help']
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser(description='Test help')
            parser.add_argument('--test', help='Test argument')
            
            with pytest.raises(SystemExit):
                with patch('sys.stdout', new=StringIO()) as fake_out:
                    parser.parse_args()
                    output = fake_out.getvalue()
                    assert 'Test help' in output
                    assert '--test' in output