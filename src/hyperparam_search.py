#!/usr/bin/env python3
import subprocess
import itertools
import json
import numpy as np
import argparse
from datetime import datetime
import os
import shutil
import sys

def run_experiment(params, run_id, num_episodes=100, show_output=True):
    """Run a single experiment with given hyperparameters."""
    cmd = [
        sys.executable, 'src/cartpole.py',
        '--num-episodes', str(num_episodes),
        '--lr-actor', str(params['lr_actor']),
        '--lr-critic', str(params['lr_critic']),
        '--lambda-actor', str(params['lambda_actor']),
        '--lambda-critic', str(params['lambda_critic']),
        '--noise-std', str(params['noise_std']),
        '--gamma', str(params['gamma']),
        '--reward-scale', str(params['reward_scale']),
        '--td-clip', str(params['td_clip']),
        '--save-metrics',
        '--run-id', run_id
    ]
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {run_id}")
    print(f"{'='*80}")
    print(f"Parameters:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")
    print(f"{'='*80}")
    
    try:
        if show_output:
            # Run with real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1, universal_newlines=True)
            
            # Stream output line by line
            for line in process.stdout:
                if any(keyword in line for keyword in ['Episode', 'Return', 'Creating', 'Starting']):
                    print(f"  {line.strip()}")
            
            process.wait(timeout=300)
            returncode = process.returncode
        else:
            # Run silently
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            returncode = result.returncode
        
        if returncode == 0:
            # Load the metrics file from runs directory
            metrics_file = os.path.join('runs', 'single', run_id, 'metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                final_score = metrics['final_avg_return']
                print(f"\n✓ Experiment completed. Final average return: {final_score:.2f}")
                return final_score, metrics, metrics_file
            else:
                print(f"Warning: Metrics file not found at {metrics_file}")
                return None, None, None
        else:
            print(f"✗ Error running experiment")
            return None, None, None
            
    except subprocess.TimeoutExpired:
        print(f"✗ Experiment {run_id} timed out")
        return None, None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None, None

def grid_search(param_grid, num_episodes=100, show_output=True, search_id=None):
    """Perform grid search over hyperparameter combinations."""
    # Create search directory
    if search_id is None:
        search_id = f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    search_dir = os.path.join('runs', 'search', search_id)
    os.makedirs(search_dir, exist_ok=True)
    print(f"Search directory: {search_dir}")
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*values))
    
    print(f"\n🔍 GRID SEARCH")
    print(f"Total combinations to test: {len(combinations)}")
    print(f"Episodes per trial: {num_episodes}")
    
    results = []
    best_score = -float('inf')
    best_params = None
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        run_id = f"grid_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n[{i+1}/{len(combinations)}] Starting experiment...")
        score, metrics, metrics_file = run_experiment(params, run_id, num_episodes, show_output)
        
        if score is not None:
            # Copy metrics to search directory
            trial_dir = os.path.join(search_dir, run_id)
            os.makedirs(trial_dir, exist_ok=True)
            if metrics_file and os.path.exists(metrics_file):
                shutil.copy2(metrics_file, os.path.join(trial_dir, 'metrics.json'))
            
            results.append({
                'params': params,
                'score': score,
                'run_id': run_id
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"\n🏆 NEW BEST SCORE: {best_score:.2f}")
        
        # Summary so far
        if results:
            scores = [r['score'] for r in results]
            print(f"\n📊 Progress: {i+1}/{len(combinations)} experiments completed")
            print(f"   Current best: {best_score:.2f}")
            print(f"   Average score: {np.mean(scores):.2f}")
    
    # Save search results in search directory
    search_results_path = os.path.join(search_dir, 'search_results.json')
    search_data = {
        'method': 'grid',
        'num_episodes': num_episodes,
        'results': results,
        'best_params': best_params,
        'best_score': best_score
    }
    with open(search_results_path, 'w') as f:
        json.dump(search_data, f, indent=2)
    print(f"\nSearch results saved to {search_results_path}")
    
    return results, best_params, best_score, search_dir

def random_search(param_ranges, n_trials=50, num_episodes=100, show_output=True, search_id=None):
    """Perform random search over hyperparameter ranges."""
    # Create search directory
    if search_id is None:
        search_id = f"random_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    search_dir = os.path.join('runs', 'search', search_id)
    os.makedirs(search_dir, exist_ok=True)
    print(f"Search directory: {search_dir}")
    
    print(f"\n🎲 RANDOM SEARCH")
    print(f"Total trials: {n_trials}")
    print(f"Episodes per trial: {num_episodes}")
    
    results = []
    best_score = -float('inf')
    best_params = None
    
    for i in range(n_trials):
        # Sample random parameters
        params = {}
        for param, (low, high) in param_ranges.items():
            if param in ['lr_actor', 'lr_critic', 'noise_std', 'reward_scale', 'td_clip']:
                # Log-uniform sampling for learning rates and scales
                params[param] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            else:
                # Uniform sampling for others
                params[param] = np.random.uniform(low, high)
        
        run_id = f"random_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n[{i+1}/{n_trials}] Starting experiment...")
        score, metrics, metrics_file = run_experiment(params, run_id, num_episodes, show_output)
        
        if score is not None:
            # Copy metrics to search directory
            trial_dir = os.path.join(search_dir, run_id)
            os.makedirs(trial_dir, exist_ok=True)
            if metrics_file and os.path.exists(metrics_file):
                shutil.copy2(metrics_file, os.path.join(trial_dir, 'metrics.json'))
            results.append({
                'params': params,
                'score': score,
                'run_id': run_id
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"\n🏆 NEW BEST SCORE: {best_score:.2f}")
        
        # Summary so far
        if results:
            scores = [r['score'] for r in results]
            print(f"\n📊 Progress: {i+1}/{n_trials} experiments completed")
            print(f"   Current best: {best_score:.2f}")
            print(f"   Average score: {np.mean(scores):.2f}")
            print(f"   Success rate: {len(results)}/{i+1}")
    
    # Save search results in search directory
    search_results_path = os.path.join(search_dir, 'search_results.json')
    search_data = {
        'method': 'random',
        'num_episodes': num_episodes,
        'results': results,
        'best_params': best_params,
        'best_score': best_score
    }
    with open(search_results_path, 'w') as f:
        json.dump(search_data, f, indent=2)
    print(f"\nSearch results saved to {search_results_path}")
    
    return results, best_params, best_score, search_dir

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for CartPole')
    parser.add_argument('--method', type=str, choices=['grid', 'random'], default='random',
                        help='Search method: grid or random')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of trials for random search')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes per trial')
    parser.add_argument('--output', type=str, default='search_results.json',
                        help='Output file for results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress training output from individual runs')
    parser.add_argument('--use-refined', action='store_true',
                        help='Use refined search space based on previous results')
    args = parser.parse_args()
    
    show_output = not args.quiet
    
    if args.method == 'grid':
        if args.use_refined:
            from refined_search_config import get_search_config
            param_grid = get_search_config('refined_grid')
            print("Using refined grid search configuration")
        else:
            # Define original grid search parameters
            param_grid = {
                'lr_actor': [1e-4, 5e-4, 1e-3, 5e-3],
                'lr_critic': [1e-3, 5e-3, 1e-2],
                'lambda_actor': [0.9, 0.95, 0.99],
                'lambda_critic': [0.95, 0.98, 0.99],
                'noise_std': [0.1, 0.3, 0.5],
                'gamma': [0.99],  # Fixed
                'reward_scale': [1.0, 5.0, 10.0],
                'td_clip': [5.0]  # Fixed
            }
        
        results, best_params, best_score, search_dir = grid_search(param_grid, args.num_episodes, show_output)
        
    else:  # random search
        if args.use_refined:
            from refined_search_config import get_search_config
            param_ranges = get_search_config('refined_random')
            print("Using refined random search configuration")
        else:
            # Define original parameter ranges for random search
            param_ranges = {
                'lr_actor': (1e-5, 1e-2),
                'lr_critic': (1e-4, 5e-2),
                'lambda_actor': (0.8, 0.99),
                'lambda_critic': (0.9, 0.99),
                'noise_std': (0.01, 1.0),
                'gamma': (0.95, 0.999),
                'reward_scale': (0.1, 20.0),
                'td_clip': (1.0, 10.0)
            }
        
        results, best_params, best_score, search_dir = random_search(
            param_ranges, args.n_trials, args.num_episodes, show_output
        )
    
    # Save results
    output = {
        'method': args.method,
        'num_episodes': args.num_episodes,
        'results': results,
        'best_params': best_params,
        'best_score': best_score
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SEARCH COMPLETE!")
    print(f"{'='*80}")
    print(f"Best score: {best_score:.2f}")
    print(f"\nBest parameters:")
    if best_params:
        for k, v in best_params.items():
            if isinstance(v, float):
                if v < 0.01:
                    print(f"  {k:15s}: {v:.2e}")
                else:
                    print(f"  {k:15s}: {v:.4f}")
            else:
                print(f"  {k:15s}: {v}")
    print(f"\nResults saved to {args.output}")
    
    # Show top 5 configurations
    if results:
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        print(f"\nTop 5 configurations:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. Score: {result['score']:.2f} (run_id: {result['run_id']})")
    
    # Note about search results location
    print(f"\n📁 All search results saved in: {search_dir}")
    print(f"   - Individual run metrics: {search_dir}/<run_id>/metrics.json")
    print(f"   - Combined results: {search_dir}/search_results.json")

if __name__ == "__main__":
    main()