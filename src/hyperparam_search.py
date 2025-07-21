#!/usr/bin/env python3
import subprocess
import itertools
import json
import numpy as np
import random
import argparse
from datetime import datetime
import os
import sys

try:
    from cartpole import get_run_directory, ensure_directory_exists
except ImportError:
    from .cartpole import get_run_directory, ensure_directory_exists

def run_experiment(params, run_id, num_episodes=100, show_output=True, refinement_info=None, out_dir=None, num_envs=1):
    """Run a single experiment with given hyperparameters."""
    # Choose which script to use based on num_envs
    if num_envs > 1:
        script = 'src/cartpole_vec.py'
    else:
        script = 'src/cartpole.py'
    
    cmd = [
        sys.executable, script,
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
        '--save-checkpoint', f'{run_id}_checkpoint.pth',
        '--run-id', run_id
    ]
    
    # Add num-envs if using vectorized version
    if num_envs > 1:
        cmd.extend(['--num-envs', str(num_envs)])
    
    # Add output directory if specified
    if out_dir:
        cmd.extend(['--out-dir', out_dir])
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {run_id}")
    print(f"{'='*80}")
    
    if refinement_info:
        print(f"Refining:")
        print(f"  orig_run    : {refinement_info['orig_run']}")
        print(f"  method      : {refinement_info['method']}")
        if 'top_percent' in refinement_info:
            print(f"  top_percent : {refinement_info['top_percent']*100:.0f}%")
        if 'noise_level' in refinement_info:
            print(f"  noise_level : {refinement_info['noise_level']}")
        if 'seed_run' in refinement_info:
            print(f"  seed_run    : {refinement_info['seed_run']}")
        print()
    
    print(f"Parameters:")
    for k, v in params.items():
        if isinstance(v, float):
            param_str = f"{v:.2e}" if v < 0.01 else f"{v:.4f}"
            
            # Add range info if refinement provided it
            if refinement_info and 'param_ranges' in refinement_info and k in refinement_info['param_ranges']:
                old_range = refinement_info['param_ranges'][k]['old']
                new_range = refinement_info['param_ranges'][k]['new']
                if isinstance(old_range, tuple) and isinstance(new_range, tuple):
                    print(f"  {k:15s}: {param_str} [{new_range[0]:.2e}, {new_range[1]:.2e}] from [{old_range[0]:.2e}, {old_range[1]:.2e}]")
                else:
                    print(f"  {k:15s}: {param_str}")
            else:
                print(f"  {k:15s}: {param_str}")
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
            # Load the metrics file from the output directory
            if out_dir:
                metrics_file = os.path.join(out_dir, run_id, 'metrics.json')
            else:
                metrics_file = os.path.join('runs', 'singles', run_id, 'metrics.json')
            
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

def grid_search(param_grid, num_episodes=100, show_output=True, search_id=None, num_envs=1):
    """Perform grid search over hyperparameter combinations."""
    # Create search directory
    if search_id is None:
        search_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    search_dir = os.path.join('runs', 'searches', search_id)
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
        score, metrics, metrics_file = run_experiment(params, run_id, num_episodes, show_output, out_dir=search_dir, num_envs=num_envs)
        
        if score is not None:
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

def random_search(param_ranges, n_trials=50, num_episodes=100, show_output=True, search_id=None, refinement_info=None, num_envs=1):
    """Perform random search over hyperparameter ranges."""
    # Create search directory
    if search_id is None:
        search_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    search_dir = os.path.join('runs', 'searches', search_id)
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
        score, metrics, metrics_file = run_experiment(params, run_id, num_episodes, show_output, refinement_info, out_dir=search_dir, num_envs=num_envs)
        
        if score is not None:
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

def probe_nearby_search(top_runs, n_trials, num_episodes=100, show_output=True, noise_level=0.3, search_id=None, refine_dir=None, num_envs=1):
    """
    Perform random search by probing near top performers.
    
    Args:
        top_runs: List of top performing runs to probe around
        n_trials: Number of trials to run
        num_episodes: Number of episodes per trial
        show_output: Whether to show training output
        noise_level: Amount of noise to add to parameters
        search_id: Optional identifier for the search
    
    Returns:
        results, best_params, best_score, search_dir
    """
    if search_id is None:
        search_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    search_dir = get_run_directory('searches', search_id)
    ensure_directory_exists(search_dir)
    
    results = []
    best_params = None
    best_score = -float('inf')
    
    print(f"\n{'='*80}")
    print(f"Starting probe-nearby search with {n_trials} trials")
    print(f"Probing around {len(top_runs)} top performers with noise level {noise_level}")
    print(f"{'='*80}\n")
    
    for i in range(n_trials):
        # Sample parameters near a top performer
        seed_run = random.choice(top_runs)
        params = sample_near_top_performer(top_runs, noise_level)
        
        # Create refinement info
        refinement_info = {
            'orig_run': refine_dir or 'unknown',
            'method': 'probe-nearby',
            'noise_level': noise_level,
            'seed_run': seed_run['run_id'] if 'run_id' in seed_run else f"score={seed_run['score']:.2f}"
        }
        
        run_id = f"probe_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n[{i+1}/{n_trials}] Starting experiment...")
        
        score, metrics, metrics_file = run_experiment(params, run_id, num_episodes, show_output, refinement_info, out_dir=search_dir, num_envs=num_envs)
        
        if score is not None:
            results.append({
                'params': params,
                'score': score,
                'run_id': run_id
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"\n🏆 NEW BEST SCORE: {best_score:.2f}")
        else:
            print(f"\n❌ Experiment failed")
    
    # Save search results in search directory
    search_results_path = os.path.join(search_dir, 'search_results.json')
    search_data = {
        'method': 'probe-nearby',
        'num_episodes': num_episodes,
        'results': results,
        'best_params': best_params,
        'best_score': best_score,
        'refinement_info': {
            'n_top_runs': len(top_runs),
            'noise_level': noise_level
        }
    }
    with open(search_results_path, 'w') as f:
        json.dump(search_data, f, indent=2)
    print(f"\nSearch results saved to {search_results_path}")
    
    return results, best_params, best_score, search_dir

def load_previous_results(search_dir):
    """Load search results from a previous run."""
    search_results_path = os.path.join(search_dir, 'search_results.json')
    if not os.path.exists(search_results_path):
        raise FileNotFoundError(f"No search results found at {search_results_path}")
    
    with open(search_results_path, 'r') as f:
        data = json.load(f)
    
    return data['results'], data['best_params'], data['best_score']

def calculate_shrunken_ranges(results, top_percent=0.2, expand=0.2, original_ranges=None):
    """Calculate refined parameter ranges based on top performers."""
    # Get top N% of runs
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    n_top = max(1, int(len(results) * top_percent))
    top_runs = sorted_results[:n_top]
    
    print(f"\nAnalyzing top {n_top} runs (top {top_percent*100:.0f}%) for refinement...")
    
    # Find min/max for each parameter
    param_names = list(top_runs[0]['params'].keys())
    new_ranges = {}
    range_info = {}
    
    for param in param_names:
        values = [run['params'][param] for run in top_runs]
        min_val, max_val = min(values), max(values)
        
        # Expand range by expand factor
        range_size = max_val - min_val
        if range_size == 0:  # All top runs had same value
            # Use 20% of the value as range
            range_size = abs(min_val) * 0.2
        
        new_min = min_val - range_size * expand
        new_max = max_val + range_size * expand
        
        # Ensure positive values for parameters that should be positive
        if param in ['lr_actor', 'lr_critic', 'noise_std', 'reward_scale', 'td_clip']:
            new_min = max(new_min, 1e-8)
        
        new_ranges[param] = (new_min, new_max)
        
        # Store both old and new ranges for display
        if original_ranges and param in original_ranges:
            range_info[param] = {
                'old': original_ranges[param],
                'new': (new_min, new_max)
            }
        
        print(f"  {param}: [{min_val:.2e}, {max_val:.2e}] -> [{new_min:.2e}, {new_max:.2e}]")
    
    return new_ranges, range_info

def sample_near_top_performer(top_runs, noise_level=0.3):
    """Sample parameters near a randomly selected top performer."""
    # Parameter bounds for clipping
    param_bounds = {
        'lr_actor': (1e-6, 1e-2),
        'lr_critic': (1e-5, 1e-1),
        'lambda_actor': (0.0, 1.0),
        'lambda_critic': (0.0, 1.0),
        'gamma': (0.0, 1.0),
        'noise_std': (1e-3, 1.0),
        'reward_scale': (0.1, 100.0),
        'td_clip': (0.1, 20.0)
    }
    
    # Pick random top performer
    seed_run = random.choice(top_runs)
    
    # Add noise to each parameter
    new_params = {}
    for param, value in seed_run['params'].items():
        if param in ['lr_actor', 'lr_critic', 'noise_std', 'reward_scale', 'td_clip']:
            # Multiplicative noise for log-scale parameters
            # Use log-space to ensure we explore orders of magnitude evenly
            log_value = np.log10(value)
            log_noise = np.random.uniform(-noise_level, noise_level)
            new_value = 10 ** (log_value + log_noise)
        else:
            # Additive noise for linear-scale parameters
            # Scale noise by parameter range
            param_range = param_bounds[param][1] - param_bounds[param][0]
            noise = np.random.uniform(-noise_level, noise_level) * param_range * 0.3
            new_value = value + noise
        
        # Apply bounds
        if param in param_bounds:
            min_val, max_val = param_bounds[param]
            new_params[param] = np.clip(new_value, min_val, max_val)
        else:
            new_params[param] = new_value
    
    return new_params

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for CartPole')
    parser.add_argument('--method', type=str, choices=['grid', 'random'], default='random',
                        help='Search method: grid or random')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of trials for random search')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes per trial')
    # Removed --output argument as results are saved in runs/search/<timestamp>/
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress training output from individual runs')
    parser.add_argument('--use-refined', action='store_true',
                        help='Use refined search space based on previous results')
    parser.add_argument('--refine', type=str, default=None,
                        help='Directory of previous search to refine from')
    parser.add_argument('--refine-method', type=str, 
                        choices=['shrink-hparam-space', 'probe-nearby'],
                        default='shrink-hparam-space',
                        help='Refinement strategy to use')
    parser.add_argument('--refine-top-percent', type=float, default=0.2,
                        help='Percentage of top runs to consider for refinement (default: 0.2)')
    parser.add_argument('--refine-noise', type=float, default=0.3,
                        help='Noise level for probe-nearby method (default: 0.3)')
    parser.add_argument('--refine-expand', type=float, default=0.2,
                        help='Expansion factor for shrink-hparam-space (default: 0.2)')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of parallel environments (default: 1, uses vectorized implementation if > 1)')
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
        
        results, best_params, best_score, search_dir = grid_search(param_grid, args.num_episodes, show_output, num_envs=args.num_envs)
        
    else:  # random search
        if args.refine:
            # Load previous results for refinement
            prev_results, prev_best_params, prev_best_score = load_previous_results(args.refine)
            print(f"\nLoaded {len(prev_results)} results from {args.refine}")
            print(f"Previous best score: {prev_best_score:.2f}")
            
            if args.refine_method == 'shrink-hparam-space':
                # Get original parameter ranges
                original_ranges = {
                    'lr_actor': (1e-5, 1e-2),
                    'lr_critic': (1e-4, 5e-2),
                    'lambda_actor': (0.8, 0.99),
                    'lambda_critic': (0.9, 0.99),
                    'noise_std': (0.01, 1.0),
                    'gamma': (0.95, 0.999),
                    'reward_scale': (0.1, 20.0),
                    'td_clip': (1.0, 10.0)
                }
                
                # Calculate refined ranges
                param_ranges, range_info = calculate_shrunken_ranges(
                    prev_results, 
                    top_percent=args.refine_top_percent,
                    expand=args.refine_expand,
                    original_ranges=original_ranges
                )
                
                # Pass refinement info to random search
                refinement_info = {
                    'orig_run': args.refine,
                    'method': 'shrink-hparam-space',
                    'top_percent': args.refine_top_percent,
                    'param_ranges': range_info
                }
                
                results, best_params, best_score, search_dir = random_search(
                    param_ranges, args.n_trials, args.num_episodes, show_output,
                    refinement_info=refinement_info, num_envs=args.num_envs
                )
            else:  # probe-nearby
                # Get top performers for probe-nearby method
                sorted_results = sorted(prev_results, key=lambda x: x['score'], reverse=True)
                n_top = max(1, int(len(prev_results) * args.refine_top_percent))
                top_runs = sorted_results[:n_top]
                print(f"\nUsing top {n_top} runs for probe-nearby refinement")
                
                # Run probe-nearby search
                results, best_params, best_score, search_dir = probe_nearby_search(
                    top_runs, args.n_trials, args.num_episodes, show_output,
                    noise_level=args.refine_noise,
                    refine_dir=args.refine, num_envs=args.num_envs
                )
        elif args.use_refined:
            from refined_search_config import get_search_config
            param_ranges = get_search_config('refined_random')
            print("Using refined random search configuration")
            results, best_params, best_score, search_dir = random_search(
                param_ranges, args.n_trials, args.num_episodes, show_output, num_envs=args.num_envs
            )
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
                param_ranges, args.n_trials, args.num_episodes, show_output, num_envs=args.num_envs
            )
    
    # Results are already saved in the search directory
    # No need for duplicate output file
    
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
    # Results location already printed above
    
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