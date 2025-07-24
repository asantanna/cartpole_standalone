#!/usr/bin/env python3
"""
Test recovery performance of trained models after weight perturbation.

Usage:
    python test_recovery.py checkpoint.pth --num-episodes 50 --noise-std 0.01 0.1 0.5
"""
import argparse
import json
import subprocess
import os
import sys
import numpy as np
from datetime import datetime
import glob
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def ensure_directory_exists(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path

def run_cartpole_test(checkpoint_path, output_dir, num_episodes, disturb_weights=None, run_name="test", training_mode=True):
    """Run cartpole.py with specified parameters and return metrics."""
    cmd = [
        sys.executable, "src/cartpole.py",
        "--load-checkpoint", checkpoint_path,
        "--num-episodes", str(num_episodes),
        "--save-checkpoint",
        "--run-id", run_name,
        "--out-dir", output_dir,
        "--training-mode", "true" if training_mode else "false"
    ]
    
    if disturb_weights is not None:
        cmd.extend(["--disturb-weights", str(disturb_weights)])
    
    print(f"\nRunning: {' '.join(cmd[-6:])}")  # Print last part of command
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Run completed successfully")
        
        # Load metrics from the output directory
        metrics_file = os.path.join(output_dir, run_name, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Metrics file not found at {metrics_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running cartpole: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None

def calculate_performance_stats(returns, window=20):
    """Calculate average performance over the last 'window' episodes."""
    if len(returns) >= window:
        avg = np.mean(returns[-window:])
        std = np.std(returns[-window:])
    else:
        avg = np.mean(returns)
        std = np.std(returns)
    return avg, std

def run_recovery_tests(checkpoint_path, num_episodes, noise_levels, num_runs=1):
    """Run recovery tests for multiple noise levels."""
    # Create output directory
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = ensure_directory_exists(f"recovery_tests/{checkpoint_name}_{timestamp}")
    
    results = {
        'checkpoint': checkpoint_path,
        'num_episodes': num_episodes,
        'num_runs': num_runs,
        'noise_levels': noise_levels,
        'baseline': {},
        'noise_tests': {}
    }
    
    # Run baseline tests (no disturbance)
    print("\n" + "="*60)
    print("Running BASELINE tests (no disturbance)")
    print("="*60)
    
    baseline_dir = ensure_directory_exists(os.path.join(base_output_dir, "baseline"))
    baseline_returns = []
    
    for run_idx in range(num_runs):
        print(f"\nBaseline run {run_idx + 1}/{num_runs}")
        metrics = run_cartpole_test(
            checkpoint_path, 
            baseline_dir, 
            num_episodes,
            disturb_weights=None,
            run_name=f"run_{run_idx + 1}",
            training_mode=False  # Run baseline in evaluation mode
        )
        
        if metrics:
            returns = metrics.get('returns', [])
            baseline_returns.extend(returns[-20:])  # Last 20 episodes
            
            # Save individual run stats
            avg, std = calculate_performance_stats(returns)
            if 'runs' not in results['baseline']:
                results['baseline']['runs'] = []
            results['baseline']['runs'].append({
                'run_id': run_idx + 1,
                'avg_return': avg,
                'std_return': std,
                'returns': returns
            })
    
    # Calculate baseline statistics
    if baseline_returns:
        results['baseline']['avg_performance'] = np.mean(baseline_returns)
        results['baseline']['std_performance'] = np.std(baseline_returns)
        print(f"\nBaseline performance: {results['baseline']['avg_performance']:.2f} ± {results['baseline']['std_performance']:.2f}")
    
    # Test each noise level
    for noise_std in noise_levels:
        print(f"\n" + "="*60)
        print(f"Testing noise level: {noise_std}")
        print("="*60)
        
        noise_dir = ensure_directory_exists(os.path.join(base_output_dir, f"noise_{noise_std}"))
        results['noise_tests'][str(noise_std)] = {
            'noise_std': noise_std,
            'runs': []
        }
        
        all_initial_returns = []
        all_final_returns = []
        
        for run_idx in range(num_runs):
            print(f"\nNoise {noise_std} run {run_idx + 1}/{num_runs}")
            metrics = run_cartpole_test(
                checkpoint_path,
                noise_dir,
                num_episodes,
                disturb_weights=noise_std,
                run_name=f"run_{run_idx + 1}"
            )
            
            if metrics:
                returns = metrics.get('returns', [])
                
                # Calculate initial performance (first 10 episodes)
                initial_avg, initial_std = calculate_performance_stats(returns[:10], window=10)
                all_initial_returns.extend(returns[:10])
                
                # Calculate final performance (last 20 episodes)
                final_avg, final_std = calculate_performance_stats(returns)
                all_final_returns.extend(returns[-20:])
                
                # Calculate recovery percentage
                baseline_avg = results['baseline']['avg_performance']
                if baseline_avg - initial_avg > 0:
                    recovery_pct = (final_avg - initial_avg) / (baseline_avg - initial_avg) * 100
                else:
                    recovery_pct = 100.0 if final_avg >= baseline_avg else 0.0
                
                run_data = {
                    'run_id': run_idx + 1,
                    'initial_avg': float(initial_avg),
                    'initial_std': float(initial_std),
                    'final_avg': float(final_avg),
                    'final_std': float(final_std),
                    'recovery_percentage': float(recovery_pct),
                    'recovered_to_90pct': bool(final_avg >= 0.9 * baseline_avg),
                    'returns': returns
                }
                
                results['noise_tests'][str(noise_std)]['runs'].append(run_data)
                
                print(f"  Initial: {initial_avg:.2f} ± {initial_std:.2f}")
                print(f"  Final: {final_avg:.2f} ± {final_std:.2f}")
                print(f"  Recovery: {recovery_pct:.1f}%")
        
        # Calculate aggregate statistics for this noise level
        if all_initial_returns and all_final_returns:
            results['noise_tests'][str(noise_std)]['avg_initial'] = np.mean(all_initial_returns)
            results['noise_tests'][str(noise_std)]['std_initial'] = np.std(all_initial_returns)
            results['noise_tests'][str(noise_std)]['avg_final'] = np.mean(all_final_returns)
            results['noise_tests'][str(noise_std)]['std_final'] = np.std(all_final_returns)
            
            # Overall recovery rate
            baseline_avg = results['baseline']['avg_performance']
            avg_initial = results['noise_tests'][str(noise_std)]['avg_initial']
            avg_final = results['noise_tests'][str(noise_std)]['avg_final']
            
            if baseline_avg - avg_initial > 0:
                overall_recovery = (avg_final - avg_initial) / (baseline_avg - avg_initial) * 100
            else:
                overall_recovery = 100.0 if avg_final >= baseline_avg else 0.0
                
            results['noise_tests'][str(noise_std)]['overall_recovery_pct'] = float(overall_recovery)
            results['noise_tests'][str(noise_std)]['recovered_to_90pct'] = bool(avg_final >= 0.9 * baseline_avg)
    
    # Save results
    report_file = os.path.join(base_output_dir, "recovery_report.json")
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*60)
    print("RECOVERY TEST SUMMARY")
    print("="*60)
    print(f"Baseline performance: {results['baseline']['avg_performance']:.2f} ± {results['baseline']['std_performance']:.2f}")
    
    for noise_std in noise_levels:
        noise_data = results['noise_tests'][str(noise_std)]
        print(f"\nNoise level {noise_std}:")
        print(f"  Initial after disturbance: {noise_data['avg_initial']:.2f} ± {noise_data['std_initial']:.2f}")
        print(f"  Final after {num_episodes} episodes: {noise_data['avg_final']:.2f} ± {noise_data['std_final']:.2f}")
        print(f"  Recovery: {noise_data['overall_recovery_pct']:.1f}%")
        print(f"  Recovered to 90% of baseline: {'Yes' if noise_data['recovered_to_90pct'] else 'No'}")
    
    print(f"\nFull report saved to: {report_file}")
    
    # Generate visualization
    if MATPLOTLIB_AVAILABLE:
        generate_recovery_plot(results, base_output_dir)
    else:
        print("Warning: Matplotlib not available, skipping plot generation")
    
    return results

def generate_recovery_plot(results, output_dir):
    """Generate a plot showing recovery curves for all noise levels."""
    plt.figure(figsize=(10, 6))
    
    # Plot baseline average
    baseline_avg = results['baseline']['avg_performance']
    plt.axhline(y=baseline_avg, color='green', linestyle='--', label=f'Baseline ({baseline_avg:.1f})')
    plt.axhline(y=0.9 * baseline_avg, color='gray', linestyle=':', label='90% of baseline')
    
    # Plot recovery curves for each noise level
    for noise_str, noise_data in results['noise_tests'].items():
        # Average across all runs for this noise level
        all_returns = []
        max_len = max(len(run['returns']) for run in noise_data['runs'])
        
        for i in range(max_len):
            episode_returns = []
            for run in noise_data['runs']:
                if i < len(run['returns']):
                    episode_returns.append(run['returns'][i])
            if episode_returns:
                all_returns.append(np.mean(episode_returns))
        
        if all_returns:
            episodes = range(1, len(all_returns) + 1)
            plt.plot(episodes, all_returns, label=f'Noise={noise_str}', alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Recovery Performance After Weight Perturbation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = os.path.join(output_dir, "recovery_curves.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Recovery plot saved to: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description='Test recovery performance after weight perturbation')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='Number of episodes to run for each test (default: 50)')
    parser.add_argument('--noise-std', type=float, nargs='+', default=[0.01, 0.1, 0.5],
                        help='Noise standard deviations to test (default: 0.01 0.1 0.5)')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of runs per noise level (default: 1)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Run recovery tests
    run_recovery_tests(
        args.checkpoint,
        args.num_episodes,
        args.noise_std,
        args.num_runs
    )

if __name__ == "__main__":
    main()