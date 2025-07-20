#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os
from matplotlib.patches import Rectangle

def load_metrics(filename):
    """Load metrics from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_single_run(metrics, ax=None, label=None, alpha=1.0):
    """Plot learning curve for a single run."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    returns = metrics['returns']
    episodes = range(1, len(returns) + 1)
    
    # Plot raw returns
    ax.plot(episodes, returns, alpha=0.3*alpha, label=None if label else 'Episode returns')
    
    # Plot rolling average
    window = min(20, len(returns) // 5) if len(returns) > 5 else 0
    if window > 0 and len(returns) > window:
        rolling_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(returns) + 1), rolling_avg, 
                linewidth=2, alpha=alpha, label=label if label else f'Rolling avg (window={window})')
    
    return ax

def plot_comparison(metrics_files, title="Learning Curves Comparison"):
    """Plot multiple learning curves for comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Collect all metrics
    all_metrics = []
    for file in metrics_files:
        if os.path.exists(file):
            metrics = load_metrics(file)
            all_metrics.append({
                'file': file,
                'metrics': metrics,
                'final_score': metrics.get('final_avg_return', 0)
            })
    
    # Sort by final score
    all_metrics.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Plot top 5 and bottom 5
    n_plots = min(5, len(all_metrics))
    
    # Top performers
    ax1.set_title("Top Performers")
    for i in range(min(n_plots, len(all_metrics))):
        m = all_metrics[i]
        label = f"{os.path.basename(m['file'])} (score: {m['final_score']:.1f})"
        plot_single_run(m['metrics'], ax=ax1, label=label, alpha=0.8)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Bottom performers
    ax2.set_title("Bottom Performers")
    for i in range(max(0, len(all_metrics) - n_plots), len(all_metrics)):
        m = all_metrics[i]
        label = f"{os.path.basename(m['file'])} (score: {m['final_score']:.1f})"
        plot_single_run(m['metrics'], ax=ax2, label=label, alpha=0.8)
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Return")
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig

def plot_hyperparameter_analysis(search_results_file):
    """Analyze hyperparameter impact on performance."""
    with open(search_results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    if not results:
        print("No results to analyze")
        return
    
    # Extract parameters and scores
    param_names = list(results[0]['params'].keys())
    scores = [r['score'] for r in results]
    
    # Create subplots for each parameter
    n_params = len(param_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        values = [r['params'][param] for r in results]
        
        # Color by performance
        colors = plt.cm.viridis((np.array(scores) - min(scores)) / (max(scores) - min(scores) + 1e-8))
        
        scatter = ax.scatter(values, scores, c=colors, alpha=0.6, s=50)
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        
        # Log scale for learning rates and scales
        if param in ['lr_actor', 'lr_critic', 'noise_std', 'reward_scale']:
            ax.set_xscale('log')
        
        # Highlight best performer
        best_idx = np.argmax(scores)
        ax.scatter(values[best_idx], scores[best_idx], 
                  color='red', s=200, marker='*', edgecolor='black', linewidth=2)
    
    # Remove unused subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle("Hyperparameter Impact on Performance", fontsize=14)
    plt.tight_layout()
    return fig

def plot_best_run_details(metrics_file):
    """Detailed plot of the best run."""
    metrics = load_metrics(metrics_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Learning curve with statistics
    ax = axes[0, 0]
    returns = metrics['returns']
    episodes = range(1, len(returns) + 1)
    
    ax.plot(episodes, returns, alpha=0.3, color='blue', label='Episode returns')
    
    # Rolling statistics
    window = 20
    if len(returns) > window:
        rolling_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
        rolling_std = [np.std(returns[max(0, i-window):i]) for i in range(window, len(returns)+1)]
        
        x = range(window, len(returns) + 1)
        ax.plot(x, rolling_avg, 'b-', linewidth=2, label=f'Rolling mean (w={window})')
        ax.fill_between(x, 
                       rolling_avg - rolling_std, 
                       rolling_avg + rolling_std, 
                       alpha=0.2, color='blue', label='±1 std')
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Learning Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Return distribution
    ax = axes[0, 1]
    ax.hist(returns, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.1f}')
    ax.axvline(np.median(returns), color='orange', linestyle='--', label=f'Median: {np.median(returns):.1f}')
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.set_title("Return Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Hyperparameters
    ax = axes[1, 0]
    ax.axis('off')
    params = metrics['hyperparameters']
    param_text = "Hyperparameters:\n\n"
    for k, v in params.items():
        if isinstance(v, float):
            if v < 0.01:
                param_text += f"{k:15s}: {v:.2e}\n"
            else:
                param_text += f"{k:15s}: {v:.4f}\n"
        else:
            param_text += f"{k:15s}: {v}\n"
    
    ax.text(0.1, 0.9, param_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Performance statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate statistics
    stats_text = "Performance Statistics:\n\n"
    stats_text += f"Total episodes:     {len(returns)}\n"
    stats_text += f"Final avg return:   {metrics['final_avg_return']:.2f}\n"
    stats_text += f"Best return:        {max(returns):.2f}\n"
    stats_text += f"Worst return:       {min(returns):.2f}\n"
    stats_text += f"Mean return:        {np.mean(returns):.2f}\n"
    stats_text += f"Std deviation:      {np.std(returns):.2f}\n"
    
    # Calculate improvement
    early_avg = np.mean(returns[:20]) if len(returns) >= 20 else np.mean(returns)
    late_avg = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
    improvement = late_avg - early_avg
    stats_text += f"\nImprovement:        {improvement:+.2f}\n"
    stats_text += f"(last 20 - first 20 episodes)"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f"Detailed Analysis: {metrics.get('run_id', 'Unknown')}", fontsize=14)
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize CartPole learning curves')
    parser.add_argument('input', type=str, nargs='?', help='Input metrics file or pattern (e.g., "runs/single/*/metrics.json")')
    parser.add_argument('--search-results', type=str, help='Search results file for hyperparameter analysis')
    parser.add_argument('--output', type=str, help='Output plot file (default: show plot)')
    parser.add_argument('--best-only', action='store_true', help='Only show the best run when using wildcard')
    parser.add_argument('--compare', action='store_true', help='Force comparison mode even for single file')
    args = parser.parse_args()
    
    # Handle different input patterns and modes
    if args.search_results:
        # Hyperparameter analysis mode
        fig = plot_hyperparameter_analysis(args.search_results)
    
    elif args.input:
        # Expand input pattern
        files = glob.glob(args.input, recursive=True) if '*' in args.input else [args.input]
        
        if not files:
            print(f"No files found matching pattern: {args.input}")
            return
        
        # Filter out non-existent files
        files = [f for f in files if os.path.exists(f)]
        
        if not files:
            print("No valid files found")
            return
        
        # Determine visualization mode
        if args.best_only:
            # Show only the best run
            best_file = None
            best_score = -float('inf')
            for f in files:
                try:
                    m = load_metrics(f)
                    score = m.get('final_avg_return', -float('inf'))
                    if score > best_score:
                        best_score = score
                        best_file = f
                except:
                    continue
            
            if best_file:
                fig = plot_best_run_details(best_file)
            else:
                print("Could not find valid metrics files")
                return
        
        elif len(files) > 1 or args.compare:
            # Comparison mode (default for wildcards or forced)
            fig = plot_comparison(files)
        
        else:
            # Single file mode
            metrics = load_metrics(files[0])
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_single_run(metrics, ax=ax)
            ax.set_title(f"Learning Curve: {os.path.basename(files[0])}")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Return")
            ax.grid(True, alpha=0.3)
    
    else:
        print("Usage examples:")
        print("  python visualize_learning.py runs/single/*/metrics.json")
        print("  python visualize_learning.py runs/single/train_20250720_123456/metrics.json")
        print("  python visualize_learning.py runs/search/*/search_results.json --search-results")
        print("  python visualize_learning.py 'runs/**/*.json' --best-only")
        return
    
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()