#!/usr/bin/env python3
"""
Visualize dream patterns from dreaming actor-critic training runs.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob


def load_dream_metrics(filepath):
    """Load metrics containing dream statistics."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def plot_dream_patterns(metrics, save_path=None):
    """Create comprehensive visualization of dream patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dreaming Actor-Critic Analysis', fontsize=16)
    
    returns = metrics['returns']
    dream_stats = metrics.get('dream_stats', {})
    
    # 1. Learning curve with dream markers
    ax1 = axes[0, 0]
    episodes = range(1, len(returns) + 1)
    ax1.plot(episodes, returns, 'b-', alpha=0.7, label='Episode returns')
    
    # Add rolling average
    window = 20
    if len(returns) >= window:
        rolling_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(returns) + 1), rolling_avg, 'r-', linewidth=2, label=f'{window}-ep average')
    
    # Mark dream episodes
    dream_episodes = dream_stats.get('dream_episodes', [])
    if dream_episodes:
        # Calculate cumulative dream episodes
        cum_dream_eps = []
        cum_sum = 0
        for interval in dream_episodes:
            cum_sum += interval
            cum_dream_eps.append(cum_sum)
        
        for ep in cum_dream_eps:
            if ep <= len(returns):
                ax1.axvline(x=ep, color='purple', alpha=0.3, linestyle='--')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Learning Curve with Dream Events')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sleep pressure history
    ax2 = axes[0, 1]
    pressure_history = dream_stats.get('sleep_pressure_history', [])
    if pressure_history:
        episodes_p = [p['episode'] for p in pressure_history]
        trace_sat = [p['trace_saturation'] for p in pressure_history]
        td_var = [p['td_variance'] for p in pressure_history]
        action_ent = [p['action_entropy'] for p in pressure_history]
        combined = [p['combined_pressure'] for p in pressure_history]
        
        ax2.plot(episodes_p, trace_sat, label='Trace saturation', alpha=0.7)
        ax2.plot(episodes_p, td_var, label='TD variance', alpha=0.7)
        ax2.plot(episodes_p, action_ent, label='Action entropy', alpha=0.7)
        ax2.plot(episodes_p, combined, 'k-', linewidth=2, label='Combined pressure')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Dream threshold')
        
        ax2.set_xlabel('Episodes since last dream')
        ax2.set_ylabel('Pressure value')
        ax2.set_title('Sleep Pressure Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Dream frequency over time
    ax3 = axes[1, 0]
    if dream_episodes:
        dream_intervals = dream_episodes
        dream_numbers = range(1, len(dream_intervals) + 1)
        
        ax3.bar(dream_numbers, dream_intervals, alpha=0.7, color='purple')
        ax3.set_xlabel('Dream number')
        ax3.set_ylabel('Episodes between dreams')
        ax3.set_title('Dream Frequency Over Time')
        
        # Add trend line
        if len(dream_intervals) > 5:
            z = np.polyfit(dream_numbers, dream_intervals, 1)
            p = np.poly1d(z)
            ax3.plot(dream_numbers, p(dream_numbers), "r--", alpha=0.8, 
                    label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
            ax3.legend()
        
        ax3.grid(True, alpha=0.3)
    
    # 4. Dream effectiveness
    ax4 = axes[1, 1]
    dream_improvements = dream_stats.get('dream_improvements', [])
    if dream_improvements:
        # Extract dream success rates
        success_rates = []
        for improvement in dream_improvements:
            rewards = improvement['dream_rewards']
            baseline = improvement['pre_dream_avg']
            if rewards:
                success_rate = sum(1 for r in rewards if r > baseline) / len(rewards)
                success_rates.append(success_rate * 100)
        
        if success_rates:
            dream_nums = range(1, len(success_rates) + 1)
            ax4.plot(dream_nums, success_rates, 'go-', markersize=8, alpha=0.7)
            ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Dream session')
            ax4.set_ylabel('Success rate (%)')
            ax4.set_title('Dream Success Rate (% better than baseline)')
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dream analysis to {save_path}")
    else:
        plt.show()


def plot_dream_comparison(metrics_list, labels, save_path=None):
    """Compare dream patterns across multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Dream Pattern Comparison', fontsize=16)
    
    # 1. Dream frequency comparison
    ax1 = axes[0]
    for metrics, label in zip(metrics_list, labels):
        dream_stats = metrics.get('dream_stats', {})
        dream_episodes = dream_stats.get('dream_episodes', [])
        if dream_episodes:
            cumulative_dreams = np.arange(1, len(dream_episodes) + 1)
            cumulative_episodes = np.cumsum(dream_episodes)
            ax1.plot(cumulative_episodes, cumulative_dreams, '-o', alpha=0.7, label=label)
    
    ax1.set_xlabel('Total episodes')
    ax1.set_ylabel('Total dreams')
    ax1.set_title('Cumulative Dream Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average dream interval comparison
    ax2 = axes[1]
    avg_intervals = []
    for metrics, label in zip(metrics_list, labels):
        dream_stats = metrics.get('dream_stats', {})
        dream_episodes = dream_stats.get('dream_episodes', [])
        if dream_episodes:
            avg_interval = np.mean(dream_episodes)
            avg_intervals.append((label, avg_interval))
    
    if avg_intervals:
        labels_bar, values = zip(*avg_intervals)
        x = np.arange(len(labels_bar))
        bars = ax2.bar(x, values, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_bar, rotation=45, ha='right')
        ax2.set_ylabel('Average episodes between dreams')
        ax2.set_title('Average Dream Interval')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.annotate(f'{value:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dream comparison to {save_path}")
    else:
        plt.show()


def analyze_dream_run(metrics_path):
    """Print summary statistics for a dream run."""
    metrics = load_dream_metrics(metrics_path)
    dream_stats = metrics.get('dream_stats', {})
    
    print(f"\nDream Analysis for: {metrics_path}")
    print("=" * 60)
    
    # Basic statistics
    returns = metrics['returns']
    print(f"Total episodes: {len(returns)}")
    print(f"Final average return: {metrics.get('final_avg_return', 'N/A'):.2f}")
    
    # Dream statistics
    total_dreams = dream_stats.get('total_dreams', 0)
    dream_episodes = dream_stats.get('dream_episodes', [])
    
    if total_dreams > 0:
        print(f"\nDream Statistics:")
        print(f"  Total dreams: {total_dreams}")
        print(f"  Average interval: {np.mean(dream_episodes):.1f} episodes")
        print(f"  Min interval: {min(dream_episodes)} episodes")
        print(f"  Max interval: {max(dream_episodes)} episodes")
        
        # Analyze trend
        if len(dream_episodes) > 5:
            first_half = np.mean(dream_episodes[:len(dream_episodes)//2])
            second_half = np.mean(dream_episodes[len(dream_episodes)//2:])
            trend = "increasing" if second_half > first_half else "decreasing"
            print(f"  Interval trend: {trend} (first half: {first_half:.1f}, second half: {second_half:.1f})")
    
    # Hyperparameters
    print(f"\nHyperparameters:")
    for key, value in metrics.get('hyperparameters', {}).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize dream patterns from training')
    parser.add_argument('metrics_files', nargs='+', 
                        help='Path(s) to metrics.json file(s)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple runs')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of displaying')
    parser.add_argument('--analyze', action='store_true',
                        help='Print analysis summary')
    
    args = parser.parse_args()
    
    # Expand wildcards
    all_files = []
    for pattern in args.metrics_files:
        matched = glob.glob(pattern)
        if matched:
            all_files.extend(matched)
        else:
            all_files.append(pattern)
    
    if args.analyze:
        for filepath in all_files:
            if os.path.exists(filepath):
                analyze_dream_run(filepath)
    
    if args.compare and len(all_files) > 1:
        # Load all metrics
        metrics_list = []
        labels = []
        for filepath in all_files:
            if os.path.exists(filepath):
                metrics_list.append(load_dream_metrics(filepath))
                # Create label from path
                label = os.path.basename(os.path.dirname(filepath))
                labels.append(label)
        
        if metrics_list:
            plot_dream_comparison(metrics_list, labels, args.save)
    else:
        # Single file visualization
        if all_files and os.path.exists(all_files[0]):
            metrics = load_dream_metrics(all_files[0])
            plot_dream_patterns(metrics, args.save)