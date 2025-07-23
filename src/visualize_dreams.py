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


def plot_dream_patterns(metrics, save_path=None, rolling_avg_len=20, filepath=None):
    """Create comprehensive visualization of dream patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dreaming Actor-Critic Analysis', fontsize=16)
    
    # Set window title
    if filepath:
        fig.canvas.manager.set_window_title(filepath)
    
    returns = metrics['returns']
    dream_stats = metrics.get('dream_stats', {})
    
    # 1. Learning curve with dream markers
    ax1 = axes[0, 0]
    episodes = range(1, len(returns) + 1)
    ax1.plot(episodes, returns, 'b-', alpha=0.7, label='Episode returns')
    
    # Add rolling average
    window = rolling_avg_len
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
        # Create x-axis as sequential indices since episodes_since_dream might be repetitive
        indices = range(len(pressure_history))
        
        # Use pressure values if available, otherwise compute from raw metrics
        if 'pressure_trace' in pressure_history[0]:
            # New format with pressures
            mental_fatigue = [p['pressure_trace'] for p in pressure_history]
            frustration = [p['pressure_td'] for p in pressure_history]
            boredom = [p['pressure_action'] for p in pressure_history]
        else:
            # Old format - compute pressures from raw metrics
            mental_fatigue = [min(1.0, p['trace_saturation'] / 0.8) for p in pressure_history]
            frustration = [1.0 - min(1.0, p['td_variance'] / 0.01) if p['td_variance'] < float('inf') else 0.0 for p in pressure_history]
            boredom = [1.0 - min(1.0, p['action_entropy'] / 0.1) if p['action_entropy'] < float('inf') else 0.0 for p in pressure_history]
        
        combined = [p['combined_pressure'] for p in pressure_history]
        
        # Plot lines with markers for better visibility
        ax2.plot(indices, mental_fatigue, 'o-', label='Mental fatigue', alpha=0.7, markersize=4)
        ax2.plot(indices, frustration, 's-', label='Frustration', alpha=0.7, markersize=4)
        ax2.plot(indices, boredom, '^-', label='Boredom', alpha=0.7, markersize=4)
        ax2.plot(indices, combined, 'k-', linewidth=2, label='Combined pressure')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Dream threshold')
        
        # Mark where dreams occurred
        dream_indices = [i for i, p in enumerate(pressure_history) if p['episode'] == 1]
        for idx in dream_indices:
            ax2.axvline(x=idx, color='purple', alpha=0.2, linestyle=':')
        
        ax2.set_xlabel('Measurement index')
        ax2.set_ylabel('Pressure value')
        ax2.set_title('Sleep Pressure Components Over Time')
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
    """Print comprehensive analysis of learning and dream effectiveness."""
    metrics = load_dream_metrics(metrics_path)
    dream_stats = metrics.get('dream_stats', {})
    returns = metrics['returns']
    
    print(f"\n{'='*70}")
    print(f"DREAM ANALYSIS: {os.path.basename(metrics_path)}")
    print(f"{'='*70}")
    
    # 1. LEARNING PROGRESS ANALYSIS
    print("\n📈 LEARNING PROGRESS")
    print("-" * 40)
    
    if len(returns) >= 10:
        # Compare early vs late performance
        early_avg = np.mean(returns[:10])
        late_avg = np.mean(returns[-10:])
        improvement = late_avg - early_avg
        improvement_pct = (improvement / abs(early_avg)) * 100 if early_avg != 0 else 0
        
        print(f"Early performance (first 10 eps):  {early_avg:>8.1f}")
        print(f"Late performance (last 10 eps):    {late_avg:>8.1f}")
        print(f"Improvement:                       {improvement:>8.1f} ({improvement_pct:+.1f}%)")
        
        # Check if learning is working
        if improvement > 0:
            print("✅ Learning is WORKING - performance improved")
        else:
            print("⚠️  Learning may be STUCK - no improvement")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total episodes:     {len(returns)}")
    print(f"  Best return:        {max(returns):.1f}")
    print(f"  Worst return:       {min(returns):.1f}")
    print(f"  Final avg (20 eps): {metrics.get('final_avg_return', 'N/A'):.1f}")
    
    # 2. DREAM EFFECTIVENESS ANALYSIS
    print("\n😴 DREAM EFFECTIVENESS")
    print("-" * 40)
    
    total_dreams = dream_stats.get('total_dreams', 0)
    dream_episodes = dream_stats.get('dream_episodes', [])
    dream_improvements = dream_stats.get('dream_improvements', [])
    
    if total_dreams > 0:
        print(f"Total dreams: {total_dreams}")
        print(f"Dream frequency: every {np.mean(dream_episodes):.1f} episodes")
        
        # Analyze dream success rates
        if dream_improvements:
            success_rates = []
            avg_improvements = []
            
            for imp in dream_improvements:
                if 'dream_rewards' in imp:
                    baseline = imp['pre_dream_avg']
                    rewards = imp['dream_rewards']
                    successes = sum(1 for r in rewards if r > baseline)
                    success_rate = (successes / len(rewards)) * 100 if rewards else 0
                    success_rates.append(success_rate)
                    
                    # Calculate average improvement
                    if rewards:
                        avg_imp = np.mean([r - baseline for r in rewards])
                        avg_improvements.append(avg_imp)
            
            if success_rates:
                avg_success = np.mean(success_rates)
                print(f"\nDream Success Metrics:")
                print(f"  Average success rate:    {avg_success:.1f}%")
                print(f"  Best dream session:      {max(success_rates):.1f}% success")
                print(f"  Worst dream session:     {min(success_rates):.1f}% success")
                
                if avg_improvements:
                    avg_dream_benefit = np.mean(avg_improvements)
                    print(f"  Avg reward improvement:  {avg_dream_benefit:+.2f}")
                
                # Check if dreams are helping
                if avg_success > 30:
                    print("✅ Dreams are HELPING - finding improvements")
                else:
                    print("⚠️  Dreams may be INEFFECTIVE - low success rate")
        
        # Analyze sleep patterns
        print(f"\nSleep Pattern Analysis:")
        if len(dream_episodes) > 5:
            first_half = np.mean(dream_episodes[:len(dream_episodes)//2])
            second_half = np.mean(dream_episodes[len(dream_episodes)//2:])
            
            if second_half > first_half * 1.2:
                print(f"  Pattern: MATURING (dreams less frequent)")
                print(f"  Early: every {first_half:.1f} eps → Late: every {second_half:.1f} eps")
                print("  ✅ Good sign - needing less exploration")
            elif second_half < first_half * 0.8:
                print(f"  Pattern: STRUGGLING (dreams more frequent)")
                print(f"  Early: every {first_half:.1f} eps → Late: every {second_half:.1f} eps")
                print("  ⚠️  May indicate learning difficulties")
            else:
                print(f"  Pattern: STABLE (consistent dream frequency)")
                print(f"  Averaging every {np.mean(dream_episodes):.1f} episodes")
    
    # 3. PRESSURE ANALYSIS
    print("\n🧠 MENTAL STATE PATTERNS")
    print("-" * 40)
    
    pressure_history = dream_stats.get('sleep_pressure_history', [])
    if pressure_history:
        # Find what triggers dreams most
        if 'pressure_trace' in pressure_history[0]:
            trace_pressures = [p['pressure_trace'] for p in pressure_history]
            td_pressures = [p['pressure_td'] for p in pressure_history]
            action_pressures = [p['pressure_action'] for p in pressure_history]
            
            avg_trace = np.mean(trace_pressures)
            avg_td = np.mean(td_pressures)
            avg_action = np.mean(action_pressures)
            
            print(f"Average pressure levels:")
            print(f"  Mental fatigue:  {avg_trace:.2f}")
            print(f"  Frustration:     {avg_td:.2f}")
            print(f"  Boredom:         {avg_action:.2f}")
            
            # Identify primary dream trigger
            pressures = {'Mental fatigue': avg_trace, 'Frustration': avg_td, 'Boredom': avg_action}
            primary_trigger = max(pressures, key=pressures.get)
            print(f"\nPrimary dream trigger: {primary_trigger}")
    
    # 4. RECOMMENDATIONS
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    # Based on analysis, provide recommendations
    recommendations = []
    
    # Check learning progress
    if len(returns) >= 10 and improvement <= 0:
        recommendations.append("• Learning is stuck or declining - try adjusting hyperparameters")
    
    # Check dream effectiveness
    if total_dreams > 0:
        if len(dream_episodes) > 0 and np.mean(dream_episodes) < 3:
            recommendations.append("• Dreams too frequent - consider raising dream threshold")
        
        # Check dream success
        if 'dream_improvements' in dream_stats and dream_stats['dream_improvements']:
            all_improvements = []
            for imp in dream_stats['dream_improvements']:
                if 'dream_rewards' in imp:
                    baseline = imp['pre_dream_avg']
                    rewards = imp['dream_rewards']
                    for r in rewards:
                        all_improvements.append(r - baseline)
            
            if all_improvements and np.mean(all_improvements) < -5:
                recommendations.append("• Dreams making performance worse - reduce dream noise or frequency")
            elif all_improvements and max(all_improvements) < 0:
                recommendations.append("• No successful dreams - consider different hyperparameters")
    else:
        recommendations.append("• No dreams occurred - consider lowering dream threshold")
    
    # Print recommendations or default message
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("• System appears to be working - continue monitoring")
    
    # Hyperparameters summary
    print(f"\n⚙️  Key Hyperparameters:")
    hp = metrics.get('hyperparameters', {})
    print(f"  Actor LR:     {hp.get('lr_actor', 'N/A')}")
    print(f"  Critic LR:    {hp.get('lr_critic', 'N/A')}")
    print(f"  Lambda (a/c): {hp.get('lambda_actor', 'N/A')} / {hp.get('lambda_critic', 'N/A')}")
    print(f"  Noise std:    {hp.get('noise_std', 'N/A')}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize dream patterns from training')
    parser.add_argument('metrics_files', nargs='*', 
                        help='Path(s) to metrics.json file(s). If not provided, uses latest dreamer run.')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple runs')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of displaying')
    parser.add_argument('--analyze', action='store_true',
                        help='Print analysis summary')
    parser.add_argument('--rolling-avg-len', type=int, default=20,
                        help='Window size for rolling average (default: 20)')
    
    args = parser.parse_args()
    
    # If no files provided, find the latest dreamer run
    if not args.metrics_files:
        import os
        dreamer_runs = glob.glob('runs/singles/dreamer_*/metrics.json')
        if dreamer_runs:
            # Get the most recent one
            latest = max(dreamer_runs, key=os.path.getmtime)
            args.metrics_files = [latest]
            print(f"Using latest dreamer run: {latest}")
        else:
            print("No dreamer runs found in runs/singles/dreamer_*/")
            print("Please specify a metrics file or run cartpole_dreamer.py first.")
            exit(1)
    
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
            plot_dream_patterns(metrics, args.save, args.rolling_avg_len, filepath=all_files[0])