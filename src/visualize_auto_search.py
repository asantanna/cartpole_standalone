#!/usr/bin/env python3
"""
Visualize the progression of an automated hyperparameter search.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_auto_search_results(auto_dir):
    """Load results from an automated search directory."""
    summary_file = os.path.join(auto_dir, 'summary.json')
    progress_file = os.path.join(auto_dir, 'progress.json')
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = None
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = None
    
    return summary, progress

def plot_score_progression(summary, progress):
    """Plot the progression of best scores over iterations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    if summary:
        scores = summary['score_progression']
        iterations = range(1, len(scores) + 1)
        
        # Plot absolute scores
        ax1.plot(iterations, scores, 'b-o', linewidth=2, markersize=8)
        ax1.axhline(y=max(scores), color='r', linestyle='--', alpha=0.5, 
                    label=f'Best: {max(scores):.2f}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Score')
        ax1.set_title('Best Score Progression')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add stage names if available
        if 'search_history' in summary:
            for i, search in enumerate(summary['search_history']):
                ax1.annotate(search['stage'], 
                           xy=(i+1, scores[i]), 
                           xytext=(i+1, scores[i] + max(scores)*0.05),
                           fontsize=8, ha='center')
        
        # Plot improvement rate
        if len(scores) > 1:
            improvements = []
            for i in range(1, len(scores)):
                if scores[i-1] > 0:
                    improvement = (scores[i] - scores[i-1]) / scores[i-1] * 100
                else:
                    improvement = 0
                improvements.append(improvement)
            
            ax2.bar(range(2, len(scores) + 1), improvements, alpha=0.7)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Relative Improvement per Iteration')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_search_statistics(summary):
    """Plot statistics about the search process."""
    if not summary or 'search_history' not in summary:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    search_history = summary['search_history']
    
    # 1. Trials per iteration
    ax = axes[0, 0]
    iterations = [s['iteration'] + 1 for s in search_history]
    trials = [s['num_trials'] for s in search_history]
    ax.bar(iterations, trials, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Trials')
    ax.set_title('Trials per Iteration')
    ax.grid(True, alpha=0.3)
    
    # 2. Episodes per iteration
    ax = axes[0, 1]
    episodes = [s['episodes'] for s in search_history]
    ax.bar(iterations, episodes, alpha=0.7, color='green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Episodes per Trial')
    ax.set_title('Episodes per Iteration')
    ax.grid(True, alpha=0.3)
    
    # 3. Stage distribution (pie chart)
    ax = axes[1, 0]
    stage_counts = {}
    for s in search_history:
        stage = s['stage']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    ax.pie(stage_counts.values(), labels=stage_counts.keys(), autopct='%1.0f%%')
    ax.set_title('Stage Distribution')
    
    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    total_time = summary.get('total_time_seconds', 0)
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    summary_text = f"Search Summary\n\n"
    summary_text += f"Total iterations: {summary['total_iterations']}\n"
    summary_text += f"Total time: {hours}h {minutes}m\n"
    summary_text += f"Total trials: {sum(trials)}\n"
    summary_text += f"Best score: {summary['best_score']:.2f}\n"
    summary_text += f"Best iteration: {summary['best_iteration'] + 1}\n"
    summary_text += f"\nBest found in:\n{summary['best_search_dir']}"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Automated Search Statistics', fontsize=14)
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize automated search results')
    parser.add_argument('auto_dir', type=str, 
                        help='Directory containing automated search results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for plots')
    
    args = parser.parse_args()
    
    # Load results
    summary, progress = load_auto_search_results(args.auto_dir)
    
    if not summary and not progress:
        print(f"No results found in {args.auto_dir}")
        return
    
    # Create plots
    fig1 = plot_score_progression(summary, progress)
    fig2 = plot_search_statistics(summary)
    
    if args.output:
        if fig1:
            base_name = os.path.splitext(args.output)[0]
            fig1.savefig(f"{base_name}_progression.png", dpi=150, bbox_inches='tight')
        if fig2:
            fig2.savefig(f"{base_name}_statistics.png", dpi=150, bbox_inches='tight')
        print(f"Plots saved to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()