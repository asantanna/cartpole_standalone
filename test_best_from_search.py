#!/usr/bin/env python3
"""
Find and test the best checkpoint from a hyperparameter search.
Usage: python test_best_from_search.py <search_directory> [--visual] [--num-episodes N]
"""

import argparse
import json
import os
import subprocess
import sys

def find_best_checkpoint(search_dir):
    """Find the best run from a search directory."""
    results_file = os.path.join(search_dir, 'search_results.json')
    
    if not os.path.exists(results_file):
        print(f"Error: No search_results.json found in {search_dir}")
        return None, None, None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    best_score = data.get('best_score', -float('inf'))
    best_params = data.get('best_params', {})
    
    # Find the run_id for the best score
    best_run_id = None
    for result in data.get('results', []):
        if result.get('score') == best_score:
            best_run_id = result.get('run_id')
            break
    
    if not best_run_id:
        print("Error: Could not find run_id for best score")
        return None, None, None
    
    # Construct checkpoint path
    checkpoint_path = os.path.join(search_dir, best_run_id, f"{best_run_id}_checkpoint.pth")
    
    return checkpoint_path, best_score, best_params

def test_checkpoint(checkpoint_path, visual=False, num_episodes=10):
    """Test a checkpoint by loading and running it."""
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Build command
    cmd = [
        sys.executable, 'src/cartpole.py',
        '--load-checkpoint', checkpoint_path,
        '--num-episodes', str(num_episodes),
        '--training-mode', 'false'  # Evaluation mode
    ]
    
    if visual:
        cmd.append('--visual')
    
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the test
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Find and test best checkpoint from search')
    parser.add_argument('search_dir', type=str, help='Search directory (e.g., runs/searches/search_20250721_001234)')
    parser.add_argument('--visual', action='store_true', help='Show visualization')
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='Number of episodes to test (default: 50)')
    
    args = parser.parse_args()
    
    # Find best checkpoint
    checkpoint_path, best_score, best_params = find_best_checkpoint(args.search_dir)
    
    if checkpoint_path:
        print(f"\nFound best run with score: {best_score:.2f}")
        print(f"Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print()
        
        # Test it
        test_checkpoint(checkpoint_path, args.visual, args.num_episodes)
    else:
        print("Could not find best checkpoint")
        
        # List available checkpoints
        import glob
        checkpoints = glob.glob(os.path.join(args.search_dir, "*/*_checkpoint.pth"))
        if checkpoints:
            print(f"\nFound {len(checkpoints)} checkpoints in search directory")
            print("You can test any of them with: python test_checkpoint.py <checkpoint_path>")

if __name__ == "__main__":
    main()