#!/usr/bin/env python3
"""
Test a saved checkpoint to see its actual performance.
Usage: python test_checkpoint.py <checkpoint_path> [--visual] [--num-episodes N]
"""

import argparse
import subprocess
import sys
import os

def test_checkpoint(checkpoint_path, visual=False, num_episodes=10):
    """Test a checkpoint by loading and running it."""
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        
        # Try to find it in common locations
        base_name = os.path.basename(checkpoint_path)
        search_paths = [
            f"runs/singles/*/{base_name}",
            f"runs/searches/*/*/{base_name}",
            checkpoint_path
        ]
        
        import glob
        for pattern in search_paths:
            matches = glob.glob(pattern)
            if matches:
                checkpoint_path = matches[0]
                print(f"Found checkpoint at: {checkpoint_path}")
                break
        else:
            print("Could not find checkpoint file.")
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
    parser = argparse.ArgumentParser(description='Test a saved checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--visual', action='store_true', help='Show visualization')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to test (default: 10)')
    
    args = parser.parse_args()
    
    test_checkpoint(args.checkpoint, args.visual, args.num_episodes)

if __name__ == "__main__":
    main()