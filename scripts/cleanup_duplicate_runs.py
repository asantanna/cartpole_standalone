#!/usr/bin/env python3
"""
Clean up duplicate search runs from the singles directory.
Search runs (random_*, grid_*, probe_*) should only be in runs/searches/.
"""
import os
import shutil
import glob

def cleanup_duplicate_runs():
    """Remove search runs from runs/singles directory."""
    singles_dir = os.path.join('runs', 'singles')
    
    if not os.path.exists(singles_dir):
        print("No runs/singles directory found")
        return
    
    # Patterns for search runs that shouldn't be in singles
    search_patterns = ['random_*', 'grid_*', 'probe_*']
    
    removed_count = 0
    for pattern in search_patterns:
        runs = glob.glob(os.path.join(singles_dir, pattern))
        for run_path in runs:
            run_name = os.path.basename(run_path)
            print(f"Removing {run_name} from singles directory...")
            shutil.rmtree(run_path)
            removed_count += 1
    
    print(f"\nRemoved {removed_count} search runs from runs/singles/")
    
    # Show what remains
    remaining = os.listdir(singles_dir)
    print(f"\nRemaining runs in singles directory: {len(remaining)}")
    if remaining:
        print("Sample:", remaining[:5])

if __name__ == "__main__":
    cleanup_duplicate_runs()