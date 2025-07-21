#!/usr/bin/env python3
"""
Find the best checkpoint from all search results.
Usage: python find_best_checkpoint.py [--test]
"""

import argparse
import json
import os
import glob
from datetime import datetime

def find_all_search_results():
    """Find all search directories and their results."""
    search_dirs = glob.glob('runs/searches/search_*')
    all_results = []
    
    for search_dir in search_dirs:
        results_file = os.path.join(search_dir, 'search_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
                all_results.append({
                    'search_dir': search_dir,
                    'best_score': data.get('best_score', -float('inf')),
                    'best_params': data.get('best_params', {}),
                    'method': data.get('method', 'unknown'),
                    'num_episodes': data.get('num_episodes', 'unknown'),
                    'results': data.get('results', [])
                })
    
    return all_results

def find_best_overall():
    """Find the best checkpoint across all searches."""
    all_results = find_all_search_results()
    
    if not all_results:
        print("No search results found in runs/searches/")
        return None
    
    # Find best overall
    best_search = max(all_results, key=lambda x: x['best_score'])
    
    # Find the specific run
    best_run_id = None
    for result in best_search['results']:
        if result.get('score') == best_search['best_score']:
            best_run_id = result.get('run_id')
            break
    
    if not best_run_id:
        print("Error: Could not find run_id for best score")
        return None
    
    # Construct checkpoint path
    checkpoint_path = os.path.join(best_search['search_dir'], best_run_id, f"{best_run_id}_checkpoint.pth")
    
    return {
        'checkpoint_path': checkpoint_path,
        'exists': os.path.exists(checkpoint_path),
        'score': best_search['best_score'],
        'params': best_search['best_params'],
        'search_dir': best_search['search_dir'],
        'run_id': best_run_id,
        'method': best_search['method'],
        'num_episodes': best_search['num_episodes']
    }

def print_top_runs(n=10):
    """Print the top N runs across all searches."""
    all_results = find_all_search_results()
    
    # Collect all individual runs
    all_runs = []
    for search in all_results:
        for result in search['results']:
            if result.get('score') is not None:
                run_id = result.get('run_id')
                checkpoint_path = os.path.join(search['search_dir'], run_id, f"{run_id}_checkpoint.pth")
                all_runs.append({
                    'score': result['score'],
                    'run_id': run_id,
                    'params': result['params'],
                    'search_dir': search['search_dir'],
                    'checkpoint_path': checkpoint_path,
                    'exists': os.path.exists(checkpoint_path)
                })
    
    # Sort by score
    all_runs.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nTop {min(n, len(all_runs))} runs:")
    print("-" * 100)
    print(f"{'Rank':<5} {'Score':<10} {'Run ID':<30} {'Checkpoint':<15} {'Search'}")
    print("-" * 100)
    
    for i, run in enumerate(all_runs[:n]):
        checkpoint_status = "✓ Exists" if run['exists'] else "✗ Missing"
        search_name = os.path.basename(run['search_dir'])
        print(f"{i+1:<5} {run['score']:<10.2f} {run['run_id']:<30} {checkpoint_status:<15} {search_name}")

def main():
    parser = argparse.ArgumentParser(description='Find best checkpoint from all searches')
    parser.add_argument('--test', action='store_true', help='Test the best checkpoint')
    parser.add_argument('--top', type=int, default=10, help='Show top N runs (default: 10)')
    
    args = parser.parse_args()
    
    # Find best checkpoint
    best = find_best_overall()
    
    if not best:
        return
    
    print("=" * 80)
    print("BEST CHECKPOINT FOUND")
    print("=" * 80)
    print(f"Score:       {best['score']:.2f}")
    print(f"Run ID:      {best['run_id']}")
    print(f"Search:      {best['search_dir']}")
    print(f"Method:      {best['method']}")
    print(f"Episodes:    {best['num_episodes']}")
    print(f"Checkpoint:  {best['checkpoint_path']}")
    print(f"Exists:      {'YES ✓' if best['exists'] else 'NO ✗'}")
    
    if best['exists']:
        print("\nTo test this checkpoint:")
        print(f"  python test_checkpoint.py {best['checkpoint_path']} --num-episodes 50")
        print("\nTo load in Python:")
        print(f"  checkpoint = torch.load('{best['checkpoint_path']}')")
    else:
        print("\n⚠️  WARNING: Checkpoint file does not exist!")
        print("This search may have been run before checkpoint saving was enabled.")
    
    print("\nBest parameters:")
    for k, v in best['params'].items():
        print(f"  {k:15s}: {v}")
    
    # Show top runs
    print_top_runs(args.top)
    
    # Test if requested
    if args.test and best['exists']:
        print("\n" + "=" * 80)
        print("TESTING BEST CHECKPOINT")
        print("=" * 80)
        import subprocess
        subprocess.run([
            'python', 'test_checkpoint.py', 
            best['checkpoint_path'], 
            '--num-episodes', '50'
        ])

if __name__ == "__main__":
    main()