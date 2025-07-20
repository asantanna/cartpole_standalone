#!/usr/bin/env python3
import json
import numpy as np
import argparse

def analyze_results(results_file):
    """Analyze hyperparameter search results."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    if not results:
        print("No results found!")
        return
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print(f"\nSearch Method: {data['method']}")
    print(f"Number of trials: {len(results)}")
    print(f"Episodes per trial: {data['num_episodes']}")
    
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    
    for i, result in enumerate(sorted_results[:10]):
        print(f"\nRank {i+1}: Score = {result['score']:.2f}")
        print("Parameters:")
        for param, value in sorted(result['params'].items()):
            if isinstance(value, float):
                if value < 0.01:
                    print(f"  {param:15s}: {value:.2e}")
                else:
                    print(f"  {param:15s}: {value:.4f}")
            else:
                print(f"  {param:15s}: {value}")
    
    # Analyze parameter importance
    print("\n" + "="*80)
    print("PARAMETER ANALYSIS")
    print("="*80)
    
    param_names = list(sorted_results[0]['params'].keys())
    
    for param in param_names:
        values = [r['params'][param] for r in results]
        scores = [r['score'] for r in results]
        
        # Calculate correlation
        if len(set(values)) > 1:  # Only if parameter varies
            correlation = np.corrcoef(values, scores)[0, 1]
            print(f"\n{param}:")
            print(f"  Range: [{min(values):.4f}, {max(values):.4f}]")
            print(f"  Correlation with score: {correlation:.3f}")
            
            # Best value
            best_idx = np.argmax(scores)
            print(f"  Best value: {values[best_idx]:.4f}")
    
    # Score statistics
    scores = [r['score'] for r in results]
    print("\n" + "="*80)
    print("SCORE STATISTICS")
    print("="*80)
    print(f"Best score:    {max(scores):.2f}")
    print(f"Worst score:   {min(scores):.2f}")
    print(f"Mean score:    {np.mean(scores):.2f}")
    print(f"Median score:  {np.median(scores):.2f}")
    print(f"Std deviation: {np.std(scores):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('--input', type=str, default='search_results.json',
                        help='Input results file')
    args = parser.parse_args()
    
    analyze_results(args.input)

if __name__ == "__main__":
    main()