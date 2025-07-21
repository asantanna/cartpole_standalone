#!/usr/bin/env python3
"""
Automated progressive hyperparameter search that iteratively refines the search space.
"""
import json
import numpy as np
import subprocess
import sys
import os
import time
from datetime import datetime, timedelta
import argparse
import glob
from typing import Dict, List, Tuple, Optional

# Parameter bounds for clipping after noise addition
PARAM_BOUNDS = {
    'lr_actor': (1e-6, 1e-2),      # Learning rates should be positive and reasonable
    'lr_critic': (1e-5, 1e-1),
    'lambda_actor': (0.0, 1.0),     # Lambda values must be in [0, 1]
    'lambda_critic': (0.0, 1.0),
    'gamma': (0.0, 1.0),            # Discount factor in [0, 1]
    'noise_std': (1e-3, 1.0),       # Positive noise
    'reward_scale': (0.1, 100.0),   # Positive scale
    'td_clip': (0.1, 20.0)          # Positive clipping value
}

# Default search stages configuration
DEFAULT_STAGES = [
    {
        'name': 'broad_exploration',
        'episodes': 20,
        'trials': 100,
        'method': 'random',
        'refine_method': None,
        'top_percent': 0.2,
        'noise_level': 0.3
    },
    {
        'name': 'initial_refinement',
        'episodes': 50,
        'trials': 50,
        'method': 'refine',
        'refine_method': 'adaptive',  # Will choose based on results
        'top_percent': 0.15,
        'noise_level': 0.2
    },
    {
        'name': 'fine_tuning',
        'episodes': 100,
        'trials': 30,
        'method': 'refine',
        'refine_method': 'probe-nearby',
        'top_percent': 0.1,
        'noise_level': 0.1
    },
    {
        'name': 'final_verification',
        'episodes': 200,
        'trials': 20,
        'method': 'refine',
        'refine_method': 'probe-nearby',
        'top_percent': 0.05,
        'noise_level': 0.05
    }
]


class ProgressiveSearch:
    def __init__(self, 
                 initial_param_ranges: Optional[Dict] = None,
                 max_iterations: int = 10,
                 improvement_threshold: float = 0.05,
                 time_budget_hours: Optional[float] = None,
                 stages: Optional[List[Dict]] = None,
                 output_dir: str = 'runs/searches/auto',
                 num_envs: int = 1):
        
        self.initial_param_ranges = initial_param_ranges or self._get_default_param_ranges()
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.time_budget = timedelta(hours=time_budget_hours) if time_budget_hours else None
        self.stages = stages or DEFAULT_STAGES
        self.output_dir = output_dir
        self.num_envs = num_envs
        
        # Track search history
        self.search_history = []
        self.best_score_history = []
        self.stage_times = []
        self.start_time = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_default_param_ranges(self) -> Dict:
        """Get default parameter ranges for initial search."""
        return {
            'lr_actor': (1e-5, 1e-3),
            'lr_critic': (1e-4, 1e-2),
            'lambda_actor': (0.8, 0.99),
            'lambda_critic': (0.9, 0.99),
            'noise_std': (0.01, 0.5),
            'gamma': (0.95, 0.99),
            'reward_scale': (0.1, 20.0),
            'td_clip': (1.0, 10.0)
        }
    
    def _run_search_command(self, cmd: List[str]) -> Tuple[Optional[str], bool]:
        """Run a hyperparameter search command and return the search directory."""
        try:
            print(f"\nRunning command: {' '.join(cmd)}")
            print("-" * 80)
            
            # Before starting, record existing search directories
            existing_dirs = set(glob.glob('runs/searches/search_*'))
            
            # Run the search with direct output to console
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                # Find the new search directory
                new_dirs = set(glob.glob('runs/searches/search_*')) - existing_dirs
                if new_dirs:
                    search_dir = sorted(new_dirs)[-1]  # Get the most recent
                    print(f"\n[Auto Search] Search completed successfully")
                    print(f"[Auto Search] Search directory: {search_dir}")
                    return search_dir, True
                else:
                    # Fallback: get the most recent directory
                    all_dirs = sorted(glob.glob('runs/searches/search_*'))
                    if all_dirs:
                        return all_dirs[-1], True
                    
            print(f"\nSearch failed with return code: {result.returncode}")
            return None, False
            
        except Exception as e:
            print(f"Error running search: {e}")
            return None, False
    
    def _load_search_results(self, search_dir: str) -> Dict:
        """Load search results from a directory."""
        results_file = os.path.join(search_dir, 'search_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    
    def _get_best_score(self, results: Dict) -> float:
        """Extract best score from search results."""
        if results and 'results' in results:
            scores = [r['score'] for r in results['results'] if r['score'] is not None]
            return max(scores) if scores else -float('inf')
        return -float('inf')
    
    def _should_continue(self, iteration: int) -> bool:
        """Determine if search should continue."""
        # Check iteration limit
        if iteration >= self.max_iterations:
            print(f"\nReached maximum iterations ({self.max_iterations})")
            return False
        
        # Check time budget
        if self.time_budget and (datetime.now() - self.start_time) > self.time_budget:
            print(f"\nExceeded time budget ({self.time_budget})")
            return False
        
        # Check improvement threshold
        if len(self.best_score_history) >= 2:
            recent_best = self.best_score_history[-1]
            previous_best = self.best_score_history[-2]
            
            if previous_best > 0:
                improvement = (recent_best - previous_best) / previous_best
                if improvement < self.improvement_threshold:
                    print(f"\nImprovement below threshold: {improvement:.2%} < {self.improvement_threshold:.2%}")
                    return False
        
        return True
    
    def _select_refinement_method(self, results: Dict) -> str:
        """Select appropriate refinement method based on results."""
        if not results or 'results' not in results:
            return 'probe-nearby'
        
        # Get top performers
        valid_results = [r for r in results['results'] if r['score'] is not None]
        if not valid_results:
            return 'probe-nearby'
        
        sorted_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)
        top_n = max(1, int(len(sorted_results) * 0.1))  # Top 10%
        top_results = sorted_results[:top_n]
        
        # Analyze parameter variance in top performers
        param_variances = {}
        for param in PARAM_BOUNDS.keys():
            values = [r['params'][param] for r in top_results]
            # Normalize variance by parameter range
            param_range = PARAM_BOUNDS[param][1] - PARAM_BOUNDS[param][0]
            normalized_std = np.std(values) / param_range
            param_variances[param] = normalized_std
        
        avg_variance = np.mean(list(param_variances.values()))
        
        # If parameters are consistent (low variance), shrink space
        # If parameters are scattered (high variance), probe nearby
        if avg_variance < 0.1:
            print(f"Low parameter variance ({avg_variance:.3f}) - using shrink-hparam-space")
            return 'shrink-hparam-space'
        else:
            print(f"High parameter variance ({avg_variance:.3f}) - using probe-nearby")
            return 'probe-nearby'
    
    def _save_progress(self, iteration: int):
        """Save current progress to file."""
        progress = {
            'iteration': iteration,
            'search_history': self.search_history,
            'best_score_history': self.best_score_history,
            'stage_times': self.stage_times,
            'start_time': self.start_time.isoformat() if self.start_time else None
        }
        
        progress_file = os.path.join(self.output_dir, 'progress.json')
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _load_progress(self) -> int:
        """Load previous progress if it exists."""
        progress_file = os.path.join(self.output_dir, 'progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                
            self.search_history = progress['search_history']
            self.best_score_history = progress['best_score_history']
            self.stage_times = progress['stage_times']
            if progress['start_time']:
                self.start_time = datetime.fromisoformat(progress['start_time'])
            
            return progress['iteration']
        return 0
    
    def run(self, resume: bool = False):
        """Run the progressive hyperparameter search."""
        print("=" * 80)
        print("AUTOMATED PROGRESSIVE HYPERPARAMETER SEARCH")
        print("=" * 80)
        
        # Initialize or resume
        start_iteration = 0
        if resume:
            start_iteration = self._load_progress()
            print(f"Resuming from iteration {start_iteration}")
        else:
            self.start_time = datetime.now()
        
        iteration = start_iteration
        previous_search_dir = None
        
        while self._should_continue(iteration):
            # Select stage based on iteration
            stage_idx = min(iteration, len(self.stages) - 1)
            stage = self.stages[stage_idx]
            
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1} - Stage: {stage['name']}")
            print(f"{'='*80}")
            print(f"Configuration:")
            print(f"  Episodes per trial: {stage['episodes']}")
            print(f"  Number of trials: {stage['trials']}")
            print(f"  Method: {stage.get('refine_method', stage['method'])}")
            if iteration > 0:
                print(f"  Top percent: {stage['top_percent']*100:.0f}%")
                print(f"  Noise level: {stage.get('noise_level', 0.2)}")
            
            stage_start = time.time()
            
            # Build command
            cmd = [
                sys.executable, 'src/hyperparam_search.py',
                '--num-episodes', str(stage['episodes']),
                '--n-trials', str(stage['trials'])
            ]
            
            # Add num-envs if > 1
            if self.num_envs > 1:
                cmd.extend(['--num-envs', str(self.num_envs)])
            
            if iteration == 0:
                # Initial search
                cmd.extend(['--method', stage['method']])
            else:
                # Refinement search
                cmd.extend(['--refine', previous_search_dir])
                
                # Select refinement method
                if stage['refine_method'] == 'adaptive':
                    prev_results = self._load_search_results(previous_search_dir)
                    refine_method = self._select_refinement_method(prev_results)
                else:
                    refine_method = stage['refine_method']
                
                cmd.extend(['--refine-method', refine_method])
                cmd.extend(['--refine-top-percent', str(stage['top_percent'])])
                
                if refine_method == 'probe-nearby':
                    cmd.extend(['--refine-noise', str(stage['noise_level'])])
            
            # Run the search
            search_dir, success = self._run_search_command(cmd)
            
            if not success:
                print("Search failed, stopping automated search")
                break
            
            # Load and analyze results
            results = self._load_search_results(search_dir)
            if results:
                best_score = self._get_best_score(results)
                self.best_score_history.append(best_score)
                
                # Record search info
                self.search_history.append({
                    'iteration': iteration,
                    'stage': stage['name'],
                    'search_dir': search_dir,
                    'best_score': best_score,
                    'num_trials': stage['trials'],
                    'episodes': stage['episodes']
                })
                
                print(f"\nIteration {iteration + 1} complete:")
                print(f"  Best score: {best_score:.2f}")
                if len(self.best_score_history) > 1:
                    prev_best = self.best_score_history[-2]
                    improvement = (best_score - prev_best) / max(abs(prev_best), 1.0)
                    print(f"  Improvement: {improvement:+.2%}")
                
                previous_search_dir = search_dir
            else:
                print("Failed to load search results")
                break
            
            # Record stage time
            stage_time = time.time() - stage_start
            self.stage_times.append(stage_time)
            print(f"  Stage time: {timedelta(seconds=int(stage_time))}")
            
            # Save progress
            self._save_progress(iteration + 1)
            
            iteration += 1
        
        # Final summary
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of the automated search."""
        print("\n" + "=" * 80)
        print("SEARCH SUMMARY")
        print("=" * 80)
        
        if not self.search_history:
            print("No searches completed")
            return
        
        print(f"Total iterations: {len(self.search_history)}")
        print(f"Total time: {datetime.now() - self.start_time}")
        print(f"Total trials: {sum(s['num_trials'] for s in self.search_history)}")
        
        print("\nScore progression:")
        for i, (search, score) in enumerate(zip(self.search_history, self.best_score_history)):
            print(f"  {i+1}. {search['stage']:20s} - Best: {score:7.2f}")
        
        # Find absolute best
        best_iteration = np.argmax(self.best_score_history)
        best_search = self.search_history[best_iteration]
        
        print(f"\nBest result found in iteration {best_iteration + 1}:")
        print(f"  Directory: {best_search['search_dir']}")
        print(f"  Score: {self.best_score_history[best_iteration]:.2f}")
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'total_iterations': len(self.search_history),
                'total_time_seconds': (datetime.now() - self.start_time).total_seconds(),
                'best_score': float(self.best_score_history[best_iteration]),
                'best_iteration': int(best_iteration),
                'best_search_dir': best_search['search_dir'],
                'score_progression': [float(s) for s in self.best_score_history],
                'search_history': self.search_history
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Automated progressive hyperparameter search')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum number of search iterations')
    parser.add_argument('--improvement-threshold', type=float, default=0.05,
                        help='Minimum improvement to continue (default: 0.05 = 5%%)')
    parser.add_argument('--time-budget', type=float, default=None,
                        help='Time budget in hours')
    parser.add_argument('--output-dir', type=str, 
                        default=f'runs/searches/auto_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Output directory for search results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous search')
    parser.add_argument('--config', type=str, default=None,
                        help='Custom stages configuration JSON file')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of parallel environments (default: 1, uses vectorized implementation if > 1)')
    
    args = parser.parse_args()
    
    # Load custom stages if provided
    stages = None
    if args.config:
        with open(args.config, 'r') as f:
            stages = json.load(f)
    
    # Create and run progressive search
    search = ProgressiveSearch(
        max_iterations=args.max_iterations,
        improvement_threshold=args.improvement_threshold,
        time_budget_hours=args.time_budget,
        stages=stages,
        output_dir=args.output_dir,
        num_envs=args.num_envs
    )
    
    search.run(resume=args.resume)


if __name__ == '__main__':
    main()