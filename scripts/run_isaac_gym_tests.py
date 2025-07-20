#!/usr/bin/env python3
"""
Run Isaac Gym tests in subprocesses to avoid segmentation faults.
"""
import subprocess
import sys
import os
import argparse

def run_test_in_subprocess(test_path):
    """Run a single test in a subprocess."""
    cmd = [sys.executable, '-m', 'pytest', test_path, '-xvs']
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    proc = subprocess.run(
        cmd,
        env=os.environ.copy(),
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    return proc.returncode

def main():
    parser = argparse.ArgumentParser(description='Run Isaac Gym tests in subprocesses')
    parser.add_argument('--test', type=str, help='Specific test to run')
    parser.add_argument('--all-isaac-gym', action='store_true', 
                        help='Run all Isaac Gym tests')
    args = parser.parse_args()
    
    if args.test:
        # Run specific test
        exit_code = run_test_in_subprocess(args.test)
        sys.exit(exit_code)
    
    elif args.all_isaac_gym:
        # Run all Isaac Gym tests - each test method individually to avoid segfaults
        isaac_gym_tests = [
            # TestIsaacGymEnvironment
            'tests/test_training.py::TestIsaacGymEnvironment::test_isaac_gym_import',
            'tests/test_training.py::TestIsaacGymEnvironment::test_make_env',
            'tests/test_training.py::TestIsaacGymEnvironment::test_env_step',
            # TestTrainingFunction
            'tests/test_training.py::TestTrainingFunction::test_train_minimal_episodes',
            'tests/test_training.py::TestTrainingFunction::test_train_with_metrics',
            'tests/test_training.py::TestTrainingFunction::test_train_with_checkpoint',
            'tests/test_training.py::TestTrainingFunction::test_train_load_checkpoint',
            'tests/test_training.py::TestTrainingFunction::test_train_best_config',
            'tests/test_training.py::TestTrainingFunction::test_training_mode_override',
            'tests/test_training.py::TestTrainingFunction::test_best_model_saving',
            # TestTrainingEdgeCases
            'tests/test_training.py::TestTrainingEdgeCases::test_train_without_args',
            'tests/test_training.py::TestTrainingEdgeCases::test_checkpoint_resolution',
            'tests/test_training.py::TestTrainingEdgeCases::test_gpu_training',
            # Integration tests
            'tests/test_integration.py::TestSingleRunWorkflow::test_complete_single_run',
            'tests/test_integration.py::TestSingleRunWorkflow::test_multiple_sequential_runs',
            'tests/test_integration.py::TestSearchWorkflow::test_small_search_to_visualization',
            'tests/test_integration.py::TestSearchWorkflow::test_grid_search_integration',
            'tests/test_integration.py::TestCheckpointWorkflow::test_checkpoint_save_load_continue',
            'tests/test_integration.py::TestEndToEndScenarios::test_research_workflow',
        ]
        
        failed_tests = []
        for test in isaac_gym_tests:
            print(f"\n{'='*80}")
            print(f"Running: {test}")
            print(f"{'='*80}\n")
            
            exit_code = run_test_in_subprocess(test)
            if exit_code != 0:
                failed_tests.append(test)
        
        if failed_tests:
            print(f"\n{'='*80}")
            print(f"FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  - {test}")
            sys.exit(1)
        else:
            print(f"\n{'='*80}")
            print("All Isaac Gym tests passed!")
            sys.exit(0)
    
    else:
        print("Please specify --test <test_path> or --all-isaac-gym")
        sys.exit(1)

if __name__ == '__main__':
    main()