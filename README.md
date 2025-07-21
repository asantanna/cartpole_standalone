# CartPole Continuous Control with Actor-Critic

This repository contains an implementation of an eligibility trace-based actor-critic algorithm for the CartPole environment in Isaac Gym, along with hyperparameter search tools.

## Key Features

- **Continuous action space** CartPole control using Isaac Gym
- **Actor-Critic with eligibility traces** for efficient credit assignment
- **Systematic hyperparameter search** with grid and random search options
- **Visualization tools** for analyzing learning curves and hyperparameter impact
- **Checkpoint save/load** with automatic training mode management
- **Automatic best model saving** during training
- **Organized runs directory** structure for experiments and searches

## Quick Start

### Basic Training
```bash
# Train with default hyperparameters (optimized based on search)
python src/cartpole.py

# Train with the best configuration found
python src/cartpole.py --best-config

# Train with visualization
python src/cartpole.py --visual

# Train for specific number of episodes
python src/cartpole.py --num-episodes 200

# Train and save checkpoint with metrics
python src/cartpole.py --best-config --save-checkpoint model.pt --save-metrics
# Creates runs/singles/<timestamp>/model.pt and runs/singles/<timestamp>/metrics.json

# Load checkpoint for evaluation (no learning)
python src/cartpole.py --load-checkpoint model.pt
# Automatically searches runs/singles/*/model.pt if not found

# Load checkpoint but continue training
python src/cartpole.py --load-checkpoint model.pt --training-mode true

# Train with automatic best model saving
python src/cartpole.py --best-config --save-checkpoint training.pt --save-metrics
# Creates runs/singles/<timestamp>/training.pt (final) and training_best.pt (best avg return)
```

### Hyperparameter Search
```bash
# Random search with refined parameter space
python src/hyperparam_search.py --method random --n-trials 20 --use-refined
# Creates runs/searches/search_<timestamp>/ with all results

# Grid search
python src/hyperparam_search.py --method grid --use-refined
# Creates runs/searches/search_<timestamp>/ with all results

# Quick test search
python src/hyperparam_search.py --n-trials 5 --num-episodes 50
# Results saved in runs/searches/search_<timestamp>/search_results.json
```

### Visualization
```bash
# Plot single learning curve
python src/visualize_learning.py runs/singles/train_20250720_123456/metrics.json

# Compare multiple runs (automatic for wildcards)
python src/visualize_learning.py "runs/singles/*/metrics.json"

# Show only the best run from multiple runs
python src/visualize_learning.py "runs/singles/*/metrics.json" --best-only

# Force comparison mode for single file
python src/visualize_learning.py runs/singles/train_20250720_123456/metrics.json --compare

# Analyze hyperparameter search results
python src/visualize_learning.py --search-results runs/searches/search_20250720_123456/search_results.json

# Plot all runs from everywhere
python src/visualize_learning.py "runs/**/metrics.json"
```

## Key Findings from Hyperparameter Search

The search revealed that **very low learning rates** are crucial for stable performance:
- Best actor learning rate: ~8e-5 (10-20x lower than typical)
- Best critic learning rate: ~2.5e-4
- Low exploration noise: 0.02-0.05
- High discount factor: 0.96-0.97
- Moderate reward scaling: 10-13

The best configuration achieved a score of **497.52**, demonstrating remarkably stable performance.

## Files and Directory Structure

### Scripts
- `src/cartpole.py` - Main training script with actor-critic implementation
- `src/hyperparam_search.py` - Hyperparameter search tool
- `src/refined_search_config.py` - Refined search spaces based on initial results
- `src/visualize_learning.py` - Visualization tools for learning curves
- `src/analyze_results.py` - Analysis tool for search results
- `scripts/run_tests.sh` - Script to run the test suite

### Tests
- `tests/` - Comprehensive pytest test suite for all functionality

### Directory Structure
```
runs/
├── singles/                 # Individual training runs
│   └── train_YYYYMMDD_HHMMSS/
│       ├── metrics.json     # Training metrics and returns
│       ├── model.pt         # Final checkpoint
│       └── model_best.pt    # Best checkpoint (highest avg return)
└── searches/                # Hyperparameter search results
    └── search_YYYYMMDD_HHMMSS/
        ├── search_results.json   # Combined search results
        └── trial_name/
            └── metrics.json      # Individual trial metrics
```

## Custom Hyperparameters

You can specify custom hyperparameters:
```bash
python src/cartpole.py --lr-actor 0.0001 --lr-critic 0.0005 --noise-std 0.1 --reward-scale 8.0
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run using the convenience script
./scripts/run_tests.sh
```

## Requirements

- Isaac Gym
- PyTorch
- NumPy
- Matplotlib (for visualization)

## Algorithm Details

The implementation uses:
- **Gaussian policy** for continuous actions with tanh squashing
- **Eligibility traces** for both actor and critic
- **TD error clipping** for stability
- **Xavier initialization** for network weights
- **Reward scaling** to prevent value explosion