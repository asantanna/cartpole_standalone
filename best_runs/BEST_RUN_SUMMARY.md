# Best Run Summary

## Overview
This directory contains the best performing runs from hyperparameter searches.

## Current Best Run

**Date Preserved**: 2025-07-21  
**Source Search**: `search_20250721_004023`  
**Run ID**: `random_289_20250721_084226`  
**Score**: 333.64  
**Search Duration**: 816 minutes (13.6 hours)

### Hyperparameters
- **lr_actor**: 1.5277384851500572e-05
- **lr_critic**: 0.024159981498761918
- **lambda_actor**: 0.8194286929893821
- **lambda_critic**: 0.9615869404343597
- **noise_std**: 0.015641425426444783
- **gamma**: 0.9667744139228851
- **reward_scale**: 0.34034704159792284
- **td_clip**: 7.383066661842487

### Files Preserved
- `metrics.json` - Training metrics and episode returns
- `random_289_20250721_084226_checkpoint.pth` - Final model checkpoint
- `random_289_20250721_084226_checkpoint_best.pth` - Best performing checkpoint during training

### Performance
- **Training Score**: 333.64 (average over 200 episodes)
- **Notable**: The extremely low reward_scale (0.34) is unusual and may indicate the search found an unexpected optimum
- Returns during training ranged from ~250 to ~500

### How to Use
To test this checkpoint:
```bash
python test_checkpoint.py best_runs/searches/search_20250721_004023_best/random_289_20250721_084226/random_289_20250721_084226_checkpoint_best.pth --num-episodes 50 --visual
```

To continue training from this checkpoint:
```bash
python src/cartpole.py --load-checkpoint best_runs/searches/search_20250721_004023_best/random_289_20250721_084226/random_289_20250721_084226_checkpoint_best.pth --training-mode true --visual
```

### Note on Performance
While this run achieved the highest score during the search, the very low reward_scale (0.34) is unusual. For more typical CartPole performance, you may want to use the default hyperparameters or run a new search with reward_scale constrained to a more typical range (e.g., 5-20).