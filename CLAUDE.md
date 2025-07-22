# CartPole Standalone - AI Assistant Context

This document provides context for AI assistants working on this project.

## Project Overview

This is a biologically plausible implementation of an actor-critic algorithm with eligibility traces for the CartPole continuous control task in Isaac Gym. The project prioritizes biological plausibility and correct temporal credit assignment over training speed.

## Key Implementation Details

### Core Algorithm
- **Actor-Critic with Eligibility Traces**: Implements true eligibility traces for temporal credit assignment
- **Continuous Actions**: Uses Gaussian policy with tanh squashing for bounded continuous actions
- **Single Environment Only**: Designed specifically for single environments (see limitations below)

### Best Hyperparameters Found
From extensive hyperparameter search (300+ runs):
- `lr_actor`: 1.5e-5 (extremely low)
- `lr_critic`: 0.024 (much higher than actor)
- `lambda_actor`: 0.82
- `lambda_critic`: 0.96
- `noise_std`: 0.05
- `gamma`: 0.96
- `reward_scale`: 10.0 (note: search found 0.34 but this seems anomalous)

### Performance
- Achieves returns of 400-500 (environment maximum is 500)
- Training typically converges within 200-300 episodes
- Single environment runs at ~20 FPS with rendering

## Critical Limitations

### Vectorized Environments Are Incompatible
We discovered through extensive testing that eligibility traces are fundamentally incompatible with vectorized/parallel environments:

1. **Temporal Coherence Required**: Eligibility traces maintain decaying memory across time steps
2. **Asynchronous Resets Break Algorithm**: When environments reset at different times, traces mix incorrectly
3. **Batch Updates Fail**: Even with synchronization, accumulated updates are too large

**Important**: Any attempt to create a vectorized version (e.g., `cartpole_vec.py`) will fail. For parallel training, use different algorithms like PPO or A2C without eligibility traces.

## Project Structure

```
cartpole_standalone/
├── src/
│   ├── cartpole.py          # Main training script
│   ├── hyperparam_search.py # Hyperparameter search tool
│   └── visualize_learning.py # Learning curve visualization
├── best_runs/               # Best performing runs
├── runs/                    # All training runs
└── tests/                   # Test suite
```

## Key Findings

1. **Very Low Actor Learning Rate Critical**: The actor needs lr ~1e-5, much lower than typical
2. **Rendering Performance**: Setting `force_render=False` avoids double rendering with rl_games
3. **PhysX Optimizations**: TGS solver and tuned parameters improve simulation stability

## Common Issues and Solutions

### Black Screen/No Rendering
- Ensure `env.render()` is called in the training loop
- Check that `force_render=False` in environment creation

### Poor Learning Performance  
- Use the best hyperparameters or run a new search
- Ensure single environment only (no vectorization)
- Check reward scaling isn't too low (0.34 from search seems problematic)

### Import Errors
- Isaac Gym must be imported before PyTorch
- Follow the specific import order in the code

## Testing

Run tests with:
```bash
pytest tests/ -v
```

Note: Some tests require creating Isaac Gym environments and may fail if run together due to Isaac Gym limitations.

## Future Work

- Investigate alternative biologically plausible algorithms that might work with parallel environments
- Explore curriculum learning approaches
- Test on more complex continuous control tasks