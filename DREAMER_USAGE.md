# Dreaming Actor-Critic Usage Guide

This guide explains how to train and visualize the biologically-inspired dreaming actor-critic implementation.

## Training

### Basic Training
```bash
# Train with default hyperparameters (dreams triggered by sleep pressure)
python src/cartpole_dreamer.py --num-episodes 100

# Train with visualization window
python src/cartpole_dreamer.py --visual --num-episodes 100

# Train with best hyperparameters from search
python src/cartpole_dreamer.py --best-config --num-episodes 100
```

### Testing Dream Behavior
```bash
# Force dreams every 10 episodes (useful for testing)
python src/cartpole_dreamer.py --force-dream-interval 10 --num-episodes 50 --save-metrics

# Save metrics and checkpoint for analysis
python src/cartpole_dreamer.py --num-episodes 200 --save-metrics --save-checkpoint dreamer.pt
```

### Command Line Options
- `--visual`: Show visualization window during training
- `--num-episodes N`: Number of training episodes (default: 500)
- `--force-dream-interval N`: Force dreams every N episodes (overrides adaptive triggering)
- `--save-metrics`: Save training metrics including dream statistics
- `--save-checkpoint FILE`: Save model checkpoint
- `--best-config`: Use best hyperparameters from search
- `--lr-actor`: Actor learning rate (default: 8e-5)
- `--lr-critic`: Critic learning rate (default: 2.5e-4)
- `--lambda-actor`: Actor eligibility trace decay (default: 0.88)
- `--lambda-critic`: Critic eligibility trace decay (default: 0.92)
- `--noise-std`: Action noise standard deviation (default: 0.05)

## Visualization

### Analyze Dream Patterns
```bash
# Print analysis of a single run
python src/visualize_dreams.py runs/singles/dreamer_*/metrics.json --analyze

# Create visualization plots
python src/visualize_dreams.py runs/singles/dreamer_*/metrics.json

# Save plots to file
python src/visualize_dreams.py runs/singles/dreamer_*/metrics.json --save dream_analysis.png
```

### Compare Multiple Runs
```bash
# Compare dream patterns across runs
python src/visualize_dreams.py runs/singles/dreamer_*/metrics.json --compare

# Compare with regular training
python src/visualize_learning.py "runs/singles/*/metrics.json" --compare
```

### Visualization Features
The dream visualization script creates four subplots:
1. **Learning Curve with Dream Events**: Episode returns with vertical lines marking dream phases
2. **Sleep Pressure Components**: Shows trace saturation, TD variance, and action entropy over time
3. **Dream Frequency**: Bar chart of episodes between dreams, with trend line
4. **Dream Effectiveness**: Success rate of dreams (% that improved over baseline)

## Quick Example

Here's a complete example to see dreaming in action:

```bash
# 1. Run a quick test with forced dreams every 10 episodes
python src/cartpole_dreamer.py --num-episodes 50 --force-dream-interval 10 --save-metrics --save-checkpoint test_dreamer.pt

# 2. Analyze the results
python src/visualize_dreams.py runs/singles/dreamer_*/metrics.json --analyze

# 3. Visualize dream patterns (if matplotlib works)
python src/visualize_dreams.py runs/singles/dreamer_*/metrics.json --save dream_patterns.png
```

## What to Look For

### Training Output
When training, you'll see output like:
```
💤 Entering dream phase at episode 10 (after 10 episodes)...
  Baseline performance: 25.0 (in 118 steps)
  Dreams 0-3: rewards ['26.5', '20.8', '24.7', '25.1']
  Dreams 4-7: rewards ['21.2', '24.2', '26.3', '-9.8']
  Dream results: 3/8 improved over baseline
  Consolidated top-3 dreams (avg improvement: 1.0)
```

### Sleep Pressure Monitoring
Every 10 episodes, you'll see sleep pressure metrics:
```
Episode  10     Return  103.2   Avg Return   81.1
  Sleep pressure: 1.00 (trace: 7.58, td_var: 0.011, action_ent: 1.10)
```

### Dream Statistics Summary
At the end of training:
```
Dream Statistics:
  Total dreams: 15
  Dream episodes: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  Average episodes between dreams: 1.0
```

## Tips and Insights

### Hyperparameter Effects
- **With `--best-config`**: Dreams happen almost every episode due to fast trace saturation
- **Default hyperparameters**: Show more natural sleep-wake cycles
- **High learning rates**: Less frequent dreams, traces don't saturate as quickly

### Performance Considerations
- Dream phase adds computational overhead (evaluates 8 policies sequentially)
- Each dream runs for up to 200 steps to evaluate the policy
- Consider using `--force-dream-interval` for consistent timing

### Adaptive vs Forced Dreams
- **Adaptive** (default): Dreams triggered by sleep pressure metrics
- **Forced** (`--force-dream-interval N`): Dreams every N episodes regardless of pressure
- Forced mode is useful for testing and comparing dream effectiveness

### Troubleshooting
- If dreams happen too frequently, the trace saturation threshold might need adjustment
- If dreams never trigger, check that training mode is enabled
- Sequential dreams mean training takes longer than standard actor-critic

## Understanding the Biology

The implementation is inspired by infant sleep patterns:
- **High initial dream frequency**: Like infants who spend 50% of sleep in REM
- **Decreasing dream needs**: As the policy improves, less "consolidation" needed
- **Sleep pressure**: Accumulates when learning stagnates, triggers dreams
- **Dream consolidation**: Successful parameter variations are integrated, like memory consolidation during sleep