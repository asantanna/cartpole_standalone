# Dreaming Actor-Critic Design Document

## Overview

This document describes a biologically-inspired reinforcement learning algorithm that incorporates "dreaming" phases to overcome learning plateaus. The algorithm is inspired by the observation that infants spend significantly more time in REM sleep than adults, suggesting that dreaming plays a crucial role in early learning.

## Biological Motivation

### Sleep Patterns Across Development
- **Infants (0-1 year)**: 14-17 hours sleep, ~50% REM
- **Children (3-5 years)**: 10-13 hours sleep, ~25% REM  
- **Adults**: 7-9 hours sleep, ~20% REM

This pattern suggests that as competence increases, less "mental simulation" is needed for learning.

### Key Insight
Rather than hard-coding sleep schedules, we let the need for dreaming emerge naturally from measurable learning signals, particularly **eligibility trace saturation**.

## Algorithm Design

### Core Concept
The algorithm alternates between two phases:

1. **Awake Phase**: Single-environment learning with eligibility traces (current implementation)
2. **Dream Phase**: Parallel exploration of parameter variations when learning stagnates

### Sleep Pressure Metrics

We monitor three signals that indicate when dreaming is needed:

#### 1. Eligibility Trace Saturation
```python
trace_saturation = torch.norm(self.e_a) / theoretical_max_norm
```
When traces approach their maximum values, it indicates the agent is stuck in repetitive patterns.

#### 2. Learning Progress Stagnation
```python
td_error_variance = variance(recent_td_errors)
```
Low variance in TD errors suggests the agent isn't learning anything new.

#### 3. Action Diversity Decay
```python
action_entropy = -sum(p * log(p)) for recent actions
```
Low entropy indicates stereotyped behavior requiring exploration.

### Dream Phase Mechanics

When sleep pressure exceeds threshold:

1. **Spawn Dream Environments**: Create 16-32 parallel environments
2. **Generate Dream Policies**: Each dream uses perturbed parameters
   ```python
   W_dream[i] = W_base + gaussian_noise(sigma=dream_intensity)
   ```
3. **Evaluate Dreams**: Run each dream for 10-50 timesteps
4. **Consolidate**: Update base policy toward successful dreams
   ```python
   W_base += learning_rate * weighted_average(successful_dreams)
   ```

### Adaptive Mechanisms

#### Dream Intensity (σ)
- Starts high (0.1) for exploration
- Decays with performance: `σ = σ_init * decay^episode`
- Increases if no improvement found

#### Dream Frequency
- Emerges naturally from saturation signals
- Early training: Frequent (high saturation rate)
- Late training: Rare (low saturation rate)

#### Dream Duration
- Short dreams (10 steps) for quick exploration
- Longer dreams (50 steps) if improvement detected

## Implementation Structure

### File: `cartpole_dreamer.py`

```python
class DreamingActorCritic(ActorCritic):
    def __init__(self, ...):
        super().__init__(...)
        # Sleep pressure tracking
        self.td_error_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        self.dream_stats = {...}
        
    def compute_sleep_pressure(self):
        # Returns combined pressure from all metrics
        
    def should_dream(self):
        # Threshold-based decision
        
    def dream_phase(self, num_dreams=16):
        # Execute parallel dreams
        
    def consolidate_dreams(self, dream_results):
        # Update toward successful variations
```

### Training Loop Modification

```python
for episode in range(num_episodes):
    # Normal awake training
    run_episode(env, ac)
    
    # Check sleep pressure
    if ac.should_dream():
        # Enter dream phase
        dream_results = ac.dream_phase()
        ac.consolidate_dreams(dream_results)
        
        # Log sleep patterns
        log_dream_event(episode, dream_results)
```

## Expected Behavior

### Early Training (High Sleep Pressure)
- Frequent dreams due to rapid trace saturation
- High parameter variance in dreams
- Large consolidation updates

### Mid Training (Moderate Sleep Pressure)
- Occasional dreams when stuck on local optima
- Moderate parameter variance
- Selective consolidation

### Late Training (Low Sleep Pressure)
- Rare dreams only for edge cases
- Small parameter perturbations
- Conservative updates

## Metrics and Visualization

Track and visualize:
1. **Sleep Pattern Graph**: Episodes vs. dream events
2. **Sleep Pressure Curves**: All three metrics over time
3. **Dream Effectiveness**: Success rate and improvement per dream
4. **Learning Curve Comparison**: With and without dreaming

## Advantages Over Standard Approaches

1. **Biologically Plausible**: Mirrors infant/child development patterns
2. **Adaptive**: Sleep emerges from need, not fixed schedule
3. **Efficient**: Parallel dreams only when beneficial
4. **Explainable**: Clear signals for why/when dreaming occurs

## Open Questions for Experimentation

1. **Threshold Tuning**: What saturation level triggers optimal dreaming?
2. **Dream Population Size**: Is 16 dreams enough? Too many?
3. **Consolidation Strategy**: Average? Weighted by success? Top-k?
4. **Cross-Task Transfer**: Do sleep patterns transfer between tasks?

## Implementation Results

### Key Findings

1. **Sequential Dreams Work**: Due to Isaac Gym limitations with parallel environments, we implemented sequential dreaming that evaluates parameter variations one at a time.

2. **High Sleep Pressure with Best Config**: With the optimized hyperparameters (very low actor learning rate), eligibility traces saturate extremely quickly, causing dreams after nearly every episode.

3. **Dream Consolidation Shows Promise**: Even with frequent dreaming, the consolidation mechanism successfully identifies and integrates improvements.

### Technical Limitations Discovered

- **No Parallel Environments**: Isaac Gym cannot create multiple environment instances in the same process (segmentation fault)
- **Sequential Overhead**: Each dream requires a full episode evaluation, making dreaming computationally expensive

### Observed Behavior

With default hyperparameters:
- Dreams occur when traces saturate (as designed)
- Dream frequency decreases as policy improves
- Performance benefits from successful dream consolidation

With optimized hyperparameters:
- Traces saturate almost immediately (within 1-2 episodes)
- Dreams occur after nearly every episode
- System behaves more like evolution strategies than sleep-wake cycles

### Future Improvements

1. **Adaptive Thresholds**: Dynamically adjust sleep pressure thresholds based on performance
2. **Dream Budgeting**: Limit dream frequency based on computational cost vs. benefit
3. **Hybrid Approach**: Use dreams primarily during early learning, switch to pure eligibility traces later
4. **Better Saturation Metrics**: Develop more nuanced measures of when learning is truly stuck

## Usage

```bash
# Train with dreaming actor-critic
python src/cartpole_dreamer.py --num-episodes 100

# Force dreams every N episodes (for testing)
python src/cartpole_dreamer.py --force-dream-interval 10

# Visualize dream patterns
python src/visualize_dreams.py runs/singles/dreamer_*/metrics.json --analyze
```