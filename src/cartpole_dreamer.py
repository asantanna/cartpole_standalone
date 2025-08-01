#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Box bound precision lowered.*')
warnings.filterwarnings('ignore', message='.*version_base parameter is not specified.*')
warnings.filterwarnings('ignore', message='.*FBX.*')
warnings.filterwarnings('ignore', message='.*torch.load.*')
warnings.filterwarnings('ignore', message='.*weights_only.*')

import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'  # Suppress CUDA warnings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Suppress CUDA device warnings

# Suppress Isaac Gym logger
import logging
logging.getLogger().setLevel(logging.ERROR)


import time
import numpy as np
import argparse
import isaacgym
import isaacgymenvs
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
import json
from datetime import datetime
from collections import deque

#─── Directory helpers ─────────────────────────────────────────────────────────
def get_run_directory(run_type='singles', run_id=None):
    """Generate run directory path."""
    if run_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"dreamer_{timestamp}"
    
    run_dir = os.path.join('runs', run_type, run_id)
    return run_dir

def ensure_directory_exists(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path

def resolve_checkpoint_path(filepath):
    """Resolve checkpoint path, checking runs directories if needed."""
    # If file exists as-is, return it
    if os.path.exists(filepath):
        return filepath
    
    # Check in runs/singles/*/
    import glob
    patterns = [
        f"runs/singles/*/{os.path.basename(filepath)}",
        f"runs/singles/*/{filepath}",
        f"runs/searches/*/*/{os.path.basename(filepath)}",
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # Return the most recent match
            return max(matches, key=os.path.getmtime)
    
    # Return original path if nothing found
    return filepath

#─── Environment setup ─────────────────────────────────────────────────────────
def make_env(headless=True, physics_fps=60):
    # Create dnne_cfg to override physics dt and add PhysX optimizations
    dnne_cfg = {
        'physics_dt': 1.0 / physics_fps,
        'sim': {
            'physx': {
                'solver_type': 1,  # TGS solver - more stable and faster
                'num_position_iterations': 4,
                'num_velocity_iterations': 0,
                'contact_offset': 0.02,
                'rest_offset': 0.001,
                'bounce_threshold_velocity': 0.2,
                'max_depenetration_velocity': 100.0,
                'default_buffer_size_multiplier': 2.0,
                'max_gpu_contact_pairs': 1048576,
                'num_subscenes': 4,  # Parallel processing even for 1 env
                'contact_collection': 0
            }
        }
    }
    
    env = isaacgymenvs.make(
        seed=0,
        task="Cartpole",
        num_envs=1,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=headless,        # control visualization
        force_render=False,  # rl_games player handles rendering separately
        dnne_cfg=dnne_cfg
    )
    
    return env

#─── Dreaming Actor-Critic ─────────────────────────────────────────────────────
class DreamingActorCritic:
    def __init__(self, obs_dim, act_dim,
                 lr_a=8e-5, lr_c=2.5e-4,
                 lambda_a=0.88, lambda_c=0.92,
                 noise_std=0.05, device='cuda:0',
                 dream_noise=0.1, refresh_after_sleep=True,
                 grad_clip_norm=50.0):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.grad_clip_norm = grad_clip_norm
        
        # weights - Xavier initialization
        self.W_a = torch.randn(act_dim, obs_dim, device=self.device) * np.sqrt(2.0 / (act_dim + obs_dim))
        self.W_c = torch.randn(1,       obs_dim, device=self.device) * np.sqrt(2.0 / (1 + obs_dim))
        # Log std for exploration (learnable)
        self.log_std = torch.full((act_dim,), np.log(noise_std), device=self.device, requires_grad=False)
        # eligibility traces
        self.e_a = torch.zeros_like(self.W_a)
        self.e_c = torch.zeros_like(self.W_c)
        # hyperparams
        self.lr_a, self.lr_c = lr_a, lr_c
        self.lambda_a, self.lambda_c = lambda_a, lambda_c
        # training mode
        self.training_mode = True
        
        # Sleep pressure tracking
        self.td_error_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        self.recent_returns = deque(maxlen=10)
        
        # Dream statistics
        self.dream_stats = {
            'episodes_since_last_dream': 0,
            'total_dreams': 0,
            'dream_episodes': [],  # Episodes when dreams occurred
            'dream_improvements': [],  # Performance improvements from dreams
            'sleep_pressure_history': [],
            'dream_noise_history': []
        }
        
        # Dream parameters
        self.dream_noise_sigma = dream_noise  # Initial noise for parameter perturbation
        self.dream_noise_decay = 0.995  # Decay rate for noise
        self.min_dream_noise = min(0.01, dream_noise)  # Don't decay below initial if starting low
        self.refresh_after_sleep = refresh_after_sleep  # Whether to reset traces after dreams
        
        # Sleep pressure thresholds
        # These are calibrated for well-performing agents
        self.trace_saturation_threshold = 0.7  # Traces near max capacity
        self.td_variance_threshold = 1.0  # High uncertainty in value estimates  
        self.action_entropy_threshold = 0.2  # Low exploration (normalized entropy)
        self.dream_threshold = 0.8  # Overall threshold for triggering dreams
        
        # Theoretical max norm for eligibility traces
        self.theoretical_max_norm = np.sqrt(obs_dim * act_dim)

    def policy(self, s):
        # Gaussian policy for continuous actions
        mean = torch.tanh(self.W_a @ s)  # Bound mean to [-1, 1]
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample_action(self, mean, std):
        # Sample from Gaussian and apply tanh squashing
        eps = torch.randn_like(mean)
        action = mean + std * eps
        action_clamped = torch.clamp(action, -1.0, 1.0)
        
        # Track actions for entropy calculation
        if self.training_mode:
            self.action_history.append(action_clamped.cpu().numpy())
        
        return action_clamped

    def value(self, s):
        return (self.W_c @ s).squeeze(0)

    def reset_traces(self):
        self.e_a.zero_()
        self.e_c.zero_()

    def update_traces(self, s, a, mean):
        if self.training_mode:
            # Proper stochastic policy gradient for continuous actions
            # For Gaussian policy: ∇ log π(a|s) = (a - μ)/σ² * ∇μ
            std = torch.exp(self.log_std)
            
            # Diagnostic: Print gradient components
            action_diff = a - mean
            variance = std**2
            grad_scale = action_diff / variance
            
            # Print diagnostics every 100 steps
            if hasattr(self, '_trace_update_count'):
                self._trace_update_count += 1
            else:
                self._trace_update_count = 1
                
            if self._trace_update_count % 100 == 0:
                print(f"\n--- Gradient Diagnostics (step {self._trace_update_count}) ---")
                print(f"  σ (std): {std.mean().item():.6f}")
                print(f"  σ² (variance): {variance.mean().item():.6f}")
                print(f"  (a - μ): {action_diff.abs().mean().item():.6f}")
                print(f"  (a - μ)/σ²: {grad_scale.abs().mean().item():.2f}")
                print(f"  Amplification factor: {(1.0/variance.mean()).item():.1f}x")
            
            # Gradient of mean w.r.t weights: ∇μ = (1 - tanh²) * s
            grad_mean = (1 - mean**2).unsqueeze(1) * s.unsqueeze(0)
            
            # Full policy gradient: (a - μ)/σ² * ∇μ
            grad_log = grad_scale.unsqueeze(1) * grad_mean
            
            # Diagnostic: Print gradient norm
            if self._trace_update_count % 100 == 0:
                grad_norm = torch.norm(grad_log)
                print(f"  Policy gradient norm: {grad_norm.item():.2f}")
            
            # Update actor trace
            self.e_a.mul_(self.lambda_a)
            self.e_a.add_(grad_log)

            # critic trace (∇ V = s)
            self.e_c.mul_(self.lambda_c)
            self.e_c += s.unsqueeze(0)

    def apply_updates(self, delta, td_clip=5.0):
        # three-factor updates with gradient clipping
        if self.training_mode:
            delta_clamped = torch.clamp(delta, -td_clip, td_clip)  # Clip TD error
            
            # Apply gradient norm clipping to eligibility traces
            actor_clipped = False
            critic_clipped = False
            
            if self.grad_clip_norm > 0:
                # Clip actor trace
                actor_trace_norm = torch.norm(self.e_a)
                if actor_trace_norm > self.grad_clip_norm:
                    self.e_a = self.e_a * (self.grad_clip_norm / actor_trace_norm)
                    actor_clipped = True
                
                # Clip critic trace
                critic_trace_norm = torch.norm(self.e_c)
                if critic_trace_norm > self.grad_clip_norm:
                    self.e_c = self.e_c * (self.grad_clip_norm / critic_trace_norm)
                    critic_clipped = True
            
            # Diagnostic: Print update magnitudes every 100 steps
            if hasattr(self, '_trace_update_count') and self._trace_update_count % 100 == 0:
                actor_trace_norm = torch.norm(self.e_a)
                critic_trace_norm = torch.norm(self.e_c)
                actor_update_norm = torch.norm(self.lr_a * delta_clamped * self.e_a)
                critic_update_norm = torch.norm(self.lr_c * delta_clamped * self.e_c)
                
                print(f"  TD error (δ): {delta_clamped.item():.4f}")
                print(f"  Actor trace norm: {actor_trace_norm.item():.2f} {'[CLIPPED]' if actor_clipped else ''}")
                print(f"  Critic trace norm: {critic_trace_norm.item():.2f} {'[CLIPPED]' if critic_clipped else ''}")
                print(f"  Actor update norm: {actor_update_norm.item():.6f}")
                print(f"  Critic update norm: {critic_update_norm.item():.6f}")
                print(f"  Actor weights norm: {torch.norm(self.W_a).item():.2f}")
                print("---")
            
            self.W_a +=  self.lr_a * delta_clamped * self.e_a
            self.W_c +=  self.lr_c * delta_clamped * self.e_c
            
            # Track TD error for variance calculation
            self.td_error_history.append(delta.item())
    
    def compute_trace_saturation(self):
        """Compute how saturated the eligibility traces are (0-1 range)."""
        # Compute current trace norm (both actor and critic)
        current_norm = torch.norm(self.e_a).item() + torch.norm(self.e_c).item()
        
        # Keep track of recent trace norms and historical maximum
        if not hasattr(self, 'trace_norm_history'):
            self.trace_norm_history = deque(maxlen=50)
            self.max_trace_norm = 0.0
        
        self.trace_norm_history.append(current_norm)
        self.max_trace_norm = max(self.max_trace_norm, current_norm)
        
        if len(self.trace_norm_history) < 5:
            return 0.0  # Not enough data
        
        # Saturation occurs when:
        # 1. Current norm is close to historical maximum (traces near their peak)
        # 2. Recent growth rate is low (traces have stopped growing)
        
        recent_norms = list(self.trace_norm_history)[-10:]
        
        # Measure proximity to maximum
        if self.max_trace_norm > 0:
            proximity_to_max = current_norm / self.max_trace_norm
        else:
            proximity_to_max = 0.0
        
        # Measure growth rate (how much traces are still growing)
        if len(recent_norms) >= 2:
            growth_rate = (recent_norms[-1] - recent_norms[0]) / (len(recent_norms) - 1)
            # Normalize by current value to get relative growth
            if current_norm > 0:
                relative_growth = abs(growth_rate) / current_norm
                # Low growth = high saturation
                growth_saturation = 1.0 - min(1.0, relative_growth * 10)
            else:
                growth_saturation = 0.0
        else:
            growth_saturation = 0.0
        
        # Combine both indicators
        saturation = 0.6 * proximity_to_max + 0.4 * growth_saturation
        
        return saturation
    
    def compute_td_variance(self):
        """Compute variance of recent TD errors."""
        if len(self.td_error_history) < 2:
            return float('inf')  # Need at least 2 samples for variance
        # Use all available data, even if less than ideal
        return np.var(list(self.td_error_history))
    
    def compute_action_entropy(self):
        """Compute entropy of recent actions."""
        if len(self.action_history) < 3:
            return float('inf')  # Need at least a few samples
        
        # Convert actions to flat array (handle both scalars and arrays)
        action_list = []
        for a in self.action_history:
            if isinstance(a, np.ndarray):
                action_list.append(a.item() if a.size == 1 else a[0])
            else:
                action_list.append(a)
        
        actions = np.array(action_list)
        
        # Use adaptive binning based on data
        n_bins = min(10, max(3, len(actions) // 5))
        bins = np.linspace(-1, 1, n_bins + 1)
        digitized = np.digitize(actions, bins)
        
        # Compute entropy
        unique, counts = np.unique(digitized, return_counts=True)
        probs = counts / counts.sum()
        # Use natural log for consistency, convert to bits if needed
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        # Normalize by max possible entropy for the number of bins
        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def compute_sleep_pressure(self):
        """Compute combined sleep pressure from all metrics.
        
        Sleep pressure indicates when the agent needs dreams to improve learning:
        1. Mental fatigue: Eligibility traces saturated (can't form new associations)
        2. Frustration: High TD variance (unstable value estimates) 
        3. Boredom: Low action entropy (stuck in repetitive behavior)
        
        The system adapts based on performance:
        - Good performance → more tolerant thresholds
        - Poor performance → more sensitive to issues
        - Excellent performance → requires multiple factors to trigger dreams
        """
        trace_sat = self.compute_trace_saturation()
        td_var = self.compute_td_variance()
        action_ent = self.compute_action_entropy()
        
        # Adapt thresholds based on recent performance
        recent_returns = list(self.recent_returns)[-10:] if hasattr(self, 'recent_returns') else []
        if recent_returns and len(recent_returns) >= 3:
            avg_return = np.mean(recent_returns)
            # For CartPole, 500 is perfect, 200 is mediocre, <100 is bad
            performance_factor = min(1.0, avg_return / 300.0)  # 0-1 scale
            
            # When performing well, be more tolerant of low entropy and high variance
            adaptive_entropy_threshold = self.action_entropy_threshold * (0.5 + 0.5 * performance_factor)
            adaptive_variance_threshold = self.td_variance_threshold * (0.5 + 0.5 * performance_factor)
        else:
            # Use default thresholds when insufficient data
            adaptive_entropy_threshold = self.action_entropy_threshold
            adaptive_variance_threshold = self.td_variance_threshold
        
        # Normalize each metric to [0, 1]
        pressure_trace = min(1.0, trace_sat / self.trace_saturation_threshold)
        
        # For TD variance: high variance = low pressure (system is exploring/learning)
        # Use neutral pressure (0.5) when insufficient data
        if td_var < float('inf'):
            pressure_td = 1.0 - min(1.0, td_var / adaptive_variance_threshold)
        else:
            pressure_td = 0.5  # Neutral pressure when insufficient data
        
        # For action entropy: low entropy = high pressure (system is stuck)
        # Use neutral pressure (0.5) when insufficient data
        if action_ent < float('inf'):
            # action_ent is normalized [0,1], low values mean stuck behavior
            if action_ent < adaptive_entropy_threshold:
                # Scale pressure from 1.0 (at 0 entropy) to 0.0 (at threshold)
                pressure_action = 1.0 - (action_ent / adaptive_entropy_threshold)
            else:
                pressure_action = 0.0  # No pressure if entropy is healthy
        else:
            pressure_action = 0.5  # Neutral pressure when insufficient data
        
        # Combined pressure with performance weighting
        # If performing well, require stronger signal to trigger dreams
        if recent_returns and len(recent_returns) >= 3:
            avg_return = np.mean(recent_returns)
            if avg_return > 400:  # Excellent performance
                # Require at least two factors or one very strong factor
                pressures = [pressure_trace, pressure_td, pressure_action]
                high_pressures = [p for p in pressures if p > 0.6]
                if len(high_pressures) >= 2 or any(p > 0.9 for p in pressures):
                    sleep_pressure = max(pressures)
                else:
                    sleep_pressure = max(pressures) * 0.8  # Dampen single moderate signals
            else:
                # Normal combination for moderate performance
                sleep_pressure = max(pressure_trace, pressure_td, pressure_action)
        else:
            # Default combination when insufficient data
            sleep_pressure = max(pressure_trace, pressure_td, pressure_action)
        
        # Store detailed metrics
        self.dream_stats['sleep_pressure_history'].append({
            'episode': self.dream_stats['episodes_since_last_dream'],
            'trace_saturation': trace_sat,
            'td_variance': td_var,
            'action_entropy': action_ent,
            'pressure_trace': pressure_trace,
            'pressure_td': pressure_td,
            'pressure_action': pressure_action,
            'combined_pressure': sleep_pressure
        })
        
        return sleep_pressure, {
            'trace_saturation': trace_sat,
            'td_variance': td_var,
            'action_entropy': action_ent
        }
    
    def should_dream(self, force_interval=None, dream_threshold=None, min_episodes=5):
        """Determine if dreaming should occur."""
        # Ensure minimum gap between dreams for metrics to accumulate
        episodes_since_dream = self.dream_stats['episodes_since_last_dream']
        if episodes_since_dream < min_episodes:
            return False
        
        # For initial testing, can force dreams every N episodes
        if force_interval and episodes_since_dream >= force_interval:
            return True
        
        sleep_pressure, _ = self.compute_sleep_pressure()
        threshold = dream_threshold if dream_threshold is not None else self.dream_threshold
        return sleep_pressure > threshold
    
    def create_dream_policy(self):
        """Create a single dream variation of the policy."""
        # Scale noise by learning rate for adaptive exploration
        # This ensures dream perturbations are proportional to typical update sizes
        actor_noise_scale = self.dream_noise_sigma * np.sqrt(self.lr_a / 8e-5)  # Normalized to default LR
        critic_noise_scale = self.dream_noise_sigma * np.sqrt(self.lr_c / 2.5e-4)
        
        # Debug: Print noise scales on first dream only
        if not hasattr(self, '_printed_noise_debug'):
            print(f"  Actor noise scale: {actor_noise_scale:.6f}")
            print(f"  Critic noise scale: {critic_noise_scale:.6f}")
            print(f"  Actor weight norm: {torch.norm(self.W_a).item():.6f}")
            print(f"  Critic weight norm: {torch.norm(self.W_c).item():.6f}")
            self._printed_noise_debug = True
        
        # Add Gaussian noise to parameters
        W_a_dream = self.W_a + torch.randn_like(self.W_a) * actor_noise_scale
        W_c_dream = self.W_c + torch.randn_like(self.W_c) * critic_noise_scale
        return W_a_dream, W_c_dream
    
    def evaluate_dream_policy(self, env, W_a_dream, W_c_dream, max_steps=200, reward_scale=1.0):
        """Evaluate a single dream policy."""
        obs_dict = env.reset()
        obs = obs_dict['obs']
        state = obs[0].clone().detach().to(device=self.device, dtype=torch.float32)
        
        total_reward = 0
        steps = 0
        done = False
        
        # Debug: Track first few steps
        debug_steps = []
        
        while not done and steps < max_steps:
            # Dream policy
            mean = torch.tanh(W_a_dream @ state)
            std = torch.exp(self.log_std)
            eps = torch.randn_like(mean)
            action = torch.clamp(mean + std * eps, -1.0, 1.0)
            
            # Step
            next_obs_dict, reward, done, _ = env.step(action.unsqueeze(0))
            next_obs = next_obs_dict['obs']
            
            # Simple sum of rewards without gamma discounting
            # Dreams should see the same undiscounted returns as regular evaluation
            total_reward += reward[0].item()
            steps += 1
            
            # Debug: Capture first few steps
            if steps <= 5:
                debug_steps.append({
                    'state': state.cpu().numpy(),
                    'action': action.item(),
                    'reward': reward[0].item()
                })
            
            state = next_obs[0].clone().detach().to(device=self.device, dtype=torch.float32)
        
        # Debug: Print first evaluation details
        if not hasattr(self, '_printed_eval_debug'):
            print(f"\n  First dream evaluation details:")
            for i, step in enumerate(debug_steps):
                print(f"    Step {i+1}: state={step['state'][:2]}, action={step['action']:.3f}, reward={step['reward']:.1f}")
            self._printed_eval_debug = True
        
        return total_reward, steps
    
    def dream_phase_sequential(self, env, num_dreams=8, reward_scale=1.0, current_episode=0):
        """Execute dreaming phase with sequential parameter explorations."""
        print(f"\n💤 Entering dream phase at episode {current_episode} (after {self.dream_stats['episodes_since_last_dream']} episodes)...")
        print(f"  Dream noise sigma: {self.dream_noise_sigma:.6f}")
        print(f"  Learning rates: lr_a={self.lr_a:.2e}, lr_c={self.lr_c:.2e}")
        print(f"  Reward scale: {reward_scale}")
        
        dream_results = []
        
        # Evaluate current policy first (baseline)
        baseline_reward, baseline_steps = self.evaluate_dream_policy(env, self.W_a, self.W_c, 
                                                                      max_steps=200, reward_scale=reward_scale)
        print(f"  Baseline performance: {baseline_reward:.1f} (in {baseline_steps} steps)")
        
        # Run sequential dreams
        for i in range(num_dreams):
            W_a_dream, W_c_dream = self.create_dream_policy()
            
            # Debug: Check if dream weights are identical to base weights
            if self.dream_noise_sigma == 0:
                assert torch.allclose(W_a_dream, self.W_a), "Zero noise should give identical actor weights!"
                assert torch.allclose(W_c_dream, self.W_c), "Zero noise should give identical critic weights!"
            
            reward, steps = self.evaluate_dream_policy(env, W_a_dream, W_c_dream, 
                                                       max_steps=200, reward_scale=reward_scale)
            dream_results.append({
                'W_a_diff': W_a_dream - self.W_a,
                'W_c_diff': W_c_dream - self.W_c,
                'reward': reward,
                'steps': steps,
                'improvement': reward - baseline_reward
            })
            
            if (i + 1) % 4 == 0:
                recent_rewards = [d['reward'] for d in dream_results[-4:]]
                print(f"  Dreams {i-3}-{i}: rewards {[f'{r:.1f}' for r in recent_rewards]}")
        
        # Find successful dreams (better than baseline)
        # Add small threshold to avoid numerical precision issues
        improvement_threshold = 0.01 if self.dream_noise_sigma == 0 else 0
        successful_dreams = [d for d in dream_results if d['improvement'] > improvement_threshold]
        num_successful = len(successful_dreams)
        
        print(f"  Dream results: {num_successful}/{num_dreams} improved over baseline")
        
        # Consolidate successful dreams
        if num_successful > 0 and self.dream_noise_sigma > 0:
            self.consolidate_dreams_sequential(successful_dreams)
        elif self.dream_noise_sigma == 0:
            print("  Skipping consolidation (zero noise - no real parameter variations)")
        
        # Update dream statistics
        self.dream_stats['total_dreams'] += 1
        self.dream_stats['dream_episodes'].append(self.dream_stats['episodes_since_last_dream'])
        self.dream_stats['dream_noise_history'].append(self.dream_noise_sigma)
        
        # Decay dream noise
        self.dream_noise_sigma = max(self.min_dream_noise, 
                                     self.dream_noise_sigma * self.dream_noise_decay)
        
        # Only reset histories if dreams were unsuccessful or if explicitly requested
        # Successful dreams suggest the monitoring is working well
        reset_histories = self.refresh_after_sleep and (num_successful == 0)
        
        if reset_histories:
            print("  Resetting mental state after unsuccessful dreams")
            
            # Always reset eligibility traces after dreams
            self.reset_traces()
            
            # Only clear monitoring histories if dreams were completely unsuccessful
            if hasattr(self, 'trace_norm_history'):
                self.trace_norm_history.clear()
            self.td_error_history.clear()
            self.action_history.clear()
        else:
            # Still reset traces but keep histories
            print("  Resetting traces but keeping monitoring histories")
            self.reset_traces()
        
        return [d['reward'] for d in dream_results]
    
    def consolidate_dreams_sequential(self, successful_dreams):
        """Update base policy toward successful dreams."""
        if not successful_dreams:
            return
        
        # Sort by improvement
        successful_dreams.sort(key=lambda d: d['improvement'], reverse=True)
        
        # Use top-k dreams (e.g., top 3)
        top_k = min(3, len(successful_dreams))
        top_dreams = successful_dreams[:top_k]
        
        # Compute weighted average based on improvement
        improvements = torch.tensor([d['improvement'] for d in top_dreams], device=self.device)
        weights = torch.softmax(improvements, dim=0)
        
        # Update actor
        W_a_update = torch.zeros_like(self.W_a)
        for i, dream in enumerate(top_dreams):
            W_a_update += weights[i] * dream['W_a_diff']
        
        # Update critic similarly
        W_c_update = torch.zeros_like(self.W_c)
        for i, dream in enumerate(top_dreams):
            W_c_update += weights[i] * dream['W_c_diff']
        
        # Apply updates with learning rate
        # Scale consolidation by learning rate to avoid overshooting
        # For best hyperparams (lr_a=1.5e-5), this gives consolidation_rate ≈ 0.0019
        consolidation_rate_actor = 10 * self.lr_a  # Proportional to learning rate
        consolidation_rate_critic = 10 * self.lr_c
        
        self.W_a += consolidation_rate_actor * W_a_update
        self.W_c += consolidation_rate_critic * W_c_update
        
        avg_improvement = np.mean([d['improvement'] for d in top_dreams])
        print(f"  Consolidated top-{top_k} dreams (avg improvement: {avg_improvement:.1f})")
    
    def save_checkpoint(self, filepath, physics_fps=60):
        """Save model weights and hyperparameters to a checkpoint file."""
        checkpoint = {
            'W_a': self.W_a.cpu(),
            'W_c': self.W_c.cpu(),
            'log_std': self.log_std.cpu(),
            'hyperparams': {
                'lr_a': self.lr_a,
                'lr_c': self.lr_c,
                'lambda_a': self.lambda_a,
                'lambda_c': self.lambda_c,
                'obs_dim': self.W_a.shape[1],
                'act_dim': self.W_a.shape[0]
            },
            'physics_fps': physics_fps,
            'dream_stats': self.dream_stats,
            'dream_noise_sigma': self.dream_noise_sigma
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights and hyperparameters from checkpoint directory or file."""
        import os
        import json
        
        # If path is a directory, look for checkpoint file
        if os.path.isdir(checkpoint_path):
            # Look for checkpoint files in the directory
            import glob
            checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*checkpoint*.pth"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
            # Use the best checkpoint if available, otherwise the first one
            checkpoint_file = next((f for f in checkpoint_files if "best" in f), checkpoint_files[0])
            
            # Look for metrics.json to get hyperparameters
            metrics_file = os.path.join(checkpoint_path, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    saved_hyperparams = metrics_data.get('hyperparameters', {})
            else:
                saved_hyperparams = {}
        else:
            # Path is a file
            checkpoint_file = checkpoint_path
            saved_hyperparams = {}
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.W_a = checkpoint['W_a'].to(self.device)
        self.W_c = checkpoint['W_c'].to(self.device)
        self.log_std = checkpoint['log_std'].to(self.device)
        
        # Reset traces when loading
        self.e_a = torch.zeros_like(self.W_a)
        self.e_c = torch.zeros_like(self.W_c)
        # Set training mode to False by default when loading
        self.training_mode = False
        
        # Load dream statistics if available
        if 'dream_stats' in checkpoint:
            self.dream_stats = checkpoint['dream_stats']
        if 'dream_noise_sigma' in checkpoint:
            self.dream_noise_sigma = checkpoint['dream_noise_sigma']
        
        # Get hyperparameters from checkpoint or metrics file
        checkpoint_hyperparams = checkpoint.get('hyperparams', {})
        # Prefer hyperparams from metrics.json if available
        hyperparams = saved_hyperparams if saved_hyperparams else checkpoint_hyperparams
        
        # Update learning rates and other hyperparameters
        if 'lr_actor' in hyperparams:
            self.lr_a = hyperparams['lr_actor']
        elif 'lr_a' in hyperparams:
            self.lr_a = hyperparams['lr_a']
            
        if 'lr_critic' in hyperparams:
            self.lr_c = hyperparams['lr_critic']
        elif 'lr_c' in hyperparams:
            self.lr_c = hyperparams['lr_c']
            
        if 'lambda_actor' in hyperparams:
            self.lambda_a = hyperparams['lambda_actor']
        elif 'lambda_a' in hyperparams:
            self.lambda_a = hyperparams['lambda_a']
            
        if 'lambda_critic' in hyperparams:
            self.lambda_c = hyperparams['lambda_critic']
        elif 'lambda_c' in hyperparams:
            self.lambda_c = hyperparams['lambda_c']
        
        # Get physics_fps if available
        physics_fps = checkpoint.get('physics_fps', 60)
        
        print(f"Checkpoint loaded from {checkpoint_file}")
        print(f"Loaded hyperparameters: lr_a={self.lr_a}, lr_c={self.lr_c}, lambda_a={self.lambda_a}, lambda_c={self.lambda_c}")
        if physics_fps != 60:
            print(f"Physics FPS: {physics_fps}")
        
        return hyperparams, physics_fps

#─── Training loop ─────────────────────────────────────────────────────────────
def train(headless=True, num_episodes=500, args=None):
    # Set up run directory
    run_dir = None
    if args and args.save_checkpoint:
        # If save_checkpoint is a string path, use that as the directory
        if args.save_checkpoint and isinstance(args.save_checkpoint, str):
            run_dir = args.save_checkpoint
        else:
            run_dir = get_run_directory(run_type='singles', run_id=args.run_id if args.run_id != 'default' else None)
        ensure_directory_exists(run_dir)
        print(f"Run directory: {run_dir}")
    
    # Determine physics FPS (will be updated if loading checkpoint)
    physics_fps = 60  # default
    
    if args and hasattr(args, 'physics_fps') and not args.load_checkpoint:
        # Only use command line physics_fps if NOT loading a checkpoint
        physics_fps = args.physics_fps
    
    # Suppress Isaac Gym startup messages
    import io
    import contextlib
    
    if headless:
        print("Creating environment...")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = make_env(headless=headless, physics_fps=physics_fps)
        print("Environment created!")
    else:
        # In visual mode, show some output but suppress the worst warnings
        import sys
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        print("Creating environment...")
        env = make_env(headless=headless, physics_fps=physics_fps)
        print("Environment created!")
        
        # Restore stderr but filter out the captured warnings
        captured_warnings = sys.stderr.getvalue()
        sys.stderr = old_stderr
        
        # Only print non-Isaac Gym warnings
        for line in captured_warnings.split('\n'):
            if line and not any(skip in line for skip in ['ninja:', 'WARNING: dzn', '[Warning]', 'Not connected']):
                print(line, file=sys.stderr)
    
    # Initialize timing variables for real-time rendering
    # Default render FPS to physics FPS unless explicitly specified
    if args and hasattr(args, 'render_fps') and args.render_fps is not None:
        render_fps = args.render_fps
    else:
        render_fps = physics_fps
    render_dt = 1.0 / render_fps
    last_frame_time = time.time()
    frame_count = 0
    fps_start_time = time.time()

    obs_dict = env.reset()
    obs = obs_dict['obs']                     # Extract observation tensor from dict
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    obs_dim = obs.shape[1]
    
    act_dim = env.action_space.shape[0]  # For continuous actions
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    # Use args if provided, otherwise defaults
    if args:
        # Check if we're loading a checkpoint
        if args.load_checkpoint:
            # Create AC with defaults first, but use command line dream_noise
            refresh_after_sleep = (args.refresh_after_sleep == 'true')
            ac = DreamingActorCritic(obs_dim=obs_dim, act_dim=act_dim, device=device,
                                   dream_noise=args.dream_noise,
                                   refresh_after_sleep=refresh_after_sleep,
                                   grad_clip_norm=args.grad_clip_norm)
            # Resolve checkpoint path
            checkpoint_path = resolve_checkpoint_path(args.load_checkpoint)
            if checkpoint_path != args.load_checkpoint:
                print(f"Resolved checkpoint path: {checkpoint_path}")
            # Load checkpoint (hyperparameters are now updated inside load_checkpoint)
            loaded_params, loaded_physics_fps = ac.load_checkpoint(checkpoint_path)
            
            # If load_hparams_only is set, reinitialize the weights
            if args.load_hparams_only:
                print("Reinitializing weights (keeping only hyperparameters from checkpoint)")
                # Reinitialize weights with Xavier initialization
                ac.W_a = torch.randn(act_dim, obs_dim, device=ac.device) * np.sqrt(2.0 / (act_dim + obs_dim))
                ac.W_c = torch.randn(1, obs_dim, device=ac.device) * np.sqrt(2.0 / (1 + obs_dim))
                # Reset eligibility traces
                ac.e_a = torch.zeros_like(ac.W_a)
                ac.e_c = torch.zeros_like(ac.W_c)
                # Reset dream-specific histories
                ac.recent_returns = deque(maxlen=20)
                ac.td_error_history = deque(maxlen=100)
                ac.action_history = deque(maxlen=100)
                ac.trace_norm_history = deque(maxlen=50)
                ac.max_trace_norm = 0.0
                # Reset dream statistics
                ac.dream_stats = {
                    'episodes_since_last_dream': 0,
                    'total_dreams': 0,
                    'dream_episodes': [],  # Episodes when dreams occurred
                    'dream_improvements': [],  # Performance improvements from dreams
                    'sleep_pressure_history': [],
                    'dream_noise_history': []
                }
                # Set training mode to True when loading hparams only
                ac.training_mode = True
            
            # Check physics FPS compatibility
            if loaded_physics_fps != physics_fps:
                print(f"WARNING: Environment physics FPS ({physics_fps}) doesn't match checkpoint ({loaded_physics_fps})")
                if hasattr(args, 'physics_fps') and args.physics_fps != 60:
                    print(f"ERROR: Cannot specify --physics-fps when loading a checkpoint.")
                    print(f"Checkpoint was trained with physics FPS {loaded_physics_fps}.")
                    sys.exit(1)
            
            # Use loaded hyperparams unless overridden
            gamma = args.gamma
            reward_scale = args.reward_scale
            td_clip = args.td_clip
        else:
            refresh_after_sleep = (args.refresh_after_sleep == 'true')
            ac = DreamingActorCritic(obs_dim=obs_dim, act_dim=act_dim, device=device,
                            lr_a=args.lr_actor, lr_c=args.lr_critic,
                            lambda_a=args.lambda_actor, lambda_c=args.lambda_critic,
                            noise_std=args.noise_std,
                            dream_noise=args.dream_noise,
                            refresh_after_sleep=refresh_after_sleep,
                            grad_clip_norm=args.grad_clip_norm)
            gamma = args.gamma
            reward_scale = args.reward_scale
            td_clip = args.td_clip
        
        # Handle training mode override
        if args.training_mode is not None:
            ac.training_mode = (args.training_mode == 'true')
            print(f"Training mode set to: {ac.training_mode}")
        
        # Set dream parameters if provided
        if hasattr(args, 'force_dream_interval'):
            force_dream_interval = args.force_dream_interval
        else:
            force_dream_interval = None
        
        # Set dream threshold
        dream_threshold = args.dream_threshold if hasattr(args, 'dream_threshold') else 0.8
            
    else:
        ac = DreamingActorCritic(obs_dim=obs_dim, act_dim=act_dim, device=device)
        gamma = 0.99
        reward_scale = 5.0
        td_clip = 5.0
        force_dream_interval = None
        dream_threshold = 0.8
    
    mode_str = "training" if ac.training_mode else "evaluation"
    print(f"Starting {mode_str} for {num_episodes} episodes...")
    if args and args.disable_dreams:
        print("Dreams are DISABLED")
    elif force_dream_interval:
        print(f"Forcing dreams every {force_dream_interval} episodes")
    else:
        print(f"Dreams will be triggered by sleep pressure (threshold: {dream_threshold})")
    
    # Track running average
    returns = []
    window_size = 20
    best_avg_return = -float('inf')

    try:
        for ep in range(1, num_episodes+1):
            # Reset environment at the start of each episode
            obs_dict = env.reset()
            obs = obs_dict['obs']
            
            ac.reset_traces()
            state = obs[0].clone().detach().to(device=ac.device, dtype=torch.float32)
            done = False
            total_r = 0
            
            while not done:
                # select action (continuous)
                mean, std = ac.policy(state)
                action = ac.sample_action(mean, std)
                
                # step - action needs to be reshaped for the environment
                next_obs_dict, reward, done, _ = env.step(action.unsqueeze(0))
                next_obs = next_obs_dict['obs']
                if not headless:
                    env.render()  # Need to call render explicitly since force_render=False
                    frame_count += 1
                    # Small sleep to let rendering catch up (like rl_games player)
                    time.sleep(0.002)
                total_r += reward[0].item() if torch.is_tensor(reward[0]) else reward[0]
                
                # compute TD error with scaled reward
                scaled_reward = reward[0] / reward_scale
                next_s = next_obs[0].clone().detach().to(device=ac.device, dtype=torch.float32)
                v    = ac.value(state)
                v_p  = ac.value(next_s) * (1 - int(done))
                delta = scaled_reward + gamma * v_p - v
                
                # traces & updates with proper stochastic policy gradient
                ac.update_traces(state, action, mean)
                ac.apply_updates(delta, td_clip)
                
                state = next_s
            
            # Track returns
            total_r_scalar = total_r.item() if torch.is_tensor(total_r) else total_r
            returns.append(total_r_scalar)
            ac.recent_returns.append(total_r_scalar)
            avg_return = np.mean(returns[-window_size:]) if len(returns) >= window_size else np.mean(returns)
            
            # Update episode counter
            ac.dream_stats['episodes_since_last_dream'] += 1
            
            # Check if we should dream
            if ac.training_mode and not args.disable_dreams and ac.should_dream(force_dream_interval, dream_threshold, min_episodes=args.min_episodes_between_dreams):
                # Store performance before dreaming
                pre_dream_avg = avg_return
                
                # Execute dream phase
                dream_rewards = ac.dream_phase_sequential(
                    env, 
                    num_dreams=8,
                    reward_scale=reward_scale,
                    current_episode=ep
                )
                
                # Reset episode counter
                ac.dream_stats['episodes_since_last_dream'] = 0
                
                # We'll check improvement in next episode
                if len(returns) > 0:
                    ac.dream_stats['dream_improvements'].append({
                        'episode': ep,
                        'pre_dream_avg': pre_dream_avg,
                        'dream_rewards': dream_rewards
                    })
            
            # Print episode results
            if ep % 5 == 0 or (ac.training_mode and ac.dream_stats['episodes_since_last_dream'] <= 1):
                print(f"Episode {ep:3d}\tReturn {total_r_scalar:6.1f}\tAvg Return {avg_return:6.1f}")
                
                # Print sleep pressure if in training mode
                if ac.training_mode:
                    pressure, metrics = ac.compute_sleep_pressure()
                    print(f"  Sleep pressure: {pressure:.2f} (fatigue: {metrics['trace_saturation']:.2f}, "
                          f"frustration: {metrics['td_variance']:.3f}, boredom: {metrics['action_entropy']:.2f})")
            
            # Auto-save best model during training
            if ac.training_mode and len(returns) >= window_size:
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    if args and args.save_checkpoint and run_dir:
                        # Save with _best suffix in run directory
                        best_checkpoint_path = os.path.join(run_dir, "checkpoint_best.pth")
                        ac.save_checkpoint(best_checkpoint_path, physics_fps=physics_fps)
                        print(f"New best average return: {best_avg_return:.2f}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving progress...")
    
    finally:
        # Save checkpoint and metrics if requested
        if args and args.save_checkpoint and run_dir:
            checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
            ac.save_checkpoint(checkpoint_path, physics_fps=physics_fps)
            
            # Always save metrics with checkpoint
            # Use actual hyperparameters from the AC object (in case they were loaded from checkpoint)
            metrics = {
                'run_id': args.run_id,
                'hyperparameters': {
                    'lr_actor': ac.lr_a,
                    'lr_critic': ac.lr_c,
                    'lambda_actor': ac.lambda_a,
                    'lambda_critic': ac.lambda_c,
                    'noise_std': torch.exp(ac.log_std).item(),
                    'gamma': gamma,
                    'reward_scale': reward_scale,
                    'td_clip': td_clip
                },
                'returns': returns,
                'final_avg_return': np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns),
                'dream_stats': ac.dream_stats
            }
            filename = os.path.join(run_dir, "metrics.json")
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {filename}")
        
        # Print FPS statistics
        if not headless and frame_count > 0:
            total_time = time.time() - fps_start_time
            avg_fps = frame_count / total_time
            print(f"\nRendering Statistics:")
            print(f"  Total frames: {frame_count}")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Target FPS: {render_fps}")
            print(f"  Physics FPS: {physics_fps}")
        
        # Print dream statistics
        if ac.dream_stats['total_dreams'] > 0:
            print(f"\nDream Statistics:")
            print(f"  Total dreams: {ac.dream_stats['total_dreams']}")
            print(f"  Dream episodes: {ac.dream_stats['dream_episodes']}")
            print(f"  Average episodes between dreams: {np.mean(ac.dream_stats['dream_episodes']):.1f}")
        
        # Clean up environment (Isaac Gym environments don't have close method)
        pass
    
    return np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CartPole with Dreaming Actor-Critic')
    parser.add_argument('--visual', action='store_true', 
                        help='Show visualization window (default: headless)')
    parser.add_argument('--num-episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    # Learning rates
    parser.add_argument('--lr-actor', type=float, default=8e-5,
                        help='Actor learning rate (default: 8e-5)')
    parser.add_argument('--lr-critic', type=float, default=2.5e-4,
                        help='Critic learning rate (default: 2.5e-4)')
    # Trace decay
    parser.add_argument('--lambda-actor', type=float, default=0.88,
                        help='Actor eligibility trace decay (default: 0.88)')
    parser.add_argument('--lambda-critic', type=float, default=0.92,
                        help='Critic eligibility trace decay (default: 0.92)')
    # Exploration
    parser.add_argument('--noise-std', type=float, default=0.05,
                        help='Action noise standard deviation (default: 0.05)')
    # Other hyperparameters
    parser.add_argument('--gamma', type=float, default=0.96,
                        help='Discount factor (default: 0.96)')
    parser.add_argument('--reward-scale', type=float, default=10.0,
                        help='Reward scaling divisor (default: 10.0)')
    parser.add_argument('--td-clip', type=float, default=5.0,
                        help='TD error clipping value (default: 5.0)')
    parser.add_argument('--grad-clip-norm', type=float, default=50.0,
                        help='Gradient norm clipping threshold, 0 to disable (default: 50.0)')
    # Output
    parser.add_argument('--run-id', type=str, default='default',
                        help='Run identifier for output directory')
    parser.add_argument('--load-hparams-only', action='store_true',
                        help='Load only hyperparameters from checkpoint, not weights (use with --load-checkpoint)')
    # Checkpoint arguments
    parser.add_argument('--save-checkpoint', nargs='?', const=True, default=None,
                        help='Save checkpoint after training. Optional: specify directory path (default: auto-generated in runs/singles/)')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to load checkpoint from')
    parser.add_argument('--training-mode', type=str, choices=['true', 'false'], default=None,
                        help='Override training mode (true/false)')
    parser.add_argument('--render-fps', type=int, default=None,
                        help='Rendering FPS for visualization (default: same as physics FPS)')
    parser.add_argument('--physics-fps', type=int, default=60,
                        help='Physics simulation FPS (default: 60)')
    # Dream-specific arguments
    parser.add_argument('--force-dream-interval', type=int, default=None,
                        help='Force dreams every N episodes (default: adaptive based on sleep pressure)')
    parser.add_argument('--dream-threshold', type=float, default=0.8,
                        help='Sleep pressure threshold for triggering dreams (default: 0.8)')
    parser.add_argument('--dream-noise', type=float, default=0.1,
                        help='Noise magnitude for dream parameter variations (default: 0.1)')
    parser.add_argument('--refresh-after-sleep', type=str, choices=['true', 'false'], default='true',
                        help='Reset traces/history after dreams (default: true)')
    parser.add_argument('--disable-dreams', action='store_true',
                        help='Disable dreaming entirely for testing')
    parser.add_argument('--min-episodes-between-dreams', type=int, default=5,
                        help='Minimum episodes between dreams to allow metrics to accumulate (default: 5)')
    args = parser.parse_args()
    
    train(headless=not args.visual, num_episodes=args.num_episodes, args=args)