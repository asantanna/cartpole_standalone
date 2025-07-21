#!/usr/bin/env python3
import time
import numpy as np
import argparse
import isaacgym
import isaacgymenvs
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from datetime import datetime

#─── Directory helpers ─────────────────────────────────────────────────────────
def get_run_directory(run_type='singles', run_id=None):
    """Generate run directory path."""
    if run_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"train_{timestamp}"
    
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
def make_env(headless=True):
    return isaacgymenvs.make(
        seed=0,
        task="Cartpole",
        num_envs=1,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=headless,        # control visualization
    )

#─── Actor-Critic with traces ─────────────────────────────────────────────────
class ActorCritic:
    def __init__(self, obs_dim, act_dim,
                 lr_a=8e-5, lr_c=2.5e-4,
                 lambda_a=0.88, lambda_c=0.92,
                 noise_std=0.05, device='cuda:0'):
        self.device = torch.device(device)
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

    def policy(self, s):
        # Gaussian policy for continuous actions
        mean = torch.tanh(self.W_a @ s)  # Bound mean to [-1, 1]
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample_action(self, mean, std):
        # Sample from Gaussian and apply tanh squashing
        eps = torch.randn_like(mean)
        action = mean + std * eps
        return torch.clamp(action, -1.0, 1.0)

    def value(self, s):
        return (self.W_c @ s).squeeze(0)

    def reset_traces(self):
        self.e_a.zero_(); self.e_c.zero_()

    def update_traces(self, s, a):
        if self.training_mode:
            # For continuous actions, gradient is simpler
            # actor trace - gradient of tanh(W_a @ s) w.r.t W_a
            grad_log = (1 - a**2).unsqueeze(1) * s.unsqueeze(0)
            self.e_a.mul_(self.lambda_a)
            self.e_a.add_(grad_log)

            # critic trace (∇ V = s)
            self.e_c.mul_(self.lambda_c)
            self.e_c += s.unsqueeze(0)

    def apply_updates(self, delta, td_clip=5.0):
        # three-factor updates with gradient clipping
        if self.training_mode:
            delta = torch.clamp(delta, -td_clip, td_clip)  # Clip TD error
            self.W_a +=  self.lr_a * delta * self.e_a
            self.W_c +=  self.lr_c * delta * self.e_c
    
    def save_checkpoint(self, filepath):
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
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model weights from a checkpoint file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.W_a = checkpoint['W_a'].to(self.device)
        self.W_c = checkpoint['W_c'].to(self.device)
        self.log_std = checkpoint['log_std'].to(self.device)
        # Reset traces when loading
        self.e_a = torch.zeros_like(self.W_a)
        self.e_c = torch.zeros_like(self.W_c)
        # Set training mode to False by default when loading
        self.training_mode = False
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint['hyperparams']

#─── Training loop ─────────────────────────────────────────────────────────────
def train(headless=True, num_episodes=500, args=None):
    # Set up run directory
    run_dir = None
    if args and (args.save_checkpoint or args.save_metrics):
        if hasattr(args, 'out_dir') and args.out_dir:
            # Use custom output directory
            run_id = args.run_id if args.run_id != 'default' else None
            if run_id is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                run_id = f"train_{timestamp}"
            run_dir = os.path.join(args.out_dir, run_id)
        else:
            # Default behavior
            run_dir = get_run_directory(run_type='singles', run_id=args.run_id if args.run_id != 'default' else None)
        ensure_directory_exists(run_dir)
        print(f"Run directory: {run_dir}")
    
    print("Creating environment...")
    env = make_env(headless=headless)
    print("Environment created!")

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
            # Create AC with defaults first
            ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, device=device)
            # Resolve checkpoint path
            checkpoint_path = resolve_checkpoint_path(args.load_checkpoint)
            if checkpoint_path != args.load_checkpoint:
                print(f"Resolved checkpoint path: {checkpoint_path}")
            # Load checkpoint
            loaded_params = ac.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint with params: {loaded_params}")
            # Use loaded hyperparams unless overridden
            gamma = args.gamma
            reward_scale = args.reward_scale
            td_clip = args.td_clip
        else:
            ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, device=device,
                            lr_a=args.lr_actor, lr_c=args.lr_critic,
                            lambda_a=args.lambda_actor, lambda_c=args.lambda_critic,
                            noise_std=args.noise_std)
            gamma = args.gamma
            reward_scale = args.reward_scale
            td_clip = args.td_clip
        
        # Handle training mode override
        if args.training_mode is not None:
            ac.training_mode = (args.training_mode == 'true')
            print(f"Training mode set to: {ac.training_mode}")
    else:
        ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, device=device)
        gamma = 0.99
        reward_scale = 5.0
        td_clip = 5.0
    
    mode_str = "training" if ac.training_mode else "evaluation"
    print(f"Starting {mode_str} for {num_episodes} episodes...")
    
    # Track running average
    returns = []
    window_size = 20
    best_avg_return = -float('inf')

    for ep in range(1, num_episodes+1):
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
                env.render()              # update viewer
                time.sleep(1/60)
            total_r += reward[0].item() if torch.is_tensor(reward[0]) else reward[0]
            # compute TD error with scaled reward
            scaled_reward = reward[0] / reward_scale
            next_s = next_obs[0].clone().detach().to(device=ac.device, dtype=torch.float32)
            v    = ac.value(state)
            v_p  = ac.value(next_s) * (1 - int(done))
            delta = scaled_reward + gamma * v_p - v
            # traces & updates (use mean for gradient)
            ac.update_traces(state, mean)
            ac.apply_updates(delta, td_clip)
            state = next_s

        obs_dict = env.reset()
        obs = obs_dict['obs']
        
        # Track returns
        total_r_scalar = total_r.item() if torch.is_tensor(total_r) else total_r
        returns.append(total_r_scalar)
        avg_return = np.mean(returns[-window_size:]) if len(returns) >= window_size else np.mean(returns)
        
        if ep % 10 == 0:
            print(f"Episode {ep:3d}\tReturn {total_r_scalar:6.1f}\tAvg Return {avg_return:6.1f}")
        
        # Auto-save best model during training
        if ac.training_mode and len(returns) >= window_size:
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                if args and args.save_checkpoint and run_dir:
                    # Save with _best suffix in run directory
                    checkpoint_name = os.path.basename(args.save_checkpoint)
                    best_path = checkpoint_name.rsplit('.', 1)
                    if len(best_path) == 2:
                        best_checkpoint_name = f"{best_path[0]}_best.{best_path[1]}"
                    else:
                        best_checkpoint_name = f"{checkpoint_name}_best"
                    best_checkpoint_path = os.path.join(run_dir, best_checkpoint_name)
                    ac.save_checkpoint(best_checkpoint_path)
                    print(f"New best average return: {best_avg_return:.2f}")
    
    # Save checkpoint if requested
    if args and args.save_checkpoint and run_dir:
        checkpoint_path = os.path.join(run_dir, os.path.basename(args.save_checkpoint))
        ac.save_checkpoint(checkpoint_path)
    
    # Save metrics if requested
    if args and args.save_metrics and run_dir:
        import json
        metrics = {
            'run_id': args.run_id,
            'hyperparameters': {
                'lr_actor': args.lr_actor,
                'lr_critic': args.lr_critic,
                'lambda_actor': args.lambda_actor,
                'lambda_critic': args.lambda_critic,
                'noise_std': args.noise_std,
                'gamma': args.gamma,
                'reward_scale': args.reward_scale,
                'td_clip': args.td_clip
            },
            'returns': returns,
            'final_avg_return': np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
        }
        filename = os.path.join(run_dir, "metrics.json")
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filename}")
    
    return np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CartPole with Actor-Critic')
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
    # Output
    parser.add_argument('--save-metrics', action='store_true',
                        help='Save training metrics to file')
    parser.add_argument('--run-id', type=str, default='default',
                        help='Run identifier for saving metrics')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory for saving results (default: runs/singles)')
    parser.add_argument('--best-config', action='store_true',
                        help='Use the best hyperparameters from search')
    # Checkpoint arguments
    parser.add_argument('--save-checkpoint', type=str, default=None,
                        help='Path to save checkpoint after training')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to load checkpoint from')
    parser.add_argument('--training-mode', type=str, choices=['true', 'false'], default=None,
                        help='Override training mode (true/false)')
    args = parser.parse_args()
    
    # Override with best config if requested
    if args.best_config:
        args.lr_actor = 7.93676080244564e-05
        args.lr_critic = 0.00023467499600008876
        args.lambda_actor = 0.8756946618682102
        args.lambda_critic = 0.9156406148267633
        args.noise_std = 0.02340585371545556
        args.gamma = 0.9632491659519583
        args.reward_scale = 12.941425759749812
        args.td_clip = 5.381298599089067
        print("Using best configuration from hyperparameter search")
    
    train(headless=not args.visual, num_episodes=args.num_episodes, args=args)