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
import sys
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
            'physics_fps': physics_fps
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
    
    # Determine physics FPS (will be updated if loading checkpoint)
    physics_fps = 60  # default
    
    if args and hasattr(args, 'physics_fps') and not args.load_checkpoint:
        # Only use command line physics_fps if NOT loading a checkpoint
        physics_fps = args.physics_fps
    
    print("Creating environment...")
    env = make_env(headless=headless, physics_fps=physics_fps)
    print("Environment created!")
    
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
    
    # Debug timing variables
    debug_timing = args and hasattr(args, 'debug_loop_timing') and args.debug_loop_timing
    if debug_timing:
        timing_stats = {
            'policy': [],
            'step': [],
            'td_compute': [],
            'update': [],
            'reset': [],
            'total_loop': []
        }

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
            # Load checkpoint (hyperparameters are now updated inside load_checkpoint)
            loaded_params, loaded_physics_fps = ac.load_checkpoint(checkpoint_path)
            
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

    try:
        for ep in range(1, num_episodes+1):
            ac.reset_traces()
            state = obs[0].clone().detach().to(device=ac.device, dtype=torch.float32)
            done = False
            total_r = 0
            while not done:
                if debug_timing:
                    loop_start = time.time()
                
                # select action (continuous)
                if debug_timing:
                    t0 = time.time()
                mean, std = ac.policy(state)
                action = ac.sample_action(mean, std)
                if debug_timing:
                    timing_stats['policy'].append(time.time() - t0)
                
                # step - action needs to be reshaped for the environment
                if debug_timing:
                    t0 = time.time()
                next_obs_dict, reward, done, _ = env.step(action.unsqueeze(0))
                next_obs = next_obs_dict['obs']
                if debug_timing:
                    timing_stats['step'].append(time.time() - t0)
                if not headless:
                    env.render()  # Need to call render explicitly since force_render=False
                    frame_count += 1
                    # Small sleep to let rendering catch up (like rl_games player)
                    time.sleep(0.002)
                total_r += reward[0].item() if torch.is_tensor(reward[0]) else reward[0]
                
                # compute TD error with scaled reward
                if debug_timing:
                    t0 = time.time()
                scaled_reward = reward[0] / reward_scale
                next_s = next_obs[0].clone().detach().to(device=ac.device, dtype=torch.float32)
                v    = ac.value(state)
                v_p  = ac.value(next_s) * (1 - int(done))
                delta = scaled_reward + gamma * v_p - v
                if debug_timing:
                    timing_stats['td_compute'].append(time.time() - t0)
                
                # traces & updates (use mean for gradient)
                if debug_timing:
                    t0 = time.time()
                ac.update_traces(state, mean)
                ac.apply_updates(delta, td_clip)
                if debug_timing:
                    timing_stats['update'].append(time.time() - t0)
                
                state = next_s
                
                if debug_timing:
                    timing_stats['total_loop'].append(time.time() - loop_start)

            if debug_timing:
                t0 = time.time()
            obs_dict = env.reset()
            obs = obs_dict['obs']
            if debug_timing:
                timing_stats['reset'].append(time.time() - t0)
            
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
                        best_checkpoint_path = os.path.join(run_dir, "checkpoint_best.pth")
                        ac.save_checkpoint(best_checkpoint_path, physics_fps=physics_fps)
                        print(f"New best average return: {best_avg_return:.2f}")
        
        # Save checkpoint if requested
        if args and args.save_checkpoint and run_dir:
            checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
            ac.save_checkpoint(checkpoint_path, physics_fps=physics_fps)
        
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
    
    finally:
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
        
        # Print debug timing statistics
        if debug_timing and timing_stats['total_loop']:
            print(f"\nLoop Timing Statistics (milliseconds):")
            print(f"{'Component':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Total':>10}")
            print("-" * 65)
            
            for component, times in timing_stats.items():
                if times:
                    times_ms = np.array(times) * 1000  # Convert to milliseconds
                    print(f"{component:<15} {np.mean(times_ms):>8.2f} {np.std(times_ms):>8.2f} "
                          f"{np.min(times_ms):>8.2f} {np.max(times_ms):>8.2f} {np.sum(times_ms):>10.2f}")
            
            # Print percentage breakdown
            total_loop_time = sum(timing_stats['total_loop'])
            print(f"\nPercentage breakdown of total loop time:")
            for component in ['policy', 'step', 'td_compute', 'update']:
                if timing_stats[component]:
                    pct = (sum(timing_stats[component]) / total_loop_time) * 100
                    print(f"  {component:<12}: {pct:>5.1f}%")
    
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
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='Save checkpoint after training to run directory')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to load checkpoint from')
    parser.add_argument('--training-mode', type=str, choices=['true', 'false'], default=None,
                        help='Override training mode (true/false)')
    parser.add_argument('--render-fps', type=int, default=None,
                        help='Rendering FPS for visualization (default: same as physics FPS)')
    parser.add_argument('--physics-fps', type=int, default=60,
                        help='Physics simulation FPS (default: 60)')
    parser.add_argument('--debug-loop-timing', action='store_true',
                        help='Show detailed timing information for loop components')
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