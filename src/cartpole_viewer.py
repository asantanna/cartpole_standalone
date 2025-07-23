#!/usr/bin/env python3
"""
Vectorized viewer for evaluating trained CartPole policies.
This is evaluation-only - no training or weight updates.
"""
import time
import numpy as np
import argparse
import isaacgym
import isaacgymenvs
import torch
import os
import sys

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
def make_env(headless=True, num_envs=1, physics_fps=60):
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
        num_envs=num_envs,
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
    def __init__(self, obs_dim, act_dim, num_envs=1,
                 lr_a=8e-5, lr_c=2.5e-4,
                 lambda_a=0.88, lambda_c=0.92,
                 noise_std=0.05, device='cuda:0'):
        self.device = torch.device(device)
        self.num_envs = num_envs
        # weights - Xavier initialization
        self.W_a = torch.randn(act_dim, obs_dim, device=self.device) * np.sqrt(2.0 / (act_dim + obs_dim))
        self.W_c = torch.randn(1,       obs_dim, device=self.device) * np.sqrt(2.0 / (1 + obs_dim))
        # Log std for exploration (learnable)
        self.log_std = torch.full((act_dim,), np.log(noise_std), device=self.device, requires_grad=False)
        # Always in evaluation mode
        self.training_mode = False

    def policy(self, s):
        # Gaussian policy for continuous actions
        # s shape: (num_envs, obs_dim)
        # W_a shape: (act_dim, obs_dim)
        # mean shape: (num_envs, act_dim)
        mean = torch.tanh(s @ self.W_a.T)  # Bound mean to [-1, 1]
        std = torch.exp(self.log_std).expand(self.num_envs, -1)
        return mean, std
    
    def sample_action(self, mean, std):
        # Sample from Gaussian and apply tanh squashing
        # mean, std shape: (num_envs, act_dim)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        return torch.clamp(action, -1.0, 1.0)

    def value(self, s):
        # s shape: (num_envs, obs_dim)
        # W_c shape: (1, obs_dim)
        # return shape: (num_envs,)
        return (s @ self.W_c.T).squeeze(1)

    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from a checkpoint file or directory."""
        import glob
        
        # Handle directory paths
        if os.path.isdir(checkpoint_path):
            checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*checkpoint*.pth"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
            checkpoint_file = next((f for f in checkpoint_files if "best" in f), checkpoint_files[0])
        else:
            checkpoint_file = checkpoint_path
        
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.W_a = checkpoint['W_a'].to(self.device)
        self.W_c = checkpoint['W_c'].to(self.device)
        self.log_std = checkpoint['log_std'].to(self.device)
        
        # Get physics_fps if available
        physics_fps = checkpoint.get('physics_fps', 60)
        
        print(f"Checkpoint loaded from {checkpoint_file}")
        if physics_fps != 60:
            print(f"Physics FPS: {physics_fps}")
            
        return checkpoint.get('hyperparams', {}), physics_fps

#─── Evaluation loop ─────────────────────────────────────────────────────────────
def evaluate(headless=False, num_envs=1, num_steps=None, args=None):
    if not args or not args.load_checkpoint:
        print("ERROR: --load-checkpoint is required for the viewer")
        sys.exit(1)
    
    # First load checkpoint to get physics FPS
    checkpoint_path = resolve_checkpoint_path(args.load_checkpoint)
    temp_ac = ActorCritic(obs_dim=4, act_dim=1, num_envs=1)
    _, physics_fps = temp_ac.load_checkpoint(checkpoint_path)
    
    print(f"Creating environment with {num_envs} parallel environments...")
    import sys
    sys.stdout.flush()
    env = make_env(headless=headless, num_envs=num_envs, physics_fps=physics_fps)
    print("Environment created!")
    sys.stdout.flush()
    
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
    obs = obs_dict['obs']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    obs_dim = obs.shape[1]
    act_dim = env.action_space.shape[0]
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    
    # Create and load actor-critic
    ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, num_envs=num_envs, device=device)
    ac.load_checkpoint(checkpoint_path)
    
    print(f"Starting evaluation with {num_envs} parallel environments...")
    if num_steps is None:
        print("Running indefinitely (press Ctrl-C to stop)")
    else:
        print(f"Running for {num_steps} steps")
    
    # Statistics tracking
    episode_returns = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int32)
    completed_returns = []
    total_steps = 0
    
    # Initialize states
    states = obs.clone().detach().to(device=ac.device, dtype=torch.float32)
    
    try:
        while num_steps is None or total_steps < num_steps:
            # Select actions for all environments
            with torch.no_grad():
                means, stds = ac.policy(states)
                actions = ac.sample_action(means, stds)
            
            # Step all environments
            next_obs_dict, rewards, dones, _ = env.step(actions)
            next_obs = next_obs_dict['obs']
            
            if not headless:
                env.render()
                frame_count += 1
                
                # Control timing for smooth visualization
                now = time.time()
                elapsed = now - last_frame_time
                if elapsed < render_dt:
                    time.sleep(render_dt - elapsed)
                last_frame_time = time.time()
            
            # Update statistics
            episode_returns += rewards.squeeze()
            episode_lengths += 1
            total_steps += 1
            
            # Update states
            states = next_obs.clone().detach().to(device=ac.device, dtype=torch.float32)
            
            # Handle episode completions
            done_indices = torch.where(dones)[0]
            if len(done_indices) > 0:
                for idx in done_indices:
                    return_value = episode_returns[idx].item()
                    length_value = episode_lengths[idx].item()
                    completed_returns.append(return_value)
                    
                    # Reset this environment's statistics
                    episode_returns[idx] = 0
                    episode_lengths[idx] = 0
                
                # Print statistics periodically
                if len(completed_returns) % 10 == 0:
                    recent_returns = completed_returns[-20:] if len(completed_returns) >= 20 else completed_returns
                    avg_return = np.mean(recent_returns)
                    avg_length = np.mean([l for l in episode_lengths if l > 0])
                    print(f"Episodes: {len(completed_returns):4d} | Avg Return: {avg_return:6.1f} | Active envs avg length: {avg_length:5.1f}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    
    finally:
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"  Total steps: {total_steps}")
        print(f"  Total episodes completed: {len(completed_returns)}")
        if completed_returns:
            print(f"  Average return: {np.mean(completed_returns):.2f}")
            print(f"  Best return: {max(completed_returns):.2f}")
            print(f"  Worst return: {min(completed_returns):.2f}")
        
        # Print FPS statistics
        if not headless and frame_count > 0:
            total_time = time.time() - fps_start_time
            avg_fps = frame_count / total_time
            print(f"\nRendering Statistics:")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Target FPS: {render_fps}")
            print(f"  Physics FPS: {physics_fps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vectorized CartPole Policy Viewer')
    parser.add_argument('--load-checkpoint', type=str, required=True,
                        help='Path to checkpoint file or directory (required)')
    parser.add_argument('--num-envs', type=int, default=16,
                        help='Number of parallel environments (default: 16)')
    parser.add_argument('--num-steps', type=int, default=None,
                        help='Number of steps to run (default: infinite)')
    parser.add_argument('--visual', action='store_true', default=True,
                        help='Show visualization window (default: True)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without visualization')
    parser.add_argument('--render-fps', type=int, default=None,
                        help='Rendering FPS for visualization (default: same as physics FPS)')
    args = parser.parse_args()
    
    # Run evaluation
    evaluate(headless=args.headless, num_envs=args.num_envs, num_steps=args.num_steps, args=args)