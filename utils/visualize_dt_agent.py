import os
import argparse
import torch
import numpy as np
import gymnasium as gym
import time
import sys
import torch.nn as nn

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from memory_dt import MemoryDecisionTransformer
from pomdp_envs.velocity_cartpole import VelocityCartPoleEnv
from pomdp_envs.flickering_pendulum import FlickeringPendulumEnv
from pomdp_envs.lidar_mountain_car import LiDARMountainCarEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='velocity_cartpole',
                        choices=['velocity_cartpole', 'flickering_pendulum', 'lidar_mountain_car'],
                        help='POMDP environment to use')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained Decision Transformer model')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    parser.add_argument('--context_length', type=int, default=20,
                        help='Context length for the transformer')
    parser.add_argument('--memory_type', type=str, default='gru', choices=['gru', 'lstm', 'none'],
                        help='Type of memory used in the model')
    parser.add_argument('--memory_dim', type=int, default=64,
                        help='Hidden dimension of memory state')
    parser.add_argument('--n_layer', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_embed', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--target_return', type=float, default=None,
                        help='Target return value to aim for (default: environment specific)')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Delay between steps for better visualization')
    return parser.parse_args()


def create_env(env_name, render_mode='human'):
    """Create environment with the specified render mode."""
    if env_name == 'velocity_cartpole':
        # Create the base environment with rendering
        base_env = gym.make("CartPole-v1", render_mode=render_mode)
        # Wrap it with our VelocityCartPoleEnv but keep the render functionality
        env = VelocityCartPoleEnv.__new__(VelocityCartPoleEnv)
        VelocityCartPoleEnv.__init__(env)
        env.env = base_env
        return env
    elif env_name == 'flickering_pendulum':
        # Create the base environment with rendering
        base_env = gym.make("Pendulum-v1", render_mode=render_mode)
        # Wrap it with our FlickeringPendulumEnv
        env = FlickeringPendulumEnv.__new__(FlickeringPendulumEnv)
        FlickeringPendulumEnv.__init__(env, flicker_probability=0.3)
        env.env = base_env
        return env
    elif env_name == 'lidar_mountain_car':
        # Create the base environment with rendering
        base_env = gym.make("MountainCar-v0", render_mode=render_mode)
        # Wrap it with our LiDARMountainCarEnv
        env = LiDARMountainCarEnv.__new__(LiDARMountainCarEnv)
        LiDARMountainCarEnv.__init__(env, num_sensors=8)
        env.env = base_env
        return env
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def load_model(model_path, env, args):
    """Load a trained Decision Transformer model."""
    # determine observation dimension
    if isinstance(env.observation_space, gym.spaces.Box):
        state_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, gym.spaces.Dict):
        if 'observation' in env.observation_space.spaces:
            state_dim = env.observation_space.spaces['observation'].shape[0]
            if 'mask' in env.observation_space.spaces:
                state_dim += 1
        else:
            state_dim = sum(space.shape[0] if hasattr(space, 'shape') else 1
                          for space in env.observation_space.spaces.values())
    else:
        raise ValueError(f"Unsupported observation space: {env.observation_space}")
    
    n_actions = env.action_space.n
    
    memory_type = args.memory_type if args.memory_type != 'none' else None
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    if 'state_encoder.weight' in state_dict:
        actual_n_embed = state_dict['state_encoder.weight'].shape[0]
        if actual_n_embed != args.n_embed:
            print(f"WARNING: Determined n_embed={actual_n_embed} from model, overriding argument n_embed={args.n_embed}")
            n_embed = actual_n_embed
        else:
            n_embed = args.n_embed
    else:
        n_embed = args.n_embed
    
    if memory_type is not None and 'memory_proj.weight' in state_dict:
        actual_memory_dim = state_dict['memory_proj.weight'].shape[1]
        if actual_memory_dim != args.memory_dim:
            print(f"WARNING: Determined memory_dim={actual_memory_dim} from model, overriding argument memory_dim={args.memory_dim}")
            memory_dim = actual_memory_dim
        else:
            memory_dim = args.memory_dim
    else:
        memory_dim = args.memory_dim
    
    layer_count = 0
    for k in state_dict.keys():
        if k.startswith('transformer.layers.') and '.norm1.weight' in k:
            layer_idx = int(k.split('.')[2])
            layer_count = max(layer_count, layer_idx + 1)
    
    if layer_count > 0 and layer_count != args.n_layer:
        print(f"WARNING: Determined n_layer={layer_count} from model, overriding argument n_layer={args.n_layer}")
        n_layer = layer_count
    else:
        n_layer = args.n_layer
    
    if 'transformer.layers.0.self_attn.in_proj_weight' in state_dict:
        in_proj_weight_shape = state_dict['transformer.layers.0.self_attn.in_proj_weight'].shape
        attn_head_size = n_embed // args.n_head
        actual_n_head = in_proj_weight_shape[0] // 3 // attn_head_size
        
        if actual_n_head != args.n_head:
            print(f"WARNING: Determined n_head={actual_n_head} from model, overriding argument n_head={args.n_head}")
            n_head = actual_n_head
        else:
            n_head = args.n_head
    else:
        n_head = args.n_head
    
    action_head_sequential = False
    for key in state_dict.keys():
        if key.startswith('action_head.') and '.' in key.split('action_head.')[1]:
            action_head_sequential = True
            break
    
    print(f"Model parameters: n_embed={n_embed}, n_layer={n_layer}, n_head={n_head}, memory_dim={memory_dim}")
    print(f"action_head structure: {'Sequential' if action_head_sequential else 'Linear'}")
    
    model = MemoryDecisionTransformer(
        state_dim=state_dim,
        n_actions=n_actions,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        context_length=args.context_length,
        memory_type=memory_type,
        memory_dim=memory_dim
    )
    
    if action_head_sequential:
        dropout = 0.1  # Default dropout value
        model.action_head = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.LayerNorm(n_embed),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_embed, n_actions)
        )
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to load state dict with custom mapping...")
        
        new_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('action_head.'):
                if action_head_sequential:
                    new_state_dict[key] = value
                else:
                    if key == 'action_head.4.weight':
                        new_state_dict['action_head.weight'] = value
                    elif key == 'action_head.4.bias':
                        new_state_dict['action_head.bias'] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded with modified state dict.")
    
    return model


def evaluate_agent(env, model, num_episodes, context_length=20, target_return=None, delay=0.05):
    """Evaluate and visualize the agent."""
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    def preprocess_obs(obs):
        """Helper to preprocess observations for the model."""
        if isinstance(obs, dict):
            if 'observation' in obs and 'mask' in obs:
                preproc_obs = np.concatenate([
                    obs['observation'].flatten(),
                    np.array([obs['mask']], dtype=np.float32)
                ])
                
                is_visible = obs['mask'] == 1
                if not is_visible:
                    print("Observation HIDDEN (flickering)")
            else:
                preproc_obs = np.concatenate([
                    o.flatten() if hasattr(o, 'flatten') else np.array([o], dtype=np.float32)
                    for o in obs.values()
                ])
        else:
            preproc_obs = obs
            
            if len(preproc_obs) == 4:
                preproc_obs[0] = preproc_obs[0] / 3.0
                
                preproc_obs[0] = preproc_obs[0] / 3.0
                
                preproc_obs[1] = np.clip(preproc_obs[1] / 5.0, -1.0, 1.0)
                
                angle = preproc_obs[2]
                
                preproc_obs[3] = np.clip(preproc_obs[3] / 5.0, -1.0, 1.0)
                
                sin_theta = np.sin(angle)
                cos_theta = np.cos(angle)
                
                preproc_obs = np.array([
                    preproc_obs[0],    # normalized position
                    preproc_obs[1],    # normalized velocity
                    sin_theta,         # sin of the angle
                    cos_theta,         # cos of the angle
                    preproc_obs[3]     # normalized angular velocity
                ], dtype=np.float32)
        
        if np.isnan(preproc_obs).any() or np.isinf(preproc_obs).any():
            print("WARNING: NaN or Inf in observation, replacing with zeros")
            preproc_obs = np.nan_to_num(preproc_obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return preproc_obs
    
    if target_return is None:
        if isinstance(env, VelocityCartPoleEnv):
            target_return = 500.0
        elif isinstance(env, FlickeringPendulumEnv):
            target_return = -200.0
        elif isinstance(env, LiDARMountainCarEnv):
            target_return = -100.0
        else:
            target_return = 500.0
    
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        model.reset_memory()
        
        states = []
        actions = []
        returns_to_go = []
        
        episode_return = 0
        done = False
        timestep = 0
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print("=" * 40)
        print(f"Target return: {target_return}")
        
        while not done:
            processed_obs = preprocess_obs(obs)
            states.append(processed_obs)

            context_size = min(len(states), context_length)
            context_states = np.array(states[-context_size:])

            if context_size == 1:
                context_actions = np.array([0])
            elif len(actions) >= context_size - 1:
                context_actions = np.array(actions[-(context_size - 1):] + [0])
            else:
                context_actions = np.array(
                    actions + [0] * (context_size - 1 - len(actions))
                )

            rtg = target_return - episode_return
            context_rtgs = np.full(context_size, rtg, dtype=np.float32)

            action = model.get_action(
                states=context_states,
                actions=context_actions,
                rtgs=context_rtgs.reshape(-1, 1),
                device=device
            )

            print(f"Step {timestep + 1}: Action={action}, RTG={rtg:.2f}")

            # states.append(processed_obs)
            #
            # if len(states) <= 1:
            #     action = 0
            #     print(f"Step {timestep+1}: Initial action={action} (default)")
            # else:
            #     context_size = min(len(states), context_length)
            #     context_states = np.array(states[-context_size:])
            #
            #     if len(actions) >= context_size - 1:
            #         context_actions = np.array(actions[-(context_size-1):] + [0])
            #     else:
            #         context_actions = np.array(actions + [0] * (context_size - 1 - len(actions)))
            #
            #     rtg = target_return - episode_return
            #     context_rtgs = np.full(context_size, rtg)
            #
            #     action = model.get_action(
            #         states=context_states,
            #         actions=context_actions,
            #         rtgs=context_rtgs.reshape(-1, 1),
            #         device=device
            #     )
            #
            #     print(f"Step {timestep+1}: Action={action}, RTG={rtg:.2f}")
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            actions.append(action)
            episode_return += reward
            returns_to_go.append(target_return - episode_return)
            
            obs = next_obs
            timestep += 1
            
            time.sleep(delay)
        
        total_rewards.append(episode_return)
        print(f"Episode {episode+1} finished with reward: {episode_return}")
    
    print("\nEvaluation complete!")
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    
    return total_rewards


def main():
    args = parse_args()
    
    env = create_env(args.env, render_mode='human')
    
    model = load_model(args.model_path, env, args)
    
    print(f"Visualizing {args.env} agent from {args.model_path}")
    print(f"Memory type: {args.memory_type}, Context length: {args.context_length}")
    
    if hasattr(model, 'enable_exploration'):
        model.enable_exploration(False)
        print("Exploration disabled for visualization")
    
    evaluate_agent(
        env=env, 
        model=model, 
        num_episodes=args.num_episodes, 
        context_length=args.context_length,
        target_return=args.target_return,
        delay=args.delay
    )
    
    env.close()


if __name__ == "__main__":
    main() 