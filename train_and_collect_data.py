import os
import argparse
import gymnasium as gym
import numpy as np
import torch

from recurrent_ppo import RecurrentPPO
from pomdp_envs.velocity_cartpole import VelocityCartPoleEnv
from pomdp_envs.flickering_pendulum import FlickeringPendulumEnv
from pomdp_envs.lidar_mountain_car import LiDARMountainCarEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='velocity_cartpole',
                        choices=['velocity_cartpole', 'flickering_pendulum', 'lidar_mountain_car'],
                        help='POMDP environment to use')
    parser.add_argument('--num_trajectories', type=int, default=1000,
                        help='Number of trajectories to collect')
    parser.add_argument('--output_dir', type=str, default='pomdp_datasets',
                        help='Directory to save collected trajectories')
    parser.add_argument('--train_timesteps', type=int, default=100_000,
                        help='Number of timesteps to train the recurrent PPO agent')
    parser.add_argument('--hidden_dim', type=int, default=128, # 64
                        help='Hidden dimension for the recurrent policy')
    parser.add_argument('--rnn_type', type=str, default='gru', choices=['gru', 'lstm'],
                        help='Type of RNN backbone to use (GRU or LSTM)')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='Whether to save the trained PPO model')
    parser.add_argument('--render', type=bool, default=False,
                        help='Whether to render the environment during data collection')
    parser.add_argument('--collect_data_from_checkpoint', action='store_true',
                        help='If True, collect data using a pre-trained model checkpoint. '
                             'Otherwise, train the agent from scratch before collecting data.')
    parser.add_argument('--reward_threshold', type=float, default=None,
                        help='Reward threshold for collecting data. If None, all trajectories are collected. '
                             'Environment max rewards: CartPole: 500.0, FlickeringPendulum: -200.0, LiDARMountainCar: -100.0. '
                             'Recommended thresholds: CartPole: 475.0, FlickeringPendulum: ???, LiDARMountainCar: ???')
    parser.add_argument('--rtg_gamma', type=float, default=1.0,
        help='Discount factor used only for return-to-go saved in DT dataset. '
             'Use 1.0 for standard undiscounted Decision Transformer RTG.')
    return parser.parse_args()


def create_env(env_name):
    if env_name == 'velocity_cartpole':
        return VelocityCartPoleEnv()
    elif env_name == 'flickering_pendulum':
        return FlickeringPendulumEnv(flicker_probability=0.3)
    elif env_name == 'lidar_mountain_car':
        return LiDARMountainCarEnv(num_sensors=8)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def main():
    args = parse_args()
    
    env = create_env(args.env)
    
    output_dir = os.path.join(args.output_dir, args.env)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.env == 'velocity_cartpole':
        print("Using optimized parameters for velocity_cartpole")
        hidden_dim = 128
        rnn_type = 'gru'
        lr = 0.0005
        gamma = 0.99
        gae_lambda = 0.95
        clip_ratio = 0.2
        epochs = 4
        sequence_length = 128
        reward_threshold = 450 if args.reward_threshold is None else args.reward_threshold
    else:
        hidden_dim = args.hidden_dim
        rnn_type = args.rnn_type
        lr = 0.0003
        gamma = 0.99
        gae_lambda = 0.95
        clip_ratio = 0.2
        epochs = 10
        sequence_length = 64
        reward_threshold = args.reward_threshold
    
    ppo_agent = RecurrentPPO(
        env=env,
        hidden_dim=hidden_dim,
        rnn_type=rnn_type,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        epochs=epochs,
        sequence_length=sequence_length,
        reward_threshold=reward_threshold
    )
    
    model_path = os.path.join(output_dir, f"recurrent_ppo_{args.env}.pt")
    checkpoint_exists = os.path.exists(model_path)
    
    # Case 1: Checkpoint exists and --collect_data_from_checkpoint=True
    if checkpoint_exists and args.collect_data_from_checkpoint:
        print(f"Loading model from checkpoint: {model_path}")
        ppo_agent.policy.load_state_dict(torch.load(model_path))
    # Case 2 & 3: Either checkpoint doesn't exist or --collect_data_from_checkpoint=False
    else:
        #train_timesteps = 300000 if args.env == 'velocity_cartpole' else args.train_timesteps
        train_timesteps = args.train_timesteps
        train_reason = "from scratch" if not checkpoint_exists else "ignoring existing checkpoint"
        print(f"Training recurrent PPO agent {train_reason} on {args.env} for {train_timesteps} timesteps...")
        ppo_agent.learn(total_timesteps=train_timesteps)
        
        if args.save_model:
            torch.save(ppo_agent.policy.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Collect trajectories
    print(f"Collecting {args.num_trajectories} trajectories...")
    ppo_agent.save_trajectories(output_dir, args.num_trajectories, rtg_gamma=args.rtg_gamma)
    
    print("Done!")


if __name__ == "__main__":
    main() 