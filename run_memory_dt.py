import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import json
import csv
from datetime import datetime

from pomdp_envs.velocity_cartpole import VelocityCartPoleEnv
from pomdp_envs.flickering_pendulum import FlickeringPendulumEnv
from pomdp_envs.lidar_mountain_car import LiDARMountainCarEnv
from memory_dt import train_memory_dt, evaluate_memory_dt, MemoryDecisionTransformer

def set_global_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='velocity_cartpole',
                        choices=['velocity_cartpole', 'flickering_pendulum', 'lidar_mountain_car'],
                        help='POMDP environment to use')
    parser.add_argument('--data_dir', type=str, default='pomdp_datasets',
                        help='Directory containing collected trajectories')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--context_length', type=int, default=20,
                        help='Context length for the transformer')
    parser.add_argument('--n_embed', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--memory_type', type=str, default='gru',
                        choices=['gru', 'lstm', 'tdm', 'none'],
                        help='Type of memory to use (gru, lstm, tdm or none)')
    parser.add_argument('--memory_dim', type=int, default=64,
                        help='Dimension of memory state')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--target_return', type=float, default=None,
                        help='Target return for cartpole (adjust for each env)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained model to load (skip training)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for experiment logs')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional custom name for this run')
    parser.add_argument("--eval_rtg_mode", type=str,default="constant",
        choices=["constant", "history"],
        help=(
            "How to construct RTG context during evaluation: "
            "'constant' keeps the original template behavior; "
            "'history' uses per-timestep RTG history."
        ),
    )

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


def load_model(model_path, env, args):
    """Load a pre-trained model and create the model instance."""
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
    
    model = MemoryDecisionTransformer(
        state_dim=state_dim,
        n_actions=n_actions,
        n_embed=args.n_embed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        context_length=args.context_length,
        memory_type=args.memory_type if args.memory_type != 'none' else None,
        memory_dim=args.memory_dim,
        ##debug=args.debug
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return model


def main():
    args = parse_args()
    set_global_seed(args.seed)

    memory_label = args.memory_type

    if args.memory_type == 'none':
        args.memory_type = None
    
    dataset_path = os.path.join(args.data_dir, args.env)
    
    env = create_env(args.env)

    if args.target_return is not None:
        target_return = args.target_return
    elif args.env == 'velocity_cartpole':
        target_return = 500.0
    elif args.env == 'flickering_pendulum':
        target_return = -200.0 # the goal is to minimize loss in pendulum
    elif args.env == 'lidar_mountain_car':
        target_return = -100.0 # the goal is to reach the flag with minimum steps
    else:
        raise ValueError(
            f"No default target_return for env={args.env}. "
            "Please pass --target_return explicitly."
        )

    # Create a separate directory for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"{timestamp}_{args.env}_{memory_label}"
        f"_ctx{args.context_length}_seed{args.seed}"
    )

    run_dir = os.path.join(args.results_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run directory: {run_dir}")

    # Save run configuration
    config_to_save = vars(args).copy()
    config_to_save["memory_label"] = memory_label
    config_to_save["run_name"] = run_name
    config_to_save["run_dir"] = run_dir
    config_to_save["dataset_path"] = dataset_path
    config_to_save["resolved_target_return"] = target_return

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    print(f"Saved config to: {config_path}")

    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}")
        model = load_model(args.load_model, env, args)
        train_losses = []
    else:
        print(f"Training Memory Decision Transformer on {args.env}...")
        model, train_losses, val_returns = train_memory_dt(
            env_name=args.env,
            dataset_path=dataset_path,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            context_length=args.context_length,
            n_embed=args.n_embed,
            n_layer=args.n_layer,
            n_head=args.n_head,
            memory_type=args.memory_type,
            memory_dim=args.memory_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            debug=args.debug,
            run_dir=run_dir,
            run_config=config_to_save
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Evaluating model on {args.env}...")
    mean_return, returns, success_rate, episode_lengths = evaluate_memory_dt(
        model=model,
        env=env,
        num_episodes=args.eval_episodes,
        render=args.render,
        target_return=target_return,
        context_length=args.context_length,
        debug=args.debug,
        seed=args.seed + 10000,
        eval_rtg_mode=args.eval_rtg_mode,
    )
    
    print(f"Evaluation complete. Mean return: {mean_return:.2f}, Success rate: {success_rate:.2%}")

    final_eval = {
        "env": args.env,
        "memory_label": memory_label,
        "memory_type": memory_label,
        "run_name": run_name,
        "run_dir": run_dir,
        "evaluated_checkpoint": "best" if not args.load_model else args.load_model,
        "target_return": target_return,
        "eval_rtg_mode": args.eval_rtg_mode,
        "num_eval_episodes": args.eval_episodes,
        "mean_return": float(mean_return),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "success_rate": float(success_rate),
        "returns": [float(r) for r in returns],
        "lengths": [int(l) for l in episode_lengths]
    }

    final_eval_path = os.path.join(run_dir, "final_eval.json")
    with open(final_eval_path, "w", encoding="utf-8") as f:
        json.dump(final_eval, f, indent=2)

    print(f"Saved final evaluation to: {final_eval_path}")

    eval_episodes_path = os.path.join(run_dir, "eval_episodes.csv")
    with open(eval_episodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "return",
                "length",
                "success"
            ]
        )
        writer.writeheader()

        for episode_idx, episode_return in enumerate(returns, start=1):
            writer.writerow({
                "episode": episode_idx,
                "return": float(episode_return),
                "length": int(episode_lengths[episode_idx - 1]),
                "success": int(
                    episode_return >= 450.0 if args.env == "velocity_cartpole"
                    else episode_return >= -250.0 if args.env == "flickering_pendulum"
                    else episode_return >= -100.0 if args.env == "lidar_mountain_car"
                    else False
                )
            })

    print(f"Saved evaluation episodes to: {eval_episodes_path}")

    summary_path = os.path.join(args.results_dir, "experiments_summary.csv")
    os.makedirs(args.results_dir, exist_ok=True)

    summary_fieldnames = [
        "run_name",
        "env",
        "memory_type",
        "context_length",
        "seed",
        "n_epochs",
        "batch_size",
        "n_embed",
        "n_layer",
        "n_head",
        "memory_dim",
        "learning_rate",
        "weight_decay",
        "target_return",
        "num_eval_episodes",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "success_rate",
        "mean_length",
        "std_length",
        "run_dir",
        "eval_rtg_mode",
    ]

    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "run_name": run_name,
            "env": args.env,
            "memory_type": memory_label,
            "context_length": args.context_length,
            "seed": args.seed,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "n_embed": args.n_embed,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "memory_dim": args.memory_dim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "target_return": target_return,
            "num_eval_episodes": args.eval_episodes,
            "mean_return": float(mean_return),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "success_rate": float(success_rate),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
            "run_dir": run_dir,
            "eval_rtg_mode": args.eval_rtg_mode,
        })

    print(f"Updated experiments summary: {summary_path}")

if __name__ == "__main__":
    main() 