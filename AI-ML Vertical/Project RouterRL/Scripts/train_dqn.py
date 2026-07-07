"""
RouterRL - Step 5: Train a DQN agent and compare it against the heuristic baselines.

Usage:
    python train_dqn.py routing_table.parquet --timesteps 200000
"""
import argparse
import random
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from router_env import RouterEnv, MODELS


def run_policy(df, action_fn, n_episodes=20, budget=10000, episode_length=60, seed0=1000):
    """Average a fixed-action-rule policy (e.g. always model X) over many episodes."""
    rewards, answered_list, skipped_list = [], [], []
    for i in range(n_episodes):
        env = RouterEnv(df, budget=budget, episode_length=episode_length, seed=seed0 + i)
        obs, info = env.reset(seed=seed0 + i)
        total_reward = 0
        while True:
            action = action_fn()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        answered_list.append(info["answered"])
        skipped_list.append(info["skipped"])
    return np.mean(rewards), np.std(rewards), np.mean(answered_list), np.mean(skipped_list)


def run_trained_policy(df, model, n_episodes=20, budget=10000, episode_length=60, seed0=1000):
    rewards, answered_list, skipped_list = [], [], []
    for i in range(n_episodes):
        env = RouterEnv(df, budget=budget, episode_length=episode_length, seed=seed0 + i)
        obs, info = env.reset(seed=seed0 + i)
        total_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        answered_list.append(info["answered"])
        skipped_list.append(info["skipped"])
    return np.mean(rewards), np.std(rewards), np.mean(answered_list), np.mean(skipped_list)


def make_env(df, budget, episode_length, seed):
    def _init():
        env = RouterEnv(df, budget=budget, episode_length=episode_length, seed=seed)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--budget", type=float, default=10000)
    parser.add_argument("--episode_length", type=int, default=60)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--out", type=str, default="dqn_router.zip")
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} rows")

    train_env = make_env(df, args.budget, args.episode_length, seed=0)()

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
    )

    print(f"Training DQN for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved trained model to {args.out}")

    print("\n=== Evaluating baselines and trained agent over "
          f"{args.eval_episodes} episodes each ===")

    results = {}
    for name, action_fn in [
        ("Always cheapest (qwen3-0.6b)", lambda: 0),
        ("Always ministral-8b", lambda: 1),
        ("Always qwen3-30b-a3b", lambda: 2),
        ("Always best (qwen3-30b-a3b-instruct)", lambda: 3),
        ("Random", lambda: random.randint(0, 3)),
    ]:
        mean_r, std_r, mean_a, mean_s = run_policy(
            df, action_fn, n_episodes=args.eval_episodes,
            budget=args.budget, episode_length=args.episode_length
        )
        results[name] = (mean_r, std_r, mean_a, mean_s)

    mean_r, std_r, mean_a, mean_s = run_trained_policy(
        df, model, n_episodes=args.eval_episodes,
        budget=args.budget, episode_length=args.episode_length
    )
    results["Trained DQN agent"] = (mean_r, std_r, mean_a, mean_s)

    print(f"\n{'Policy':<40}{'Mean Reward':<15}{'Std':<10}{'Avg Answered':<15}{'Avg Skipped':<12}")
    for name, (mean_r, std_r, mean_a, mean_s) in results.items():
        print(f"{name:<40}{mean_r:<15.3f}{std_r:<10.3f}{mean_a:<15.2f}{mean_s:<12.2f}")


if __name__ == "__main__":
    main()
