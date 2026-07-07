"""
RouterRL - Step 6: A real heuristic baseline (not just "always pick model X").

The heuristic uses the same `difficulty` signal the RL agent sees: route easy
queries to the cheap model, medium queries to the mid-cost model, and hard
queries to the best (and most expensive) model. Thresholds are computed as
the 33rd/66th percentile of `difficulty` in your actual data, so the buckets
are always roughly equal-sized regardless of dataset changes.

Usage:
    python evaluate_heuristic.py routing_table.parquet --dqn_model dqn_router.zip
"""
import argparse
import random
import numpy as np
import pandas as pd

from router_env import RouterEnv, MODELS

# action indices: 0=qwen3-0.6b (cheap), 1=ministral-8b, 2=qwen3-30b-a3b, 3=qwen3-30b-a3b-instruct (best)
CHEAP, MINISTRAL, MID, BEST = 0, 1, 2, 3


def make_difficulty_heuristic(low_thresh: float, high_thresh: float):
    """Route by difficulty tertile. Never picks ministral-8b since the EDA
    showed it's dominated (worse AND pricier than qwen3-30b-a3b)."""
    def heuristic(difficulty: float) -> int:
        if difficulty <= low_thresh:
            return CHEAP
        elif difficulty <= high_thresh:
            return MID
        else:
            return BEST
    return heuristic


def run_fixed_action_policy(df, action_fn, n_episodes, budget, episode_length, seed0=1000):
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


def run_heuristic_policy(df, heuristic_fn, n_episodes, budget, episode_length, seed0=1000):
    """Unlike the fixed-action baselines, this heuristic needs the raw
    (un-normalized) difficulty value for the CURRENT turn to decide, so we
    read it directly from the environment's internal episode rows."""
    rewards, answered_list, skipped_list = [], [], []
    for i in range(n_episodes):
        env = RouterEnv(df, budget=budget, episode_length=episode_length, seed=seed0 + i)
        obs, info = env.reset(seed=seed0 + i)
        total_reward = 0
        while True:
            current_difficulty = env._episode_rows.iloc[env._t]["difficulty"]
            action = heuristic_fn(current_difficulty)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        answered_list.append(info["answered"])
        skipped_list.append(info["skipped"])
    return np.mean(rewards), np.std(rewards), np.mean(answered_list), np.mean(skipped_list)


def run_trained_policy(df, model, n_episodes, budget, episode_length, seed0=1000):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--budget", type=float, default=10000)
    parser.add_argument("--episode_length", type=int, default=60)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--dqn_model", type=str, default=None, help="path to a trained DQN .zip, optional")
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} rows")

    low_thresh, high_thresh = df["difficulty"].quantile([0.33, 0.66])
    print(f"Difficulty tertile thresholds: low<= {low_thresh:.2f}, mid<= {high_thresh:.2f}, else best")
    heuristic_fn = make_difficulty_heuristic(low_thresh, high_thresh)

    results = {}
    for name, action_fn in [
        ("Always cheapest (qwen3-0.6b)", lambda: CHEAP),
        ("Always best (qwen3-30b-a3b-instruct)", lambda: BEST),
        ("Random", lambda: random.randint(0, 3)),
    ]:
        results[name] = run_fixed_action_policy(
            df, action_fn, args.eval_episodes, args.budget, args.episode_length
        )

    results["Difficulty-heuristic (rule-based)"] = run_heuristic_policy(
        df, heuristic_fn, args.eval_episodes, args.budget, args.episode_length
    )

    if args.dqn_model:
        from stable_baselines3 import DQN
        model = DQN.load(args.dqn_model)
        results["Trained DQN agent"] = run_trained_policy(
            df, model, args.eval_episodes, args.budget, args.episode_length
        )

    print(f"\n{'Policy':<40}{'Mean Reward':<15}{'Std':<10}{'Avg Answered':<15}{'Avg Skipped':<12}")
    for name, (mean_r, std_r, mean_a, mean_s) in results.items():
        print(f"{name:<40}{mean_r:<15.3f}{std_r:<10.3f}{mean_a:<15.2f}{mean_s:<12.2f}")


if __name__ == "__main__":
    main()
