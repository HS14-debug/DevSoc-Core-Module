"""
RouterRL - Step 3: The Gym environment.

One episode = a shuffled stream of `episode_length` turns sampled fresh from the
routing table. The agent picks which of the 4 models answers each turn, under a
shared token budget. Unaffordable picks are treated as a skip (0 reward, budget
untouched) rather than crashing the episode.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

MODELS = ["qwen3-0.6b", "ministral-8b", "qwen3-30b-a3b", "qwen3-30b-a3b-instruct"]

FEATURES = [
    "query_len_chars", "query_len_words", "query_len_sentences", "has_question_mark",
    "question_mark_count", "has_code", "has_math", "has_url", "avg_word_length",
    "uppercase_ratio", "special_char_ratio", "newline_count", "num_prior_turns",
    "total_prior_context_chars", "avg_prior_query_len", "avg_prior_response_len",
    "is_first_turn", "turn_position_ratio", "has_pronoun_reference",
    "has_continuation_marker", "has_correction_marker", "has_elaboration_request",
    "self_contained_score", "word_overlap_prev_query", "word_overlap_prev_response",
    "is_short_query", "difficulty",
]


class RouterEnv(gym.Env):
    """Budget-constrained LLM routing environment."""

    def __init__(self, df: pd.DataFrame, budget: float = 10000, episode_length: int = 60,
                 reward_lambda: float = 1.0, seed: int = None):
        super().__init__()
        # drop rows with any missing score/cost for the 4 models (real data has ~140 such rows)
        needed_cols = [f"{m}_score" for m in MODELS] + [f"{m}_cost" for m in MODELS] + FEATURES
        self.df = df.dropna(subset=needed_cols).reset_index(drop=True)
        assert len(self.df) >= episode_length, "Not enough clean rows for one episode."

        self.budget_total = budget
        self.episode_length = episode_length
        self.reward_lambda = reward_lambda
        self.rng = np.random.default_rng(seed)

        # precompute per-feature mean/std for normalization
        self._feat_mean = self.df[FEATURES].mean().values
        self._feat_std = self.df[FEATURES].std().replace(0, 1).values

        # state = normalized features + remaining_budget_fraction
        obs_dim = len(FEATURES) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(MODELS))

        self._episode_rows = None
        self._t = 0
        self._remaining_budget = budget

    def _get_obs(self):
        row = self._episode_rows.iloc[self._t]
        feats = (row[FEATURES].values.astype(np.float32) - self._feat_mean) / self._feat_std
        budget_frac = np.array([self._remaining_budget / self.budget_total], dtype=np.float32)
        return np.concatenate([feats, budget_frac]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        idx = self.rng.choice(len(self.df), size=self.episode_length, replace=False)
        self._episode_rows = self.df.iloc[idx].reset_index(drop=True)
        self._t = 0
        self._remaining_budget = self.budget_total
        self._answered = 0
        self._skipped = 0
        info = {}
        return self._get_obs(), info

    def step(self, action: int):
        row = self._episode_rows.iloc[self._t]
        model = MODELS[action]
        cost = float(row[f"{model}_cost"])
        score = float(row[f"{model}_score"])

        if cost <= self._remaining_budget:
            self._remaining_budget -= cost
            reward = (score / 10.0) - self.reward_lambda * (cost / self.budget_total)
            self._answered += 1
        else:
            # can't afford this model -> query goes unanswered, no cost, no credit
            reward = 0.0
            self._skipped += 1

        self._t += 1

        # Episode ends when we've gone through episode_length turns, OR the
        # remaining budget can't afford even the single cheapest model overall.
        terminated = False
        min_possible_cost = self.df[[f"{m}_cost" for m in MODELS]].min().min()
        truncated = bool((self._t >= self.episode_length) or (self._remaining_budget < min_possible_cost))

        obs = self._get_obs() if not truncated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "answered": self._answered,
            "skipped": self._skipped,
            "remaining_budget": self._remaining_budget,
            "model_chosen": model,
        }
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "routing_table.parquet"
    df = pd.read_parquet(path)
    env = RouterEnv(df, budget=10000, episode_length=60, reward_lambda=1.0, seed=42)

    obs, info = env.reset(seed=0)
    print("Obs shape:", obs.shape)
    print("Action space:", env.action_space)

    total_reward = 0
    steps = 0
    while True:
        action = env.action_space.sample()  # random policy, just to test mechanics
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    print(f"\nRandom policy episode finished after {steps} steps")
    print("Total reward:", round(total_reward, 3))
    print("Answered:", info["answered"], "| Skipped:", info["skipped"])
    print("Remaining budget:", round(info["remaining_budget"], 1))
