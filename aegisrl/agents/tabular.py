from pathlib import Path

import numpy as np

from .base import Policy
from .discretize import NUM_STATES, to_index


class TabularQAgent(Policy):
    """Q-learning with a lookup table over bucketed observations."""

    name = "TABULAR_Q"

    def __init__(self, lr: float = 0.1, gamma: float = 0.95, seed: int = 0):
        self.lr = lr
        self.gamma = gamma
        self.q = np.zeros((NUM_STATES, 4))
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        row = self.q[to_index(obs)]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def train(self, env, episodes: int, eps_start: float = 1.0, eps_end: float = 0.05,
              log_every: int = 5000, seed_offset: int = 0) -> list[float]:
        returns = []
        decay_episodes = int(episodes * 0.8)

        for ep in range(episodes):
            epsilon = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_episodes)
            obs, _ = env.reset(seed=seed_offset + ep)
            state = to_index(obs)
            total, done = 0.0, False

            while not done:
                if self.rng.random() < epsilon:
                    action = int(self.rng.integers(4))
                else:
                    action = self.act(obs)

                obs, reward, terminated, truncated, _ = env.step(action)
                next_state = to_index(obs)

                if terminated:
                    target = reward
                else:
                    target = reward + self.gamma * self.q[next_state].max()
                self.q[state, action] += self.lr * (target - self.q[state, action])

                state = next_state
                total += reward
                done = terminated or truncated

            returns.append(total)
            if log_every and (ep + 1) % log_every == 0:
                recent = np.mean(returns[-log_every:])
                print(f"episode {ep + 1:>6}  epsilon {epsilon:.2f}  avg return {recent:7.2f}")

        return returns

    def save(self, path: str | Path) -> None:
        np.save(path, self.q)

    def load(self, path: str | Path) -> None:
        self.q = np.load(path)