from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aegisrl.environment.features import OBS_DIM
from .base import Policy

N_ACTIONS = 4


class DuelingQNetwork(nn.Module):
    """Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.body(obs)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)

class ReplayBuffer:
    """Fixed-size circular memory of past transitions."""

    def __init__(self, capacity: int, obs_dim: int = OBS_DIM, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.terminated[self.pos] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.terminated[idx]),
        )

class DQNAgent(Policy):
    """Dueling Double DQN with a replay buffer and a soft-updated target network."""

    name = "DQN"

    def __init__(self, gamma: float = 0.95, lr: float = 5e-4, batch_size: int = 128,
                 buffer_size: int = 100_000, tau: float = 0.005, seed: int = 0,
                 device: str | None = None):
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.rng = np.random.default_rng(seed)

        self.online = DuelingQNetwork().to(self.device)
        self.target = DuelingQNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, seed=seed)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(x).squeeze(0).cpu().numpy()

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.q_values(obs)))

    def learn(self) -> float:
        batch = self.buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, terminated = (t.to(self.device) for t in batch)

        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_next = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, best_next).squeeze(1)
            target = rewards + self.gamma * (1.0 - terminated) * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        with torch.no_grad():
            for t, o in zip(self.target.parameters(), self.online.parameters()):
                t.mul_(1 - self.tau).add_(self.tau * o)

        return loss.item()

    def save(self, path: str | Path) -> None:
        torch.save(self.online.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(state)
        self.target.load_state_dict(self.online.state_dict())

    def train(self, env, total_steps: int, learning_starts: int = 2_000, train_every: int = 2,
              eps_start: float = 1.0, eps_end: float = 0.05, explore_fraction: float = 0.5,
              log_every: int = 20_000, seed_offset: int = 0) -> list[float]:
        returns, episode_return, episode = [], 0.0, 0
        obs, _ = env.reset(seed=seed_offset + episode) 
        explore_steps = int(total_steps * explore_fraction)

        for step in range(total_steps):
            epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / explore_steps)
            if self.rng.random() < epsilon:
                action = int(self.rng.integers(N_ACTIONS))
            else:
                action = self.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            self.buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward

            if terminated or truncated:
                returns.append(episode_return)
                episode_return, episode = 0.0, episode + 1
                obs, _ = env.reset(seed=seed_offset + episode)

            if step >= learning_starts and step % train_every == 0:
                self.learn()

            if log_every and (step + 1) % log_every == 0:
                recent = np.mean(returns[-500:]) if returns else float("nan")
                print(f"step {step + 1:>7}  epsilon {epsilon:.2f}  episodes {episode:>6}  avg return {recent:7.2f}")

        return returns