from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from aegisrl.environment.features import OBS_DIM
from .base import Policy

N_ACTIONS = 4


def layer(in_dim: int, out_dim: int, gain: float = np.sqrt(2)) -> nn.Linear:
    """Linear layer with orthogonal initialisation, the standard choice for PPO."""
    linear = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)
    return linear


class ActorCritic(nn.Module):
    """Two separate networks: the actor picks actions, the critic estimates state value."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(),
            layer(hidden, hidden), nn.Tanh(),
            layer(hidden, n_actions, gain=0.01),
        )
        self.critic = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(),
            layer(hidden, hidden), nn.Tanh(),
            layer(hidden, 1, gain=1.0),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

class PPOAgent(Policy):
    name = "PPO"

    def __init__(self, gamma=0.95, gae_lambda=0.95, lr=3e-4, clip=0.2, ent_coef=0.01,
                 vf_coef=0.5, epochs=4, minibatch_size=256, max_grad_norm=0.5,
                 seed=0, device=None):
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip, self.ent_coef, self.vf_coef = clip, ent_coef, vf_coef
        self.epochs, self.minibatch_size, self.max_grad_norm = epochs, minibatch_size, max_grad_norm
        self.lr = lr
        self.net = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)

    def _tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        logits = self.net.actor(self._tensor(obs).unsqueeze(0))
        return int(logits.argmax(dim=1).item())

    @torch.no_grad()
    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        return self.net.distribution(self._tensor(obs).unsqueeze(0)).probs.squeeze(0).cpu().numpy()

    def collect(self, envs, obs: np.ndarray, n_steps: int):
        n = envs.num_envs
        buf_obs = np.zeros((n_steps, n, OBS_DIM), dtype=np.float32)
        buf_actions = np.zeros((n_steps, n), dtype=np.int64)
        buf_logp = np.zeros((n_steps, n), dtype=np.float32)
        buf_values = np.zeros((n_steps, n), dtype=np.float32)
        buf_rewards = np.zeros((n_steps, n), dtype=np.float32)
        buf_dones = np.zeros((n_steps, n), dtype=np.float32)

        for t in range(n_steps):
            with torch.no_grad():
                x = self._tensor(obs)
                dist = self.net.distribution(x)
                actions = dist.sample()
                logp = dist.log_prob(actions)
                values = self.net.value(x)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, final_obs = envs.step(actions_np)

            for i in np.flatnonzero(truncated):
                with torch.no_grad():
                    rewards[i] += self.gamma * self.net.value(self._tensor(final_obs[i]).unsqueeze(0)).item()

            buf_obs[t], buf_actions[t] = obs, actions_np
            buf_logp[t], buf_values[t] = logp.cpu().numpy(), values.cpu().numpy()
            buf_rewards[t], buf_dones[t] = rewards, terminated | truncated
            obs = next_obs

        with torch.no_grad():
            last_value = self.net.value(self._tensor(obs)).cpu().numpy()

        advantages = np.zeros_like(buf_rewards)
        gae = np.zeros(n, dtype=np.float32)
        for t in reversed(range(n_steps)):
            next_value = last_value if t == n_steps - 1 else buf_values[t + 1]
            not_done = 1.0 - buf_dones[t]
            delta = buf_rewards[t] + self.gamma * next_value * not_done - buf_values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages + buf_values

        batch = {
            "obs": buf_obs.reshape(-1, OBS_DIM),
            "actions": buf_actions.reshape(-1),
            "logp": buf_logp.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }
        return batch, obs

    def update(self, batch: dict) -> dict:
        data = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        size = len(batch["actions"])
        stats = []

        for _ in range(self.epochs):
            order = self.rng.permutation(size)
            for start in range(0, size, self.minibatch_size):
                idx = torch.as_tensor(order[start:start + self.minibatch_size], device=self.device)
                obs, actions = data["obs"][idx], data["actions"][idx]
                old_logp, returns = data["logp"][idx], data["returns"][idx]
                adv = data["advantages"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                dist = self.net.distribution(obs)
                logp = dist.log_prob(actions)
                ratio = torch.exp(logp - old_logp)

                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv,
                ).mean()
                value_loss = 0.5 * (self.net.value(obs) - returns).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats.append((policy_loss.item(), value_loss.item(), entropy.item(),
                              ((ratio - 1).abs() > self.clip).float().mean().item()))

        p, v, e, c = np.mean(stats, axis=0)
        return {"policy_loss": p, "value_loss": v, "entropy": e, "clip_fraction": c}
    def train(self, envs, total_steps: int, n_steps: int = 256, log_every: int = 10) -> list[float]:
        obs = envs.reset()
        iterations = total_steps // (n_steps * envs.num_envs)

        for it in range(1, iterations + 1):
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr * (1 - (it - 1) / iterations)

            batch, obs = self.collect(envs, obs, n_steps)
            stats = self.update(batch)

            if log_every and it % log_every == 0:
                recent = np.mean(envs.finished_returns[-500:])
                steps = it * n_steps * envs.num_envs
                print(f"step {steps:>8}  avg return {recent:6.2f}  entropy {stats['entropy']:.3f}  "
                      f"clip {stats['clip_fraction']:.3f}  value loss {stats['value_loss']:.3f}")

        return envs.finished_returns

    def save(self, path: str | Path) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))