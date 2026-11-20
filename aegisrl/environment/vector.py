import numpy as np

from .env import HospitalSessionEnv


class VectorEnv:
    """Runs several environments side by side and resets each one as soon as its session ends."""

    def __init__(self, num_envs: int, seed: int = 0, **env_kwargs):
        self.envs = [HospitalSessionEnv(**env_kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.next_seed = seed
        self.episode_returns = np.zeros(num_envs)
        self.finished_returns: list[float] = []

    def _new_seed(self) -> int:
        self.next_seed += 1
        return self.next_seed - 1

    def reset(self) -> np.ndarray:
        self.episode_returns[:] = 0.0
        return np.stack([env.reset(seed=self._new_seed())[0] for env in self.envs])

    def step(self, actions: np.ndarray):
        obs = np.zeros((self.num_envs, self.envs[0].observation_space.shape[0]), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        final_obs = [None] * self.num_envs

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, _ = env.step(int(action))
            self.episode_returns[i] += r
            if term or trunc:
                final_obs[i] = o
                self.finished_returns.append(self.episode_returns[i])
                self.episode_returns[i] = 0.0
                o, _ = env.reset(seed=self._new_seed())
            obs[i], rewards[i], terminated[i], truncated[i] = o, r, term, trunc

        return obs, rewards, terminated, truncated, final_obs