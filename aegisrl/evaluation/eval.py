import numpy as np

from aegisrl.environment.env import HospitalSessionEnv

EVAL_SEED_START = 1_000_000   # far away from training seeds (0, 1, 2, ...)


def evaluate(policy, episodes: int = 3000, env: HospitalSessionEnv | None = None) -> dict:
    """Run a policy on unseen sessions. Returns overall and per-actor average return."""
    env = env or HospitalSessionEnv()
    by_actor: dict[str, list[float]] = {}

    for i in range(episodes):
        obs, info = env.reset(seed=EVAL_SEED_START + i)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            total += reward
            done = terminated or truncated
        by_actor.setdefault(info["actor"], []).append(total)

    all_returns = [r for returns in by_actor.values() for r in returns]
    return {
        "overall": float(np.mean(all_returns)),
        "per_actor": {name: float(np.mean(r)) for name, r in sorted(by_actor.items())},
    }


def print_report(name: str, result: dict) -> None:
    per_actor = "  ".join(f"{k}={v:6.1f}" for k, v in result["per_actor"].items())
    print(f"{name:<12} overall={result['overall']:6.2f}  {per_actor}")