import argparse
import json
import time
from pathlib import Path

import numpy as np

from aegisrl.agents.dqn import DQNAgent
from aegisrl.agents.ppo import PPOAgent
from aegisrl.agents.tabular import TabularQAgent
from aegisrl.environment.env import HospitalSessionEnv
from aegisrl.environment.vector import VectorEnv
from aegisrl.evaluation.evaluate import evaluate

METRICS = ["overall", "attacks_fully_stopped", "leak_per_attack", "legit_served",
           "false_denial_rate", "emergency_denial_rate", "legit_challenge_rate"]


def train_agent(algo: str, seed: int, args):
    """Train one agent with one seed, or load it if it was already trained."""
    folder = Path("models/seeds")
    folder.mkdir(parents=True, exist_ok=True)

    if algo == "tabular":
        agent, path = TabularQAgent(seed=seed), folder / f"tabular_seed{seed}.npy"
    elif algo == "dqn":
        agent, path = DQNAgent(seed=seed, device="cpu"), folder / f"dqn_seed{seed}.pt"
    else:
        agent, path = PPOAgent(seed=seed, device="cpu"), folder / f"ppo_seed{seed}.pt"

    if path.exists():
        agent.load(path)
        print(f"  loaded {path}")
        return agent

    start = time.time()
    offset = seed * 10_000_000   # each seed trains on different sessions
    if algo == "tabular":
        agent.train(HospitalSessionEnv(), args.tabular_episodes, log_every=0, seed_offset=offset)
    elif algo == "dqn":
        agent.train(HospitalSessionEnv(), args.dqn_steps, log_every=0, seed_offset=offset)
    else:
        agent.train(VectorEnv(8, seed=offset), args.ppo_steps, log_every=0)
    agent.save(path)
    print(f"  trained {path} in {time.time() - start:.0f}s")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate each learning agent over several seeds.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--algos", nargs="+", default=["tabular", "dqn", "ppo"])
    parser.add_argument("--tabular-episodes", type=int, default=60_000)
    parser.add_argument("--dqn-steps", type=int, default=300_000)
    parser.add_argument("--ppo-steps", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, default=Path("reports/seed_results.json"))
    args = parser.parse_args()

    results = {}
    for algo in args.algos:
        print(f"\n=== {algo} ===")
        results[algo] = []
        for seed in range(args.seeds):
            agent = train_agent(algo, seed, args)
            r = evaluate(agent)
            results[algo].append(r)
            print(f"  seed {seed}: overall {r['overall']:.2f}  stopped {r['attacks_fully_stopped']:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"\n{'agent':<10}" + "".join(f"{m:>24}" for m in METRICS))
    for algo, runs in results.items():
        cells = []
        for m in METRICS:
            values = np.array([r[m] for r in runs])
            cells.append(f"{values.mean():>13.3f} ± {values.std(ddof=1) if len(values) > 1 else 0:.3f}")
        print(f"{algo:<10}" + "".join(f"{c:>24}" for c in cells))
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()