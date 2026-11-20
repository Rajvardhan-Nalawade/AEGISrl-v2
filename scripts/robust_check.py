import argparse
import json
from pathlib import Path

import numpy as np

from aegisrl.agents.dqn import DQNAgent
from aegisrl.agents.ppo import PPOAgent
from aegisrl.agents.rules import RBACPolicy, RiskRulesPolicy
from aegisrl.agents.tabular import TabularQAgent
from aegisrl.environment.actors import (
    AdaptiveAttacker, EmergencyClinician, Insider,
    LegitClinician, Scraper, StolenCredential,
)
from aegisrl.environment.env import HospitalSessionEnv
from aegisrl.evaluation.evaluate import evaluate

SCENARIOS = {
    "standard": dict(),
    "noisy_signals": dict(noise=1.5),
    "realistic_mix": dict(actor_mix={
        LegitClinician: 0.85, EmergencyClinician: 0.10, StolenCredential: 0.0125,
        Insider: 0.0125, Scraper: 0.0125, AdaptiveAttacker: 0.0125,
    }),
    "adaptive_wave": dict(actor_mix={
        LegitClinician: 0.45, EmergencyClinician: 0.10, AdaptiveAttacker: 0.45,
    }),
}

METRICS = ["overall", "attacks_fully_stopped", "leak_per_attack",
           "false_denial_rate", "emergency_denial_rate"]


def load_agent(algo: str, seed: int):
    folder = Path("models/seeds")
    if algo == "tabular":
        agent, path = TabularQAgent(seed=seed), folder / f"tabular_seed{seed}.npy"
    elif algo == "dqn":
        agent, path = DQNAgent(seed=seed, device="cpu"), folder / f"dqn_seed{seed}.pt"
    else:
        agent, path = PPOAgent(seed=seed, device="cpu"), folder / f"ppo_seed{seed}.pt"
    agent.load(path)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Test trained agents under conditions they never trained on.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/robustness.json"))
    args = parser.parse_args()

    results: dict = {}
    for scenario, env_kwargs in SCENARIOS.items():
        print(f"\n=== {scenario} ===")
        results[scenario] = {}

        def run(name, policy):
            r = evaluate(policy, args.episodes, HospitalSessionEnv(**env_kwargs))
            results[scenario].setdefault(name, []).append({m: r[m] for m in METRICS})

        run("rbac", RBACPolicy())
        run("risk_rules", RiskRulesPolicy())
        for algo in ["tabular", "dqn", "ppo"]:
            for seed in range(args.seeds):
                run(algo, load_agent(algo, seed))

        for name, runs in results[scenario].items():
            cells = []
            for m in METRICS:
                v = np.array([r[m] for r in runs])
                cells.append(f"{m}={v.mean():7.3f}" + (f"±{v.std(ddof=1):.3f}" if len(v) > 1 else "      "))
            print(f"  {name:<11}" + "  ".join(cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()