import argparse
import time
from pathlib import Path

from aegisrl.agents.tabular import TabularQAgent
from aegisrl.environment.env import HospitalSessionEnv
from aegisrl.evaluation.evaluate import evaluate, print_report


def main():
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on the hospital environment.")
    parser.add_argument("--episodes", type=int, default=60_000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models/tabular_q.npy"))
    args = parser.parse_args()

    agent = TabularQAgent(lr=args.lr, gamma=args.gamma, seed=args.seed)
    start = time.time()
    agent.train(HospitalSessionEnv(), args.episodes, log_every=10_000)
    print(f"trained in {time.time() - start:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.out)
    print(f"saved to {args.out}")

    print_report(agent.name, evaluate(agent))


if __name__ == "__main__":
    main()