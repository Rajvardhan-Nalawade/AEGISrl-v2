import argparse
import time
from pathlib import Path

from aegisrl.agents.dqn import DQNAgent
from aegisrl.environment.env import HospitalSessionEnv
from aegisrl.evaluation.eval import evaluate, print_report


def main():
    parser = argparse.ArgumentParser(description="Train Dueling Double DQN on the hospital environment.")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--out", type=Path, default=Path("models/dqn.pt"))
    args = parser.parse_args()

    agent = DQNAgent(gamma=args.gamma, lr=args.lr, seed=args.seed, device=args.device)
    print(f"training on {agent.device}")

    start = time.time()
    agent.train(HospitalSessionEnv(), args.steps, log_every=50_000)
    print(f"trained in {time.time() - start:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.out)
    print(f"saved to {args.out}")

    print_report(agent.name, evaluate(agent))


if __name__ == "__main__":
    main()