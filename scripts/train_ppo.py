import argparse
import time
from pathlib import Path

from aegisrl.agents.ppo import PPOAgent
from aegisrl.environment.vector import VectorEnv
from aegisrl.evaluation.evaluate import evaluate, print_report


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the hospital environment.")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--out", type=Path, default=Path("models/ppo.pt"))
    args = parser.parse_args()

    agent = PPOAgent(gamma=args.gamma, lr=args.lr, seed=args.seed, device=args.device)
    print(f"training on {agent.device} with {args.envs} environments")

    start = time.time()
    agent.train(VectorEnv(args.envs, seed=args.seed), args.steps, log_every=50)
    print(f"trained in {time.time() - start:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.out)
    print(f"saved to {args.out}")

    print_report(agent.name, evaluate(agent))


if __name__ == "__main__":
    main()