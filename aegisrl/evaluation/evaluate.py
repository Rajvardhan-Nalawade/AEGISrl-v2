import time

import numpy as np

from aegisrl.environment.env import HospitalSessionEnv
from aegisrl.environment.features import ALLOW, CHALLENGE, DENY, RESTRICT

EVAL_SEED_START = 1_000_000   # far away from training seeds (0, 1, 2, ...)


def evaluate(policy, episodes: int = 3000, env: HospitalSessionEnv | None = None) -> dict:
    """Run a policy on unseen sessions. Returns rewards plus security and care metrics."""
    env = env or HospitalSessionEnv()
    by_actor: dict[str, list[float]] = {}
    c = dict(attack_sessions=0, attack_sessions_clean=0, leaked=0.0,
             legit_requests=0, legit_granted=0, legit_blocked=0, legit_challenged=0,
             emergency_requests=0, emergency_blocked=0)
    decision_time, decisions = 0.0, 0

    for i in range(episodes):
        obs, info = env.reset(seed=EVAL_SEED_START + i)
        total, done, session_leaked = 0.0, False, 0.0
        malicious = env.actor.malicious
        emergency = env.actor.name == "EMERGENCY"

        while not done:
            start = time.perf_counter()
            action = policy.act(obs)
            decision_time += time.perf_counter() - start
            decisions += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated

            granted = action == ALLOW or (action == CHALLENGE and info["challenge_passed"])
            blocked = action == DENY or (action == CHALLENGE and not info["challenge_passed"])
            if malicious:
                if granted:
                    session_leaked += info["sensitivity"]
                elif action == RESTRICT:
                    session_leaked += 0.3 * info["sensitivity"]
            else:
                c["legit_requests"] += 1
                c["legit_granted"] += granted
                c["legit_blocked"] += blocked
                c["legit_challenged"] += action == CHALLENGE
                if emergency:
                    c["emergency_requests"] += 1
                    c["emergency_blocked"] += blocked

        by_actor.setdefault(info["actor"], []).append(total)
        if malicious:
            c["attack_sessions"] += 1
            c["attack_sessions_clean"] += session_leaked == 0.0
            c["leaked"] += session_leaked

    all_returns = [r for returns in by_actor.values() for r in returns]
    return {
        "overall": float(np.mean(all_returns)),
        "per_actor": {name: float(np.mean(r)) for name, r in sorted(by_actor.items())},
        "attacks_fully_stopped": c["attack_sessions_clean"] / max(c["attack_sessions"], 1),
        "leak_per_attack": c["leaked"] / max(c["attack_sessions"], 1),
        "legit_served": c["legit_granted"] / max(c["legit_requests"], 1),
        "false_denial_rate": c["legit_blocked"] / max(c["legit_requests"], 1),
        "emergency_denial_rate": c["emergency_blocked"] / max(c["emergency_requests"], 1),
        "legit_challenge_rate": c["legit_challenged"] / max(c["legit_requests"], 1),
        "decision_ms": 1000 * decision_time / max(decisions, 1),
    }


def print_report(name: str, result: dict) -> None:
    per_actor = "  ".join(f"{k}={v:6.1f}" for k, v in result["per_actor"].items())
    print(f"{name:<12} overall={result['overall']:6.2f}  {per_actor}")


def print_security_report(name: str, r: dict) -> None:
    print(f"{name:<12} stopped={r['attacks_fully_stopped']:6.1%}  leak/attack={r['leak_per_attack']:5.2f}  "
          f"served={r['legit_served']:6.1%}  false_denial={r['false_denial_rate']:5.1%}  "
          f"emergency_denial={r['emergency_denial_rate']:5.1%}  challenged={r['legit_challenge_rate']:5.1%}  "
          f"{r['decision_ms']:.3f} ms")