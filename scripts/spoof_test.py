import argparse
from dataclasses import replace

from aegisrl.agents.dqn import DQNAgent
from aegisrl.agents.ppo import PPOAgent
from aegisrl.agents.rules import RBACPolicy, RiskRulesPolicy
from aegisrl.agents.tabular import TabularQAgent
from aegisrl.environment.actors import AdaptiveAttacker, EmergencyClinician, Insider, StolenCredential
from aegisrl.environment.env import HospitalSessionEnv
from aegisrl.environment.features import ALLOW, CHALLENGE, RESTRICT


class OutsiderClaimsEmergency(StolenCredential):
    """Stolen credentials that claim an emergency from the very first request."""
    name = "SPOOF_OUTSIDER"

    def next_request(self):
        return replace(super().next_request(), emergency_flag=True)


class InsiderClaimsEmergency(Insider):
    """A snooping employee who ticks 'emergency' to get past the controls."""
    name = "SPOOF_INSIDER"

    def next_request(self):
        return replace(super().next_request(), emergency_flag=True)


class InsiderClaimsEmergencyOnEdPatient(Insider):
    """Worst case: the snooped patient really is in the emergency department."""
    name = "SPOOF_INSIDER_ED"

    def next_request(self):
        return replace(super().next_request(), emergency_flag=True, patient_in_ed=True)


class AdaptiveClaimsEmergency(AdaptiveAttacker):
    """Held out: a disguised attacker that claims an emergency while still blending in."""
    name = "SPOOF_ADAPTIVE"

    def next_request(self):
        return replace(super().next_request(), emergency_flag=True)


# (actor, in training?)  Held-out tactics test generalisation, not memorisation.
SCENARIOS = {
    "real emergency": (EmergencyClinician, "trained"),
    "outsider claims emergency": (OutsiderClaimsEmergency, "trained"),
    "insider claims emergency": (InsiderClaimsEmergency, "trained"),
    "disguised attacker claims emergency": (AdaptiveClaimsEmergency, "HELD OUT"),
    "insider + patient really in ED": (InsiderClaimsEmergencyOnEdPatient, "HELD OUT"),
}


def run(policy, actor_cls, sessions: int) -> dict:
    """Outcome of every emergency-flagged request in `sessions` sessions."""
    env = HospitalSessionEnv(actor_mix={actor_cls: 1.0})
    granted = restricted = blocked = 0
    for i in range(sessions):
        obs, _ = env.reset(seed=5_000_000 + i)
        done = False
        while not done:
            flagged = env.request.emergency_flag
            action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if not flagged:
                continue
            if action == ALLOW or (action == CHALLENGE and info["challenge_passed"]):
                granted += 1
            elif action == RESTRICT:
                restricted += 1
            else:
                blocked += 1
    total = max(granted + restricted + blocked, 1)
    return {"n": total, "granted": granted / total, "restricted": restricted / total,
            "blocked": blocked / total}


def main():
    parser = argparse.ArgumentParser(description="How do agents handle real vs. spoofed emergencies?")
    parser.add_argument("--sessions", type=int, default=500)
    args = parser.parse_args()

    tabular = TabularQAgent()
    tabular.load("models/tabular_q.npy")
    dqn = DQNAgent(device="cpu")
    dqn.load("models/dqn.pt")
    ppo = PPOAgent(device="cpu")
    ppo.load("models/ppo.pt")
    policies = [RBACPolicy(), RiskRulesPolicy(), tabular, dqn, ppo]

    for scenario, (actor_cls, status) in SCENARIOS.items():
        goal = "should be GRANTED" if actor_cls is EmergencyClinician else "should NOT be granted"
        print(f"\n=== {scenario} ({goal}; {status}) ===")
        for policy in policies:
            r = run(policy, actor_cls, args.sessions)
            print(f"  {policy.name:<11} granted {r['granted']:6.1%}   restricted {r['restricted']:6.1%}"
                  f"   blocked {r['blocked']:6.1%}   (n={r['n']})")


if __name__ == "__main__":
    main()