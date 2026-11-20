from aegisrl.agents.dqn import DQNAgent
from aegisrl.agents.ppo import PPOAgent
from aegisrl.agents.rules import RBACPolicy, RiskRulesPolicy
from aegisrl.agents.tabular import TabularQAgent
from aegisrl.evaluation.evaluate import evaluate, print_security_report

tabular = TabularQAgent()
tabular.load("models/tabular_q.npy")
dqn = DQNAgent(device="cpu")
dqn.load("models/dqn.pt")
ppo = PPOAgent(device="cpu")
ppo.load("models/ppo.pt")

for policy in [RBACPolicy(), RiskRulesPolicy(), tabular, dqn, ppo]:
    print_security_report(policy.name, evaluate(policy))