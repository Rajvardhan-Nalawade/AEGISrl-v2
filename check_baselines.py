import numpy as np

from aegisrl.environment.env import HospitalSessionEnv
from aegisrl.environment.features import ACTIONS

env = HospitalSessionEnv()

for action in range(4):
    returns = []
    for seed in range(2000):
        env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    print(f"always {ACTIONS[action]:<9} {np.mean(returns):7.2f}")