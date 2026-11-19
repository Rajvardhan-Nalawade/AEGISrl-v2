import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from aegisrl.environment.actors import (
    AdaptiveAttacker, EmergencyClinician, Insider,
    LegitClinician, Scraper, StolenCredential,
)
from aegisrl.environment.env import HospitalSessionEnv
from aegisrl.environment.features import ALLOW, CHALLENGE, DENY


def run_policy(policy, actor_cls, episodes=300):
    """Average return of a fixed policy against a single actor type."""
    env = HospitalSessionEnv(actor_mix={actor_cls: 1.0})
    returns = []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def test_passes_gymnasium_checker():
    check_env(HospitalSessionEnv(), skip_render_check=True)


def test_same_seed_same_episode():
    a, b = HospitalSessionEnv(), HospitalSessionEnv()
    obs_a, _ = a.reset(seed=123)
    obs_b, _ = b.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_b)
    for _ in range(5):
        step_a, step_b = a.step(CHALLENGE), b.step(CHALLENGE)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1] == step_b[1]


def test_sessions_never_exceed_max_steps():
    env = HospitalSessionEnv(max_steps=30)
    for seed in range(200):
        env.reset(seed=seed)
        steps, done = 0, False
        while not done:
            _, _, terminated, truncated, _ = env.step(ALLOW)
            steps += 1
            done = terminated or truncated
        assert steps <= 30


@pytest.mark.parametrize("actor_cls", [StolenCredential, Scraper, AdaptiveAttacker, Insider])
def test_allowing_attackers_is_costly(actor_cls):
    assert run_policy(lambda o: ALLOW, actor_cls) < -5


@pytest.mark.parametrize("actor_cls", [LegitClinician, EmergencyClinician])
def test_denying_clinicians_is_costly(actor_cls):
    assert run_policy(lambda o: DENY, actor_cls) < run_policy(lambda o: ALLOW, actor_cls)


def test_challenge_fails_against_insiders():
    assert run_policy(lambda o: CHALLENGE, Insider) < -5


def test_challenge_stops_stolen_credentials_fast():
    assert run_policy(lambda o: CHALLENGE, StolenCredential) > run_policy(lambda o: DENY, StolenCredential)