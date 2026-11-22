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


def average_length(policy, actor_cls, episodes=300):
    env = HospitalSessionEnv(actor_mix={actor_cls: 1.0})
    lengths = []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        steps, done = 0, False
        while not done:
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            steps += 1
            done = terminated or truncated
        lengths.append(steps)
    return float(np.mean(lengths))


def test_challenge_ends_stolen_credential_sessions_faster_than_deny():
    assert average_length(lambda o: CHALLENGE, StolenCredential) < average_length(lambda o: DENY, StolenCredential)


@pytest.mark.parametrize("actor_cls", [StolenCredential, Scraper, AdaptiveAttacker])
def test_some_attackers_keep_going_after_failed_mfa(actor_cls):
    """If attackers always quit after failing MFA, 'still here after a failure' would mean
    'legitimate' - a shortcut the agent would learn. Some must persist."""
    env = HospitalSessionEnv(actor_mix={actor_cls: 1.0})
    persisted = 0
    for seed in range(400):
        env.reset(seed=seed)
        _, _, terminated, truncated, info = env.step(CHALLENGE)
        if not info["challenge_passed"] and not (terminated or truncated):
            persisted += 1
    assert 0.15 * 400 < persisted < 0.45 * 400

def test_some_insiders_fake_emergencies():
    """If insiders never claimed emergencies, 'emergency on a clean network' would always be
    genuine - a shortcut the agent would learn. A minority must try it."""
    from aegisrl.environment.features import RESTRICT
    env = HospitalSessionEnv(actor_mix={Insider: 1.0})
    spoofed = 0
    for seed in range(400):
        env.reset(seed=seed)
        flagged, done = False, False
        while not done:
            flagged = flagged or env.request.emergency_flag
            _, _, terminated, truncated, _ = env.step(RESTRICT)
            done = terminated or truncated
        spoofed += flagged
    assert 0.15 * 400 < spoofed < 0.35 * 400