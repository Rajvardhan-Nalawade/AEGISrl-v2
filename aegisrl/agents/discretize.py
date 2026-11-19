import math

import numpy as np

from .base import feature

# Buckets per category: role, care-team match, sensitivity, off-hours, location, frequency, emergency
STATE_SHAPE = (4, 2, 4, 2, 3, 3, 2)
NUM_STATES = int(np.prod(STATE_SHAPE))   # 1152
MAX_RATE = 60.0


def hour_from_obs(obs: np.ndarray) -> float:
    """Recover clock hour (0-24) from the sin/cos encoding."""
    angle = math.atan2(feature(obs, "hour_sin"), feature(obs, "hour_cos"))
    return (angle / (2 * math.pi) * 24) % 24


def rate_from_obs(obs: np.ndarray) -> float:
    """Undo the log scaling to get requests per minute."""
    return math.expm1(feature(obs, "log_rate") * math.log1p(MAX_RATE))


def to_state(obs: np.ndarray) -> tuple:
    """Squash the 25 continuous features into 7 categories for a lookup table."""
    roles = [feature(obs, f"role_{r}") for r in ("doctor", "nurse", "staff", "researcher")]
    role = int(np.argmax(roles))

    dept_match = int(feature(obs, "care_team") >= 0.5)

    sens = feature(obs, "sensitivity")
    if feature(obs, "vip") == 1.0:
        sensitivity = 3
    else:
        sensitivity = 0 if sens < 0.33 else (1 if sens < 0.66 else 2)

    hour = hour_from_obs(obs)
    off_hours = int(hour < 7 or hour >= 19)

    geo = feature(obs, "geo_risk")
    location = 0 if geo < 0.3 else (1 if geo < 0.6 else 2)

    rate = rate_from_obs(obs)
    frequency = 0 if rate <= 5 else (1 if rate <= 20 else 2)

    emergency = int(feature(obs, "emergency_flag") == 1.0)

    return (role, dept_match, sensitivity, off_hours, location, frequency, emergency)


def to_index(obs: np.ndarray) -> int:
    """Turn the 7 categories into one row number (0-1151)."""
    return int(np.ravel_multi_index(to_state(obs), STATE_SHAPE))