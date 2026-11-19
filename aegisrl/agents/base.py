import numpy as np

from aegisrl.environment.features import FEATURE_NAMES

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def feature(obs: np.ndarray, name: str) -> float:
    """Read one named feature from an observation vector."""
    return float(obs[FEATURE_INDEX[name]])


class Policy:
    """Anything that maps an observation to an action."""

    name = "BASE"

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError