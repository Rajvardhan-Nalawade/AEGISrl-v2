import numpy as np

from aegisrl.environment.features import ALLOW, CHALLENGE, DENY, RESTRICT
from .base import Policy, feature


class RBACPolicy(Policy):
    """Classic role-based access control with a break-glass emergency override."""

    name = "RBAC"

    def act(self, obs: np.ndarray) -> int:
        if feature(obs, "emergency_flag") == 1.0:
            return ALLOW
        if feature(obs, "role_researcher") == 1.0:
            return DENY
        if feature(obs, "role_staff") == 1.0:
            return RESTRICT
        return ALLOW


class RiskRulesPolicy(Policy):
    """Hand-written risk rules, the way a security engineer would configure them."""

    name = "RISK_RULES"

    def act(self, obs: np.ndarray) -> int:
        geo = feature(obs, "geo_risk")
        device = feature(obs, "device_trust")
        rate = feature(obs, "log_rate")
        care = feature(obs, "care_team")
        sens = feature(obs, "sensitivity")
        emergency = feature(obs, "emergency_flag") == 1.0
        in_ed = feature(obs, "patient_in_ed") == 1.0

        if emergency and in_ed and geo < 0.4:
            return ALLOW
        if geo > 0.5 or device < 0.5 or rate > 0.75:
            return CHALLENGE
        if care < 0.4 and sens > 0.6:
            return RESTRICT
        return ALLOW