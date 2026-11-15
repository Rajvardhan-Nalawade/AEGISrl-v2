import math
from dataclasses import dataclass, field

import numpy as np

# Actions the agent can take
ALLOW, DENY, CHALLENGE, RESTRICT = 0, 1, 2, 3
ACTIONS = ["ALLOW", "DENY", "CHALLENGE", "RESTRICT"]

# User roles
ROLES = ["DOCTOR", "NURSE", "STAFF", "RESEARCHER"]

FEATURE_NAMES = [
    "role_doctor", "role_nurse", "role_staff", "role_researcher",
    "care_team", "sensitivity", "vip",
    "hour_sin", "hour_cos", "shift_deviation",
    "geo_risk", "device_trust", "new_device",
    "log_rate", "gap", "anomaly",
    "emergency_flag", "patient_in_ed",
    "sess_requests", "sess_distinct_patients", "sess_mean_sensitivity",
    "sess_challenges_passed", "sess_challenges_failed",
    "sess_denials", "sess_restricts",
]
OBS_DIM = len(FEATURE_NAMES)   # 25

# Scaling constants: the value that maps to 1.0 ("this much counts as a lot")
MAX_RATE = 60.0             # requests per minute
MAX_GAP_S = 300.0           # seconds
MAX_SESSION_REQUESTS = 30
MAX_SESSION_PATIENTS = 30
MAX_CHALLENGES = 3
MAX_DENIALS = 5
MAX_RESTRICTS = 5

# Fields that must already be risk scores in [0, 1]
UNIT_FIELDS = ("care_team", "sensitivity", "shift_deviation",
               "geo_risk", "device_trust", "anomaly")


@dataclass
class Request:
    """One access request: the raw facts the hospital's systems would log."""
    role: int
    care_team: float
    sensitivity: float
    vip: bool
    hour: float
    shift_deviation: float
    geo_risk: float
    device_trust: float
    new_device: bool
    rate: float
    gap_s: float
    anomaly: float
    emergency_flag: bool
    patient_in_ed: bool
    patient_id: int


@dataclass
class SessionStats:
    """Running memory of the current session. Updated after every decision."""
    requests: int = 0
    patients: set = field(default_factory=set)
    sens_sum: float = 0.0
    ch_passed: int = 0
    ch_failed: int = 0
    denials: int = 0
    restricts: int = 0

    def update(self, req: Request, action: int, challenge_passed: bool):
        self.requests += 1
        self.patients.add(req.patient_id)
        self.sens_sum += req.sensitivity

        if action == CHALLENGE:
            if challenge_passed:
                self.ch_passed += 1
            else:
                self.ch_failed += 1
        elif action == DENY:
            self.denials += 1
        elif action == RESTRICT:
            self.restricts += 1


def validate(req: Request) -> None:
    """Raise ValueError if the request contains impossible or missing values."""
    if not 0 <= req.role < len(ROLES):
        raise ValueError(f"role must be 0-{len(ROLES) - 1}, got {req.role}")
    for name in UNIT_FIELDS:
        v = getattr(req, name)
        if not (math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    if not (math.isfinite(req.hour) and 0.0 <= req.hour <= 24.0):
        raise ValueError(f"hour must be in [0, 24], got {req.hour}")
    for name in ("rate", "gap_s"):
        v = getattr(req, name)
        if not (math.isfinite(v) and v >= 0.0):
            raise ValueError(f"{name} must be a non-negative number, got {v}")


def _capped(value: float, maximum: float) -> float:
    return min(value / maximum, 1.0)


def to_obs(req: Request, s: SessionStats) -> np.ndarray:
    """Convert a raw Request + session memory into the vector the agent sees."""
    validate(req)

    angle = 2 * np.pi * req.hour / 24
    mean_sens = s.sens_sum / s.requests if s.requests else 0.0

    f = {f"role_{r.lower()}": float(req.role == i) for i, r in enumerate(ROLES)}
    f.update({
        "care_team": req.care_team,
        "sensitivity": req.sensitivity,
        "vip": float(req.vip),
        "hour_sin": np.sin(angle),
        "hour_cos": np.cos(angle),
        "shift_deviation": req.shift_deviation,
        "geo_risk": req.geo_risk,
        "device_trust": req.device_trust,
        "new_device": float(req.new_device),
        "log_rate": min(np.log1p(req.rate) / np.log1p(MAX_RATE), 1.0),
        "gap": _capped(req.gap_s, MAX_GAP_S),
        "anomaly": req.anomaly,
        "emergency_flag": float(req.emergency_flag),
        "patient_in_ed": float(req.patient_in_ed),
        "sess_requests": _capped(s.requests, MAX_SESSION_REQUESTS),
        "sess_distinct_patients": _capped(len(s.patients), MAX_SESSION_PATIENTS),
        "sess_mean_sensitivity": mean_sens,
        "sess_challenges_passed": _capped(s.ch_passed, MAX_CHALLENGES),
        "sess_challenges_failed": _capped(s.ch_failed, MAX_CHALLENGES),
        "sess_denials": _capped(s.denials, MAX_DENIALS),
        "sess_restricts": _capped(s.restricts, MAX_RESTRICTS),
    })

    if f.keys() != set(FEATURE_NAMES):
        missing = set(FEATURE_NAMES) - f.keys()
        extra = f.keys() - set(FEATURE_NAMES)
        raise RuntimeError(f"feature mismatch: missing={missing}, extra={extra}")

    return np.array([f[name] for name in FEATURE_NAMES], dtype=np.float32)