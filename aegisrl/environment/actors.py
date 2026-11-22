import numpy as np
from dataclasses import replace
from .features import ALLOW, CHALLENGE, DENY, RESTRICT, Request

MFA_GIVE_UP_PROB = 0.7
MAX_MFA_FAILURES = 2

INSIDER_SPOOF_PROB = 0.25
OUTSIDER_EARLY_SPOOF_PROB = 0.20

class Actor:
    """Base class for whoever is behind a session. Subclasses define the behaviour."""
    mfa_failures = 0

    name = "BASE"            # label used in logs and evaluation
    malicious = False        # hidden ground truth: the agent never sees this
    mfa_pass_prob = 0.95     # chance of passing a CHALLENGE (step-up MFA)

    def __init__(self, rng: np.random.Generator, noise: float = 1.0):
        self.rng = rng
        self.noise = noise
        self.done = False
        self.patient_id = int(rng.integers(1_000_000))

    def gives_up_after_mfa_failure(self) -> bool:
        """Record a failed MFA challenge and decide whether the attacker abandons the session."""
        self.mfa_failures += 1
        return self.mfa_failures >= MAX_MFA_FAILURES or self.rng.random() < MFA_GIVE_UP_PROB

    def signal(self, mean: float, sd: float) -> float:
        """A noisy risk score clipped to [0, 1]."""
        return float(np.clip(self.rng.normal(mean, sd * self.noise), 0.0, 1.0))

    def next_request(self) -> Request:
        raise NotImplementedError

    def react(self, action: int, challenge_passed: bool) -> None:
        raise NotImplementedError


class LegitClinician(Actor):
    """A doctor, nurse or staff member doing routine work on their own patients."""

    name = "NORMAL"

    def __init__(self, rng, noise=1.0):
        super().__init__(rng, noise)
        self.role = int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))
        self.tasks_left = int(rng.integers(3, 15))
        self.device_trust = self.signal(0.85, 0.10)
        self.hour = float(rng.normal(13, 3)) % 24
        self.failed_attempts = 0

    def next_request(self) -> Request:
        if self.rng.random() < 0.3:
            self.patient_id = int(self.rng.integers(1_000_000))
        self.hour = (self.hour + self.rng.exponential(0.05)) % 24
        return Request(
            role=self.role,
            care_team=self.signal(0.85, 0.15),
            sensitivity=self.signal(0.40, 0.20),
            vip=self.rng.random() < 0.03,
            hour=self.hour,
            shift_deviation=self.signal(0.10, 0.10),
            geo_risk=self.signal(0.10, 0.08),
            device_trust=self.device_trust,
            new_device=self.rng.random() < 0.05,
            rate=float(self.rng.gamma(2.0, 1.5)),
            gap_s=float(self.rng.exponential(40)),
            anomaly=self.signal(0.15, 0.12),
            emergency_flag=False,
            patient_in_ed=self.rng.random() < 0.05,
            patient_id=self.patient_id,
        )

    def react(self, action: int, challenge_passed: bool) -> None:
        blocked = action == DENY or (action == CHALLENGE and not challenge_passed)
        if blocked:
            self.failed_attempts += 1
            if self.failed_attempts >= 2:
                self.done = True
            return
        self.tasks_left -= 1
        if self.tasks_left <= 0:
            self.done = True

class StolenCredential(Actor):
    """An outsider using a clinician's stolen password from their own machine."""

    name = "STOLEN_CREDENTIAL"
    malicious = True
    mfa_pass_prob = 0.05

    def __init__(self, rng, noise=1.0):
        super().__init__(rng, noise)
        self.role = int(rng.choice([0, 1, 2], p=[0.50, 0.30, 0.20]))
        self.target_sensitivity = 0.5
        self.denials = 0
        self.geo_risk = self.signal(0.75, 0.15)
        self.device_trust = self.signal(0.25, 0.15)
        self.hour = float(rng.normal(2, 3)) % 24
        self.bold = rng.random() < OUTSIDER_EARLY_SPOOF_PROB

    def next_request(self) -> Request:
        self.patient_id = int(self.rng.integers(1_000_000))
        self.hour = (self.hour + self.rng.exponential(0.02)) % 24
        pushed_back = (self.denials + self.mfa_failures) > 0
        spoof_emergency = (self.bold or pushed_back) and self.rng.random() < 0.3
        return Request(
            role=self.role,
            care_team=self.signal(0.20, 0.20),
            sensitivity=self.signal(self.target_sensitivity, 0.15),
            vip=self.rng.random() < 0.20,
            hour=self.hour,
            shift_deviation=self.signal(0.70, 0.20),
            geo_risk=self.geo_risk,
            device_trust=self.device_trust,
            new_device=self.rng.random() < 0.70,
            rate=float(self.rng.gamma(3.0, 2.0)),
            gap_s=float(self.rng.exponential(15)),
            anomaly=self.signal(0.60, 0.20),
            emergency_flag=spoof_emergency,
            patient_in_ed=self.rng.random() < 0.05,
            patient_id=self.patient_id,
        )

    def react(self, action: int, challenge_passed: bool) -> None:
        if action == CHALLENGE and not challenge_passed:
            if self.gives_up_after_mfa_failure():
                self.done = True
            else:
                self.geo_risk = self.signal(0.40, 0.15)   # retries through a VPN
        elif action == DENY:
            self.denials += 1
            self.geo_risk = self.signal(0.40, 0.15)
            if self.denials >= 3:
                self.done = True
        else:
            self.target_sensitivity = min(1.0, self.target_sensitivity + 0.15)

class EmergencyClinician(LegitClinician):
    """A doctor or nurse treating an emergency patient. Must never be blocked."""

    name = "EMERGENCY"

    def __init__(self, rng, noise=1.0):
        super().__init__(rng, noise)
        self.role = int(rng.choice([0, 1], p=[0.75, 0.25]))
        self.tasks_left = int(rng.integers(2, 8))
        self.hour = float(rng.uniform(0, 24))

    def next_request(self) -> Request:
        if self.rng.random() < 0.15:
            self.patient_id = int(self.rng.integers(1_000_000))
        self.hour = (self.hour + self.rng.exponential(0.01)) % 24
        return Request(
            role=self.role,
            care_team=self.signal(0.40, 0.25),
            sensitivity=self.signal(0.75, 0.15),
            vip=self.rng.random() < 0.05,
            hour=self.hour,
            shift_deviation=self.signal(0.35, 0.25),
            geo_risk=self.signal(0.12, 0.08),
            device_trust=self.device_trust,
            new_device=self.rng.random() < 0.08,
            rate=float(self.rng.gamma(3.0, 3.0)),
            gap_s=float(self.rng.exponential(10)),
            anomaly=self.signal(0.35, 0.18),
            emergency_flag=True,
            patient_in_ed=self.rng.random() < 0.90,
            patient_id=self.patient_id,
        )

class Insider(Actor):
    """A real employee, on their own account and device, snooping on records they have no reason to see."""

    name = "INSIDER"
    malicious = True
    mfa_pass_prob = 0.98

    def __init__(self, rng, noise=1.0):
        super().__init__(rng, noise)
        self.role = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.35, 0.30, 0.20]))
        self.curiosity = int(rng.integers(2, 9))
        self.device_trust = self.signal(0.85, 0.10)
        self.hour = float(rng.normal(13, 4)) % 24
        self.spoofer = rng.random() < INSIDER_SPOOF_PROB
        self.pushed_back = False

    def next_request(self) -> Request:
        if self.rng.random() < 0.6:
            self.patient_id = int(self.rng.integers(1_000_000))
        self.hour = (self.hour + self.rng.exponential(0.05)) % 24
        return Request(
            role=self.role,
            care_team=self.signal(0.10, 0.10),
            sensitivity=self.signal(0.80, 0.15),
            vip=self.rng.random() < 0.50,
            hour=self.hour,
            shift_deviation=self.signal(0.20, 0.15),
            geo_risk=self.signal(0.08, 0.06),
            device_trust=self.device_trust,
            new_device=self.rng.random() < 0.03,
            rate=float(self.rng.gamma(1.5, 1.2)),
            gap_s=float(self.rng.exponential(90)),
            anomaly=self.signal(0.30, 0.15),
            emergency_flag=self.spoofer and (self.pushed_back or self.rng.random() < 0.3),
            patient_in_ed=self.rng.random() < 0.05,
            patient_id=self.patient_id,
        )

    def react(self, action: int, challenge_passed: bool) -> None:
        if action in (DENY, RESTRICT):
            self.pushed_back = True
        if action == DENY and self.rng.random() < 0.6:
            self.done = True
            return
        self.curiosity -= 1
        if self.curiosity <= 0:
            self.done = True

class Scraper(Actor):
    """An automated script bulk-downloading records as fast as it can."""

    name = "SCRAPER"
    malicious = True
    mfa_pass_prob = 0.0

    def __init__(self, rng, noise=1.0):
        super().__init__(rng, noise)
        self.role = int(rng.choice([0, 2, 3], p=[0.20, 0.40, 0.40]))
        self.base_rate = float(rng.uniform(25, 80))
        self.denials = 0
        self.geo_risk = self.signal(0.55, 0.25)
        self.device_trust = self.signal(0.40, 0.20)
        self.hour = float(rng.normal(3, 5)) % 24

    def next_request(self) -> Request:
        self.patient_id = int(self.rng.integers(1_000_000))
        rate = self.base_rate * float(self.rng.gamma(10.0, 0.1))
        self.hour = (self.hour + 1 / max(rate, 1.0) / 60) % 24
        return Request(
            role=self.role,
            care_team=self.signal(0.05, 0.05),
            sensitivity=self.signal(0.45, 0.25),
            vip=self.rng.random() < 0.05,
            hour=self.hour,
            shift_deviation=self.signal(0.60, 0.25),
            geo_risk=self.geo_risk,
            device_trust=self.device_trust,
            new_device=self.rng.random() < 0.40,
            rate=rate,
            gap_s=float(self.rng.exponential(60 / rate)),
            anomaly=self.signal(0.70, 0.15),
            emergency_flag=False,
            patient_in_ed=self.rng.random() < 0.05,
            patient_id=self.patient_id,
        )

    def react(self, action: int, challenge_passed: bool) -> None:
        if action == CHALLENGE and not challenge_passed:
            if self.gives_up_after_mfa_failure():
                self.done = True
            else:
                self.base_rate = max(2.0, self.base_rate / 2)
        elif action == DENY:
            self.denials += 1
            self.base_rate = max(2.0, self.base_rate / 2)
            if self.denials >= 5:
                self.done = True

class AdaptiveAttacker(Actor):
    """A careful attacker on a compromised internal account: blends in first, then strikes."""

    name = "ADAPTIVE"
    malicious = True
    mfa_pass_prob = 0.30

    def __init__(self, rng, noise=1.0):
        super().__init__(rng, noise)
        self.cover = LegitClinician(rng, noise)
        self.role = self.cover.role
        self.warmup = int(rng.integers(2, 6))
        self.steps = 0
        self.denials = 0
        self.quiet = 0
        self.aggression = 0.0

    def next_request(self) -> Request:
        req = self.cover.next_request()
        self.steps += 1

        if self.steps <= self.warmup or self.quiet > 0:
            self.quiet = max(0, self.quiet - 1)
            return req

        self.aggression = min(1.0, self.aggression + 0.25)
        a = self.aggression
        self.patient_id = int(self.rng.integers(1_000_000))
        return replace(
            req,
            patient_id=self.patient_id,
            care_team=self.signal(0.85 - 0.60 * a, 0.15),
            sensitivity=self.signal(0.40 + 0.50 * a, 0.15),
            vip=self.rng.random() < 0.03 + 0.30 * a,
            anomaly=self.signal(0.15 + 0.35 * a, 0.15),
            emergency_flag=(self.denials + self.mfa_failures) > 0 and self.rng.random() < 0.5,
            patient_in_ed=self.rng.random() < 0.10,
        )

    def react(self, action: int, challenge_passed: bool) -> None:
        if action == CHALLENGE and not challenge_passed:
            if self.gives_up_after_mfa_failure():
                self.done = True
            else:
                self.quiet = 2                                   # lies low, then tries again
                self.aggression = max(0.0, self.aggression - 0.25)
        elif action == DENY:
            self.denials += 1
            self.quiet = 2
            self.aggression = max(0.0, self.aggression - 0.25)
            if self.denials >= 4:
                self.done = True