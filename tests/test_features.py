import numpy as np
import pytest

from aegisrl.environment.features import (
    FEATURE_NAMES, OBS_DIM, Request, SessionStats, to_obs,
)


def make_request(**overrides):
    """A normal, valid request. Pass keyword arguments to change any field."""
    base = dict(role=0, care_team=0.9, sensitivity=0.3, vip=False, hour=14.0,
                shift_deviation=0.1, geo_risk=0.1, device_trust=0.9,
                new_device=False, rate=3.0, gap_s=40.0, anomaly=0.1,
                emergency_flag=False, patient_in_ed=False, patient_id=1)
    base.update(overrides)
    return Request(**base)


def test_shape_and_dtype():
    obs = to_obs(make_request(), SessionStats())
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_features_land_in_the_right_place():
    obs = dict(zip(FEATURE_NAMES, to_obs(make_request(role=1, hour=23.0), SessionStats())))
    assert obs["role_nurse"] == 1.0 and obs["role_doctor"] == 0.0
    assert obs["hour_sin"] == pytest.approx(-0.259, abs=1e-3)
    assert obs["hour_cos"] == pytest.approx(0.966, abs=1e-3)


@pytest.mark.parametrize("field, value", [
    ("sensitivity", float("nan")),
    ("geo_risk", 1.5),
    ("role", 7),
    ("rate", -1.0),
    ("rate", float("inf")),
    ("hour", 25.0),
])
def test_bad_input_is_rejected(field, value):
    with pytest.raises(ValueError):
        to_obs(make_request(**{field: value}), SessionStats())


def test_extreme_values_are_capped():
    stats = SessionStats(requests=1000, denials=1000)
    obs = to_obs(make_request(rate=10_000.0, gap_s=99_999.0), stats)
    assert obs.min() >= -1.0 and obs.max() <= 1.0   