"""Performance statistics and a dimensionless Optuna objective."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from motor_model import MotorParams

if TYPE_CHECKING:
    from simulation import SimulationResult


@dataclass(frozen=True)
class ObjectiveWeights:
    roll: float = 0.37
    pitch: float = 0.37
    peak: float = 0.08
    effort: float = 0.08
    vibration: float = 0.05
    constraints: float = 0.20

    def __post_init__(self):
        weights = tuple(vars(self).values())
        if not all(isfinite(x) and x >= 0 for x in weights) or not any(weights):
            raise ValueError("목적함수 가중치는 0 이상이며 하나 이상 양수여야 합니다.")


@dataclass(frozen=True)
class AxisMetrics:
    rms_deg: float
    mae_deg: float
    peak_deg: float
    travel_deg: float
    rate_rms_deg_s: float
    rate_peak_deg_s: float
    torque_rms_nm: float
    torque_peak_nm: float
    acceleration_rms_deg_s2: float
    absolute_work_j: float
    torque_saturation_events: int
    torque_saturation_samples: int
    rate_limit_events: int
    rate_limit_samples: int
    angle_limit_events: int
    angle_limit_samples: int
    violation_samples: int


@dataclass(frozen=True)
class Metrics:
    roll: AxisMetrics
    pitch: AxisMetrics
    peak_tilt_deg: float
    within_one_deg_pct: float
    samples: int

    def to_dict(self):
        return asdict(self)


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def _counts(mask):
    starts = int(mask[0]) + int(np.count_nonzero(mask[1:] & ~mask[:-1]))
    return starts, int(np.count_nonzero(mask))


def measure(result: "SimulationResult", motor: MotorParams) -> Metrics:
    axes = []
    dt = result.motion.dt
    for index, spec in enumerate((motor.roll, motor.pitch)):
        error = result.plate_deg[:, index]
        joint = result.joint_deg[:, index]
        speed = result.rate_deg_s[:, index]
        torque = result.torque_nm[:, index]
        power = np.abs(torque * np.deg2rad(speed))
        violations = ((np.abs(joint) > spec.max_angle_deg + 1e-9)
                      | (np.abs(speed) > spec.max_rate_deg_s + 1e-9)
                      | (np.abs(torque) > spec.torque_nm + 1e-9))
        axes.append(AxisMetrics(
            _rms(error), float(np.mean(np.abs(error))), float(np.max(np.abs(error))),
            float(np.abs(np.diff(joint)).sum()), _rms(speed), float(np.max(np.abs(speed))),
            _rms(torque), float(np.max(np.abs(torque))), _rms(np.diff(speed) / dt),
            float(np.sum(power[:-1] + power[1:]) * 0.5 * dt),
            *_counts(result.command_limited[:, index]), *_counts(result.rate_limited[:, index]),
            *_counts(result.angle_limited[:, index]), int(np.count_nonzero(violations)),
        ))
    roll, pitch = np.deg2rad(result.plate_deg).T
    tilt = np.rad2deg(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1)))
    within = np.mean(np.all(np.abs(result.plate_deg) <= 1, axis=1)) * 100
    return Metrics(*axes, float(np.max(tilt)), float(within), len(tilt))


def objective_terms(metrics: Metrics, motor: MotorParams) -> dict[str, float]:
    effort = vibration = constraints = violations = 0.0
    for m, spec in zip((metrics.roll, metrics.pitch), (motor.roll, motor.pitch)):
        effort += (m.torque_rms_nm / spec.torque_nm + m.rate_rms_deg_s / spec.max_rate_deg_s) / 4
        vibration += m.acceleration_rms_deg_s2 / (spec.max_rate_deg_s / 0.25) / 2
        constraints += (m.torque_saturation_samples + m.rate_limit_samples +
                        2 * m.angle_limit_samples) / (2 * metrics.samples)
        violations += m.violation_samples
    return {
        "roll": metrics.roll.rms_deg / 1.0,
        "pitch": metrics.pitch.rms_deg / 1.0,
        "peak": metrics.peak_tilt_deg / 5.0,
        "effort": effort,
        "vibration": vibration,
        "constraints": constraints,
        "violations": violations,
    }


def objective_score(terms: dict[str, float], weights: ObjectiveWeights) -> float:
    if terms["violations"]:
        return 1e6 + 1e3 * terms["violations"]
    score = sum(weight * terms[name] for name, weight in vars(weights).items())
    return score if isfinite(score) else 1e8


def rms_reduction(before: float, after: float) -> float | None:
    return 100 * (1 - after / before) if before > 1e-12 else None
