"""Saturated position PID with a filtered derivative and anti-windup."""

from dataclasses import dataclass
from math import exp, isfinite


@dataclass(frozen=True)
class Gains:
    kp: float = 1.2
    ki: float = 0.30
    kd: float = 0.20

    def __post_init__(self):
        if not all(isfinite(x) and x >= 0 for x in (self.kp, self.ki, self.kd)):
            raise ValueError("PID 이득은 유한한 0 이상의 값이어야 합니다.")


@dataclass(frozen=True)
class TwoAxisPID:
    roll: Gains = Gains()
    pitch: Gains = Gains(1.35, 0.35, 0.22)

    @classmethod
    def shared(cls, gains: Gains) -> "TwoAxisPID":
        return cls(gains, gains)


@dataclass(frozen=True)
class PIDSettings:
    derivative_tau_s: float = 0.08
    integral_limit_fraction: float = 0.7

    def __post_init__(self):
        if (not isfinite(self.derivative_tau_s) or self.derivative_tau_s < 0
                or not 0 <= self.integral_limit_fraction <= 1):
            raise ValueError("미분 필터 시간상수와 적분 출력 한계를 확인해 주세요.")


class PID:
    """Angle error in rad, torque in N·m. Integral state is stored as torque."""

    __slots__ = ("kp", "ki_dt", "kd", "dt", "mix", "limit", "i_limit", "integral",
                 "derivative", "last_error")

    def __init__(self, gains: Gains, dt: float, limit: float, settings: PIDSettings):
        self.kp, self.ki_dt, self.kd = gains.kp, gains.ki * dt, gains.kd
        self.dt = dt
        self.limit, self.i_limit = limit, limit * settings.integral_limit_fraction
        tau = settings.derivative_tau_s
        self.mix = 1 - exp(-dt / tau) if tau > 0 else 1.0
        self.reset()

    def reset(self):
        self.integral = self.derivative = 0.0
        self.last_error = None

    def step(self, error: float, outward_stop: int = 0) -> tuple[float, bool]:
        change = 0.0
        if self.last_error is not None:
            rate = (error - self.last_error) / self.dt
            self.derivative += self.mix * (rate - self.derivative)
            change = self.ki_dt * error
        self.last_error = error
        pd_torque = self.kp * error + self.kd * self.derivative
        candidate = max(-self.i_limit, min(self.i_limit, self.integral + change))
        preview = pd_torque + candidate
        # Disallow growth in the direction of an active output or mechanical limit.
        if not ((preview > self.limit and change > 0)
                or (preview < -self.limit and change < 0) or outward_stop * change > 0):
            self.integral = candidate
        raw = pd_torque + self.integral
        return max(-self.limit, min(self.limit, raw)), abs(raw) > self.limit
