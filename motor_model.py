"""Two rotary joints driven by a delayed, torque-limited actuator."""

from dataclasses import dataclass
from math import exp, expm1, floor, isfinite, radians


@dataclass(frozen=True)
class MotorAxis:
    bare_inertia_kg_m2: float = 0.030
    payload_radius_m: float = 0.105
    damping_nm_s_rad: float = 0.16
    torque_nm: float = 1.6
    delay_s: float = 0.05
    response_s: float = 0.025
    max_angle_deg: float = 26.0
    max_rate_deg_s: float = 95.0

    def __post_init__(self):
        values = tuple(vars(self).values())
        if (not all(isfinite(x) and x >= 0 for x in values)
                or min(self.bare_inertia_kg_m2, self.torque_nm, self.max_rate_deg_s) <= 0
                or not 0 < self.max_angle_deg <= 89.9 or self.delay_s > 10):
            raise ValueError("모터 관성, 토크, 회전각·속도 한계를 확인해 주세요.")

    def inertia(self, payload_kg: float) -> float:
        return self.bare_inertia_kg_m2 + payload_kg * self.payload_radius_m ** 2


@dataclass(frozen=True)
class MotorParams:
    payload_kg: float = 1.2
    roll: MotorAxis = MotorAxis()
    pitch: MotorAxis = MotorAxis(bare_inertia_kg_m2=0.035, payload_radius_m=0.125,
                                 damping_nm_s_rad=0.19, torque_nm=1.8)

    def __post_init__(self):
        if not isfinite(self.payload_kg) or self.payload_kg < 0:
            raise ValueError("적재물 질량은 유한한 0 이상의 값이어야 합니다.")


class RotaryJoint:
    """J*(q_ddot + projected base acceleration) + b*q_dot = applied torque."""

    __slots__ = ("dt", "angle_limit", "rate_limit", "inertia", "velocity_decay",
                 "velocity_gain", "actuator_mix", "torque_limit", "delay_samples", "delay_fraction",
                 "queue", "next_slot", "angle", "rate", "torque",
                 "angle_hit", "rate_hit")

    def __init__(self, spec: MotorAxis, payload_kg: float, dt: float):
        self.dt = dt
        self.inertia = spec.inertia(payload_kg)
        self.torque_limit = spec.torque_nm
        self.angle_limit, self.rate_limit = radians(spec.max_angle_deg), radians(spec.max_rate_deg_s)
        rate = spec.damping_nm_s_rad / self.inertia
        self.velocity_decay = exp(-rate * dt)
        self.velocity_gain = -expm1(-rate * dt) / rate if rate > 0 else dt
        self.actuator_mix = 1 - exp(-dt / spec.response_s) if spec.response_s > 0 else 1.0
        self.delay_samples = floor(spec.delay_s / dt)
        self.delay_fraction = spec.delay_s / dt - self.delay_samples
        self.queue = [0.0] * (self.delay_samples + 2)
        self.reset()

    def reset(self):
        self.next_slot = 0
        self.queue[:] = [0.0] * len(self.queue)
        self.angle = self.rate = self.torque = 0.0
        self.angle_hit = self.rate_hit = False

    def outward_stop(self) -> int:
        if self.angle >= self.angle_limit or self.rate >= self.rate_limit:
            return 1
        if self.angle <= -self.angle_limit or self.rate <= -self.rate_limit:
            return -1
        return 0

    def advance(self, command_nm: float, base_acc_rad_s2: float):
        slot = self.next_slot
        queue = self.queue
        queue[slot] = max(-self.torque_limit, min(self.torque_limit, command_nm))
        i = slot - self.delay_samples
        delayed = queue[i] + self.delay_fraction * (queue[i - 1] - queue[i])
        self.next_slot = (slot + 1) % len(queue)
        self.torque += self.actuator_mix * (delayed - self.torque)
        drive = self.torque / self.inertia - base_acc_rad_s2
        requested_rate = self.velocity_decay * self.rate + self.velocity_gain * drive
        self.rate_hit = abs(requested_rate) > self.rate_limit
        new_rate = max(-self.rate_limit, min(self.rate_limit, requested_rate))
        requested_angle = self.angle + 0.5 * (self.rate + new_rate) * self.dt
        self.angle_hit = abs(requested_angle) > self.angle_limit
        self.angle = max(-self.angle_limit, min(self.angle_limit, requested_angle))
        self.rate = 0.0 if self.angle_hit and new_rate * requested_angle > 0 else new_rate
