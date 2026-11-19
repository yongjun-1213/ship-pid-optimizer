"""Deterministic batch simulation. This module has no UI or plotting dependency."""

from dataclasses import dataclass, replace
from math import cos, exp, isfinite, pi, radians, sin

import numpy as np
import pandas as pd

from data_loader import MotionInput
from evaluation import Metrics, measure
from motor_model import MotorParams, RotaryJoint
from pid_controller import PID, PIDSettings, TwoAxisPID
from ship_model import plate_angles


@dataclass(frozen=True)
class SensorParams:
    angle_noise_deg: float = 0.28
    gyro_noise_deg_s: float = 0.23
    gyro_bias_deg_s: float = 0.10
    fusion_tau_s: float = 0.40
    seed: int = 7

    def __post_init__(self):
        if (not all(isfinite(x) and x >= 0 for x in (
                self.angle_noise_deg, self.gyro_noise_deg_s, self.gyro_bias_deg_s, self.fusion_tau_s))
                or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32):
            raise ValueError("IMU 잡음, 상보필터 시간상수 또는 난수 시드를 확인해 주세요.")


@dataclass(frozen=True)
class PreparedSimulation:
    motion: MotionInput
    schedule: tuple
    sensor: SensorParams


@dataclass(frozen=True)
class SimulationResult:
    motion: MotionInput
    controller_on: bool
    plate_deg: np.ndarray
    estimate_deg: np.ndarray
    joint_deg: np.ndarray
    rate_deg_s: np.ndarray
    command_nm: np.ndarray
    torque_nm: np.ndarray
    integral_nm: np.ndarray
    derivative_rad_s: np.ndarray
    command_limited: np.ndarray
    rate_limited: np.ndarray
    angle_limited: np.ndarray
    metrics: Metrics

    def frame(self) -> pd.DataFrame:
        frame = self.motion.frame()
        columns = {
            "plate_{}_deg": self.plate_deg,
            "estimated_{}_deg": self.estimate_deg,
            "joint_{}_deg": self.joint_deg,
            "joint_{}_rate_deg_s": self.rate_deg_s,
            "command_{}_nm": self.command_nm,
            "applied_{}_torque_nm": self.torque_nm,
            "integral_{}_nm": self.integral_nm,
            "derivative_{}_rad_s": self.derivative_rad_s,
            "{}_torque_saturated": self.command_limited,
            "{}_speed_limited": self.rate_limited,
            "{}_angle_limited": self.angle_limited,
        }
        for label, values in columns.items():
            for axis_i, axis in enumerate(("roll", "pitch")):
                frame[label.format(axis)] = values[:, axis_i]
        return frame


def prepare_simulation(motion: MotionInput, sensor: SensorParams = SensorParams()) -> PreparedSimulation:
    time, angles = motion.time_s, motion.angles_deg
    n = len(time)
    if (n < 2 or angles.shape != (n, 2) or not np.isfinite(angles).all()
            or time[0] != 0 or not np.isfinite(motion.dt) or motion.dt <= 0
            or not np.allclose(np.diff(time), motion.dt, rtol=1e-8, atol=1e-10)):
        raise ValueError("입력은 0초부터 시작하는 균일한 시간축과 유한한 Roll/Pitch 각도가 필요합니다.")
    radians_ship = np.deg2rad(angles)
    roll, pitch = radians_ship.T
    edge = 2 if n >= 3 else 1
    speed = np.gradient(radians_ship, motion.dt, axis=0, edge_order=edge)
    acceleration = np.gradient(speed, motion.dt, axis=0, edge_order=edge)
    rng = np.random.default_rng(sensor.seed)
    noise_angle = rng.normal(0, radians(sensor.angle_noise_deg), (n, 2))
    bias = rng.normal(0, radians(sensor.gyro_bias_deg_s), 2)
    noise_rate = rng.normal(0, radians(sensor.gyro_noise_deg_s), (n, 2)) + bias
    schedule = np.column_stack((roll, pitch, np.sin(pitch), np.cos(pitch),
                                acceleration, speed[:, 0] * speed[:, 1], noise_angle, noise_rate))
    return PreparedSimulation(motion, tuple(map(tuple, schedule.tolist())), sensor)


def _estimate(previous, current, fused, angle_noise, gyro_noise, gyro_weight, dt):
    change = current - previous
    if abs(change) > pi:
        change = (change + pi) % (2 * pi) - pi
    predicted = fused + change + gyro_noise * dt
    observed = current + angle_noise
    residual = observed - predicted
    if abs(residual) > pi:
        residual = (residual + pi) % (2 * pi) - pi
    angle = predicted + (1 - gyro_weight) * residual
    return (angle + pi) % (2 * pi) - pi if abs(angle) > pi else angle


def simulate(
    pid_params: TwoAxisPID,
    ship_data: MotionInput | PreparedSimulation,
    motor_params: MotorParams = MotorParams(),
    sensor_params: SensorParams = SensorParams(),
    pid_settings: PIDSettings = PIDSettings(),
    *,
    controller_on: bool = True,
) -> SimulationResult:
    """Simulate a window from zero joint angle/rate, command delay and PID memory."""
    prepared = (ship_data if isinstance(ship_data, PreparedSimulation)
                else prepare_simulation(ship_data, sensor_params))
    motion, dt = prepared.motion, prepared.motion.dt
    n = len(motion.time_s)
    pr = PID(pid_params.roll, dt, motor_params.roll.torque_nm, pid_settings)
    pp = PID(pid_params.pitch, dt, motor_params.pitch.torque_nm, pid_settings)
    jr = RotaryJoint(motor_params.roll, motor_params.payload_kg, dt)
    jp = RotaryJoint(motor_params.pitch, motor_params.payload_kg, dt)
    tau = prepared.sensor.fusion_tau_s
    gyro_weight = exp(-dt / tau) if tau > 0 else 0.0
    trace = np.empty((n, 16), dtype=float)
    flags = np.zeros((n, 6), dtype=bool)
    plate_r = plate_p = estimate_r = estimate_p = 0.0

    if controller_on:
        for index, row in enumerate(prepared.schedule):
            r, _, sin_p, cos_p, alpha_r, alpha_p, rate_cross, ar, ap, gr, gp = row
            roll_in_frame = r + jr.angle
            next_r, next_p = plate_angles(sin_p, cos_p, roll_in_frame, jp.angle)
            if index:
                estimate_r = _estimate(plate_r, next_r, estimate_r, ar, gr, gyro_weight, dt)
                estimate_p = _estimate(plate_p, next_p, estimate_p, ap, gp, gyro_weight, dt)
            else:
                estimate_r, estimate_p = next_r + ar, next_p + ap
            plate_r, plate_p = next_r, next_p
            cmd_r, clip_r = pr.step(-estimate_r, jr.outward_stop())
            cmd_p, clip_p = pp.step(-estimate_p, jp.outward_stop())
            trace[index] = (plate_r, plate_p, estimate_r, estimate_p, jr.angle, jp.angle, jr.rate, jp.rate,
                            cmd_r, cmd_p, jr.torque, jp.torque, pr.integral, pp.integral,
                            pr.derivative, pp.derivative)
            flags[index] = clip_r, clip_p, jr.rate_hit, jp.rate_hit, jr.angle_hit, jp.angle_hit
            if index < n - 1:
                # Project base angular acceleration onto the current two joint axes.
                jr.advance(cmd_r, alpha_r)
                jp.advance(cmd_p, cos(roll_in_frame) * alpha_p - sin(roll_in_frame) * rate_cross)
    else:
        trace.fill(0)
        for index, row in enumerate(prepared.schedule):
            r, p, _, _, _, _, _, ar, ap, gr, gp = row
            if index:
                estimate_r = _estimate(plate_r, r, estimate_r, ar, gr, gyro_weight, dt)
                estimate_p = _estimate(plate_p, p, estimate_p, ap, gp, gyro_weight, dt)
            else:
                estimate_r, estimate_p = r + ar, p + ap
            plate_r, plate_p = r, p
            trace[index, :4] = r, p, estimate_r, estimate_p

    trace[:, :8] = np.rad2deg(trace[:, :8])
    if not controller_on:
        trace[:, :2] = motion.angles_deg
    trace.flags.writeable = flags.flags.writeable = False
    # Metrics use the same arrays exported to the UI and CSV.
    result = SimulationResult(motion, controller_on,
                              *(trace[:, i:i + 2] for i in range(0, 16, 2)),
                              *(flags[:, i:i + 2] for i in range(0, 6, 2)), metrics=None)
    return replace(result, metrics=measure(result, motor_params))
