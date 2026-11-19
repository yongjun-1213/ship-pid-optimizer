"""Vessel/gimbal kinematics and the explicitly synthetic ocean."""

from math import atan2, cos, hypot, sin

import numpy as np

from data_loader import DataQuality, MotionData, VIRTUAL, _immutable

SEA_STATES = {
    "잔잔한 바다": (0.5, 4.0, 12.0, 5, 1.1),
    "보통 파도": (1.6, 6.5, 28.0, 6, 1.9),
    "거친 파도": (3.6, 8.0, 44.0, 8, 2.8),
}
GRAVITY = 9.81


def _rx(angle):
    c, s = cos(angle), sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(angle):
    c, s = cos(angle), sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def ship_rotation(roll_deg: float, pitch_deg: float) -> np.ndarray:
    """Body to world. +x bow, +y port, +z up. Positive pitch raises the bow."""
    r, p = np.deg2rad([roll_deg, pitch_deg])
    return _ry(-p) @ _rx(r)


def roll_joint_rotation(roll_deg: float) -> np.ndarray:
    return _rx(np.deg2rad(roll_deg))


def gimbal_rotation(roll_deg: float, pitch_deg: float) -> np.ndarray:
    """Roll is the outer joint. Pitch is about the rotated inner y axis."""
    r, p = np.deg2rad([roll_deg, pitch_deg])
    return _rx(r) @ _ry(-p)


def matrix_angles(matrix: np.ndarray) -> np.ndarray:
    x, y, z = matrix[2]
    return np.rad2deg([atan2(y, z), atan2(x, hypot(y, z))])


def plate_angles(sin_ship_pitch: float, cos_ship_pitch: float,
                 roll_in_frame: float, joint_pitch: float) -> tuple[float, float]:
    """Evaluate the third row of R_ship @ R_gimbal using scalar radian math."""
    c, s = cos(joint_pitch), sin(joint_pitch)
    after_roll_y = cos_ship_pitch * sin(roll_in_frame)
    after_roll_z = cos_ship_pitch * cos(roll_in_frame)
    world_x = sin_ship_pitch * c + after_roll_z * s
    world_z = -sin_ship_pitch * s + after_roll_z * c
    return atan2(after_roll_y, world_z), atan2(world_x, hypot(after_roll_y, world_z))


def wave_components(sea: str, seed: int = 123):
    """Seeded spectral wave mixture adapted from the supplied reference."""
    hs, tp, direction, count, gain = SEA_STATES[sea]
    rng = np.random.default_rng(seed)
    fp = 1.0 / tp
    freqs = np.linspace(fp * 0.45, fp * 2.0, count)
    amps = np.sqrt(freqs ** -5 * np.exp(-1.25 * (fp / freqs) ** 4))
    amps *= (hs / 4) / np.sqrt(np.sum(amps ** 2) / 2)
    angles = np.deg2rad(direction + np.linspace(-23, 23, count))
    return 2 * np.pi * freqs, amps, rng.uniform(0, 2 * np.pi, count), angles, gain


def ocean_surface(time_s: float, sea: str, *, x_range=(-36.0, 36.0), y_range=(-23.0, 23.0)):
    """Sample the same wave field over the visible area in world metres."""
    omega, amps, phase, direction, _ = wave_components(sea)
    x = np.linspace(*x_range, int(np.clip(np.ceil((x_range[1] - x_range[0]) / 1.75), 43, 140)))
    y = np.linspace(*y_range, int(np.clip(np.ceil((y_range[1] - y_range[0]) / 1.75), 32, 100)))
    xy = x[None, :, None] * np.cos(direction) + y[:, None, None] * np.sin(direction)
    height = np.cos(xy * (omega ** 2 / GRAVITY) - omega * time_s + phase) @ amps
    return x, y, height


def virtual_motion(sea: str = "보통 파도", duration: float = 360.0) -> MotionData:
    dt = 0.1
    t = np.arange(round(duration / dt) + 1, dtype=float) * dt
    omega, amps, phase, direction, gain = wave_components(sea)
    slopes = -np.sin(phase - t[:, None] * omega) * amps * (omega ** 2 / GRAVITY)
    angles = np.rad2deg(0.75 * gain * slopes @ np.column_stack((np.sin(direction), np.cos(direction))))
    quality = DataQuality(len(t), 0, 0, 0, dt, dt, 0)
    return MotionData(VIRTUAL, f"waves-v1:{sea}", sea, _immutable(t), _immutable(angles),
                      ((0, len(t)),), quality, wave=sea)
