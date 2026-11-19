import numpy as np
import pytest

from motor_model import MotorAxis, MotorParams, RotaryJoint
from pid_controller import Gains, PID, PIDSettings, TwoAxisPID
from ship_model import gimbal_rotation, matrix_angles, ship_rotation
from simulation import SensorParams, prepare_simulation, simulate


def test_pid_proportional_integral_and_derivative_and_reset():
    pid = PID(Gains(2, 1, 0.5), 0.1, 10, PIDSettings(0, 1))
    assert pid.step(0.25) == (0.5, False)
    assert pid.step(0.35)[0] == pytest.approx(1.235)
    pid.reset()
    assert pid.integral == pid.derivative == 0
    assert pid.step(0.35)[0] == pytest.approx(0.70)
    filtered = PID(Gains(0, 0, 1), 0.1, 20, PIDSettings(0.1, 1))
    filtered.step(0)
    assert 0 < filtered.step(1)[0] < 10


def test_anti_windup_handles_both_directions():
    pid = PID(Gains(15, 2, 0), 0.02, 0.5, PIDSettings())
    for direction in (1, -1):
        pid.reset()
        for _ in range(100):
            assert pid.step(direction)[0] == 0.5 * direction
        assert pid.integral == 0
        pid.step(0.01 * direction, outward_stop=direction)
        assert pid.integral == 0
        pid.step(-0.01 * direction)
        assert pid.integral * direction < 0


def test_motor_delay_mass_and_limits():
    spec = MotorAxis(delay_s=0.05, response_s=0, torque_nm=1, max_angle_deg=8, max_rate_deg_s=25)
    light, loaded = (RotaryJoint(spec, m, 0.02) for m in (0, 10))
    for _ in range(2):
        light.advance(20, 0)
        loaded.advance(20, 0)
        assert light.rate == loaded.rate == 0
    light.advance(20, 0)
    loaded.advance(20, 0)
    assert light.torque == pytest.approx(0.5)
    assert light.rate > loaded.rate > 0
    rate_hit = angle_hit = False
    for _ in range(100):
        light.advance(20, 0)
        rate_hit |= light.rate_hit
        angle_hit |= light.angle_hit
        assert 0 <= light.torque <= spec.torque_nm
        assert abs(light.rate) <= light.rate_limit
        assert abs(light.angle) <= light.angle_limit
    assert rate_hit and angle_hit
    light.reset()
    assert light.angle == light.rate == light.torque == 0


def test_controller_off_follows_ship_and_uses_same_initial_conditions(train):
    prepared = prepare_simulation(train, SensorParams(seed=102))
    off = simulate(TwoAxisPID(), prepared, controller_on=False)
    on = simulate(TwoAxisPID(), prepared)
    np.testing.assert_array_equal(off.plate_deg, train.angles_deg)
    np.testing.assert_array_equal(off.joint_deg, 0.0)
    np.testing.assert_array_equal(off.torque_nm, 0.0)
    np.testing.assert_allclose(on.estimate_deg[0], off.estimate_deg[0])
    np.testing.assert_array_equal(on.integral_nm[0], 0.0)
    assert not any((off.metrics.roll.violation_samples, on.metrics.roll.violation_samples))


def test_kinematics_and_repeatable_sensor_and_plant(train):
    assert (ship_rotation(12, 0) @ [0, 1, 0])[2] > 0
    assert (ship_rotation(0, 12) @ [1, 0, 0])[2] > 0
    known = ship_rotation(38, 29) @ gimbal_rotation(-21, 18)
    assert not np.allclose(matrix_angles(known), [17, 47])
    np.testing.assert_allclose(ship_rotation(38, 29) @ gimbal_rotation(-38, -29), np.eye(3), atol=1e-14)
    gains = TwoAxisPID()
    first = simulate(gains, train)
    second = simulate(gains, train)
    assert first.metrics == second.metrics
    np.testing.assert_array_equal(first.frame().values, second.frame().values)
    for index in range(0, len(train.time_s), 17):
        matrix = ship_rotation(*train.angles_deg[index]) @ gimbal_rotation(*first.joint_deg[index])
        np.testing.assert_allclose(matrix_angles(matrix), first.plate_deg[index], atol=1e-12)


def test_high_proportional_gain_can_degrade_without_exceeding_hard_limits(train):
    baseline = simulate(TwoAxisPID(), train)
    high = simulate(TwoAxisPID.shared(Gains(24, 1, 0.2)), train, MotorParams())
    assert high.metrics.roll.rms_deg > baseline.metrics.roll.rms_deg
    assert high.metrics.roll.torque_saturation_events > 0
    assert high.metrics.roll.violation_samples == 0
