import numpy as np
import pytest

from pid_controller import TwoAxisPID
from playback import Playback
from simulation import simulate


def test_play_pause_resume_preserves_simulation_state(train):
    run = simulate(TwoAxisPID(), train)
    cursor = Playback(train.dt, len(train.time_s))
    cursor.play(5)
    cursor.tick(5.105)
    assert cursor.index == 5
    cursor.pause()
    saved = run.integral_nm[cursor.index].copy()
    cursor.tick(93.0)
    assert cursor.index == 5
    np.testing.assert_array_equal(saved, run.integral_nm[cursor.index])
    cursor.play(100.0)
    cursor.tick(100.015)
    assert cursor.index == 6
    cursor.pause()
    cursor.step()
    assert cursor.index == 7


def test_stop_restart_seek_step_end(train):
    run = simulate(TwoAxisPID(), train)
    cursor = Playback(train.dt, len(train.time_s))
    cursor.seek(1.019)
    assert cursor.time_s == pytest.approx(1.02)
    assert not cursor.playing
    np.testing.assert_array_equal(run.integral_nm[cursor.index], run.integral_nm[51])
    cursor.stop()
    assert cursor.index == 0 and not cursor.playing
    np.testing.assert_array_equal(run.integral_nm[cursor.index], 0)
    np.testing.assert_array_equal(run.joint_deg[cursor.index], 0)
    np.testing.assert_array_equal(run.torque_nm[cursor.index], 0)
    cursor.restart(4)
    assert (cursor.index, cursor.playing) == (0, True)
    cursor.step()
    assert cursor.index == 0
    cursor.tick(100)
    assert cursor.index == cursor.samples - 1 and not cursor.playing
    cursor.tick(101)
    cursor.step()
    assert cursor.index == cursor.samples - 1
