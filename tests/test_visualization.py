import numpy as np

from pid_controller import TwoAxisPID
from ship_model import matrix_angles
from simulation import simulate
from visualization import AI_PID, PLATE_LOCAL, scene_figure, scene_pose


def test_plotted_plate_is_the_simulated_plate(train):
    run = simulate(TwoAxisPID(), train)
    index = 90
    pose = scene_pose(train.angles_deg[index], run.joint_deg[index])
    np.testing.assert_allclose(matrix_angles(pose.plate), run.plate_deg[index], atol=1e-10)
    view = scene_figure(run, index, close=False, show_reference=True, show_off=True,
                        sea="보통 파도", selected=AI_PID)
    trace = next(trace for trace in view.data if trace.name == f"{AI_PID} 상판")
    np.testing.assert_allclose(np.column_stack([trace.x, trace.y, trace.z]),
                               PLATE_LOCAL[0] @ pose.plate.T + pose.pivot)
    plane = next(trace for trace in view.data if trace.name == "수평 기준면")
    assert len(set(plane.z)) == 1
