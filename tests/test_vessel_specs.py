from dataclasses import replace
import json

import numpy as np
import pytest

from data_loader import resample_window
from ship_model import virtual_motion
from simulation import simulate
from ui_experiments import _export_json
from ui_settings import ModelSettings
from vessel_specs import DEFAULT_VESSEL, GUNNERUS_SPEC, TRYOSHNIKOV_SPEC, VesselSpec, user_vessel_from_manifest
from visualization import DEMO_SCALE, MANUAL_PID, hull_mesh, scene_figure, scene_pose, vessel_geometry

CUSTOM = VesselSpec("테스트 화물선", "cargo", 126.0, 20.0, 7.0, 18.0)


@pytest.mark.parametrize("vessel", [GUNNERUS_SPEC, TRYOSHNIKOV_SPEC, CUSTOM])
def test_hull_uses_registered_length_beam_and_draught(vessel):
    hull, *faces = hull_mesh(vessel)
    np.testing.assert_allclose(np.ptp(hull[:, :2], axis=0), [vessel.length_m, vessel.beam_m])
    assert hull[:, 2].min() == pytest.approx(-vessel.draft_m)
    assert hull[:, 2].max() == pytest.approx(vessel.deck_above_water_m)
    assert np.max(faces) < len(hull)
    geometry = vessel_geometry(vessel)
    assert geometry.mount[2] == pytest.approx(vessel.deck_above_water_m)
    if vessel.height_above_water_m is not None:
        top = max(piece[0][:, 2].max() for piece, _, _ in geometry.structures)
        assert top == pytest.approx(vessel.height_above_water_m)


def test_ship_kind_changes_outline_at_the_same_dimensions():
    standard = vessel_geometry(replace(CUSTOM, kind="research"))
    cargo = vessel_geometry(CUSTOM)
    other = vessel_geometry(replace(CUSTOM, kind="other"))
    polar = vessel_geometry(replace(CUSTOM, kind="research", hull_variant="polar"))
    assert not np.allclose(cargo.hull[0], standard.hull[0])
    assert not np.allclose(other.hull[0], standard.hull[0])
    assert not np.allclose(polar.hull[0], standard.hull[0])
    assert sum(label == "화물용 해치" for _, _, label in cargo.structures) == 4
    assert not any(label == "화물용 해치" for _, _, label in standard.structures)
    assert polar.deck_marking is not None


@pytest.mark.parametrize("factor", [1.0, DEMO_SCALE])
def test_device_size_and_camera_do_not_depend_on_ship_scale(train, factor):
    index = 71
    run = simulate(ModelSettings().manual, train)
    spans = []
    for vessel in (GUNNERUS_SPEC, TRYOSHNIKOV_SPEC):
        options = dict(show_reference=True, show_off=True, sea="보통 파도", selected=MANUAL_PID,
                       vessel=vessel, device_scale=factor)
        figure = scene_figure(run, index, close=False, **options)
        pose = scene_pose(train.angles_deg[index], run.joint_deg[index], vessel=vessel, device_scale=factor)
        trace = next(trace for trace in figure.data if trace.name == f"{MANUAL_PID} 상판")
        local = (np.column_stack((trace.x, trace.y, trace.z)) - pose.pivot) @ pose.plate
        np.testing.assert_allclose(np.ptp(local[:, :2], axis=0), np.array([.40, .30]) * factor)

        axes = (figure.layout.scene.xaxis, figure.layout.scene.yaxis, figure.layout.scene.zaxis)
        x_span, y_span, z_span = [np.ptp(axis.range) for axis in axes]
        spans.append(x_span)
        aspect = figure.layout.scene.aspectratio
        np.testing.assert_allclose([x_span / aspect.x, y_span / aspect.y, z_span / aspect.z], x_span)
        closeup = scene_figure(run, index, close=True, **options).layout.scene
        np.testing.assert_allclose([np.mean(axis.range) for axis in (closeup.xaxis, closeup.yaxis, closeup.zaxis)],
                                   pose.pivot)
    assert spans[1] > spans[0] * 3


def test_virtual_motion_can_be_shown_on_two_sizes_without_modifying_input():
    data = virtual_motion()
    motion = resample_window(data, 0, 0, 3)
    model = ModelSettings()
    run = simulate(model.manual, motion)
    initial = run.frame()
    choices = dict(close=False, show_reference=True, show_off=True, sea=data.wave, selected=MANUAL_PID)
    for vessel in (DEFAULT_VESSEL, CUSTOM):
        scene_figure(run, 25, vessel=vessel, **choices)
    assert initial.equals(run.frame())


def test_experiment_json_and_old_settings_are_compatible(gunnerus, train):
    model = ModelSettings()
    payload = model.to_bytes(CUSTOM)
    saved = json.loads(payload)["vessel"]
    assert ModelSettings.from_bytes(payload) == ModelSettings.from_bytes(model.to_bytes())
    assert user_vessel_from_manifest(saved, DEFAULT_VESSEL) == CUSTOM
    assert user_vessel_from_manifest(GUNNERUS_SPEC.manifest(), DEFAULT_VESSEL) is None
    with pytest.raises(ValueError, match="양수"):
        user_vessel_from_manifest({**saved, "draft_m": -1}, DEFAULT_VESSEL)
    result = simulate(model.manual, train)
    report = json.loads(_export_json("A", train, gunnerus, model, {MANUAL_PID: result}, None, vessel=CUSTOM))
    assert report["vessel"] == saved
    assert report["window"]["source"] == GUNNERUS_SPEC.name
    assert TRYOSHNIKOV_SPEC.manifest()["displacement_t"] == 16539
    assert len(GUNNERUS_SPEC.manifest()["sources"]) == 2
