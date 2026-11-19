"""Plotly view of the supplied hull and the joint states calculated by the engine."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ship_model import SEA_STATES, gimbal_rotation, ocean_surface, roll_joint_rotation, ship_rotation
from simulation import SimulationResult
from vessel_specs import GUNNERUS_SPEC, VesselSpec

INK = "#163047"
MUTED = "#526C82"
GRID = "#E1EAF1"
TEAL = "#007E77"
AMBER = "#B7631D"
MANUAL = "#416DAD"
OFF = "제어 OFF"
MANUAL_PID = "수동 PID"
AI_PID = "AI 최적 PID"
COLORS = {OFF: AMBER, MANUAL_PID: MANUAL, AI_PID: TEAL}
AXES = ("Roll", "Pitch")

# Hull, superstructure, mount, 4:3 plate, and payload from the supplied model.
BOX_FACES = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def box(cx, cy, cz, lx, ly, lz):
    V = np.array([[x, y, z] for x in (cx - lx / 2, cx + lx / 2)
                  for y in (cy - ly / 2, cy + ly / 2) for z in (cz - lz / 2, cz + lz / 2)], float)
    f = np.array(BOX_FACES)
    return V, f[:, 0], f[:, 1], f[:, 2]


def hull_mesh(vessel: VesselSpec = GUNNERUS_SPEC):
    stations = np.linspace(-1.0, 1.0, 35)
    x = stations * vessel.length_m / 2
    if vessel.kind == "cargo":
        exponent = np.where(stations < 0, 8.0, 4.0)
        transom, cheek = 0.36, 0.97
    elif vessel.hull_variant == "polar":
        exponent = np.where(stations < 0, 4.6, 2.8)
        transom, cheek = 0.23, 0.92
    else:
        exponent = 2.3 if vessel.kind == "research" else 2.0
        transom, cheek = 0.10, 0.88
    floor = np.where(stations < 0, transom**2, 0.01)
    half = (vessel.beam_m / 2) * np.sqrt(np.maximum(floor, 1 - np.abs(stations)**exponent))
    draught = vessel.draft_m * (1 - 0.72 * np.abs(stations)**6)
    deck = vessel.deck_above_water_m

    vertices = []
    for position, width, depth in zip(x, half, draught):
        vertices.extend([
            (position, -width, deck), (position, -width * cheek, -0.20 * depth),
            (position, 0, -depth), (position, width * cheek, -0.20 * depth), (position, width, deck),
        ])
    faces = []
    for station in range(len(x) - 1):
        for j in range(4):
            a = station * 5 + j
            c = a + 5
            faces.extend(((a, a + 1, c), (a + 1, c + 1, c)))
    end = len(vertices) - 5
    faces.extend([(0, a + 1, a) for a in range(1, 4)])
    faces.extend([(end, end + a, end + a + 1) for a in range(1, 4)])
    return np.asarray(vertices), *np.asarray(faces).T


# Deckhouse proportions use the original research-vessel model as the reference.
RESEARCH_LAYOUT = (
    (4.0, 0, 1.10, 9.0, 5.4, 2.2), (5.8, 0, 2.90, 4.8, 4.8, 1.7),
    (0.3, 0, 2.50, 1.0, 1.0, 2.5), (0.3, 0, 4.00, 2.2, 0.35, 0.25),
    (-3.2, 0, 1.60, 1.8, 1.5, 2.8),
)
POLAR_LAYOUT = (
    (2.1, 0, 1.65, 13.5, 7.0, 3.3), (4.5, 0, 3.80, 7.5, 6.3, 1.8),
    (5.3, 0, 5.00, 5.4, 5.8, .80), (2.6, 0, 5.95, 1.3, 1.3, 1.3),
    (-4.6, 0, 2.44, 2.0, 1.4, 2.1),
)
CARGO_BRIDGE = (
    (-12.3, 0, 1.40, 5.0, 5.0, 2.8), (-12.7, 0, 3.25, 3.5, 4.4, 1.05),
)
CARGO_HOLDS = tuple((x, 0, 0.36, 4.60, 6.4, .72) for x in (-4.0, 1.65, 7.30, 12.95))
GENERIC_LAYOUT = ((3.0, 0, .95, 6.6, 4.3, 1.9), (3.0, 0, 2.10, 3.1, 3.5, .50))


@dataclass(frozen=True)
class VesselGeometry:
    hull: tuple
    deck: tuple
    rails: tuple
    structures: tuple
    mount: np.ndarray
    deck_marking: np.ndarray | None
    top_m: float
    color: str


@lru_cache(maxsize=24)
def vessel_geometry(vessel: VesselSpec) -> VesselGeometry:
    hull = hull_mesh(vessel)
    v = hull[0]
    stations = np.arange(0, len(v) - 5, 5)
    deck = (v, np.r_[stations, stations + 4], np.r_[stations + 4, stations + 9],
            np.r_[stations + 5, stations + 5])
    height = vessel.deck_above_water_m
    rail_height = vessel.beam_m * .0125
    if vessel.height_above_water_m is not None:
        rail_height = min(rail_height, vessel.height_above_water_m - height)
    rails = (v[4::5] + [0, 0, rail_height], v[::5] + [0, 0, rail_height])
    marking = None
    structures = []
    if vessel.kind == "cargo":
        layout, mount, color = CARGO_BRIDGE, (-.226, 0), "#32475B"
        structures.extend((coords, "#778592", "화물용 해치") for coords in CARGO_HOLDS)
    elif vessel.hull_variant == "polar":
        layout, mount, color = POLAR_LAYOUT, (-.207, .24), "#AC5340"
        angle = np.linspace(0, 2 * np.pi, 49)
        radius = 0.24 * vessel.beam_m
        marking = np.column_stack((-.343 * vessel.length_m + radius * np.cos(angle),
                                   radius * np.sin(angle), np.full_like(angle, height + rail_height * .36)))
    elif vessel.kind == "research":
        layout, mount, color = RESEARCH_LAYOUT, (-11.5 / 36.2, 0), "#385878"
    else:
        layout, mount, color = GENERIC_LAYOUT, (-.265, 0), "#46697D"
    structures.extend((coords, "#F4F3E9", "상부 구조물") for coords in layout)
    x_scale, y_scale = vessel.length_m / 36.2, vessel.beam_m / 9.6
    reference_top = max(coords[2] + coords[5] / 2 for coords, _, _ in structures)
    z_scale = (vessel.height_above_water_m - height) / reference_top if (
        vessel.height_above_water_m is not None
    ) else y_scale
    boxes = tuple((box(x * x_scale, y * y_scale, height + z * z_scale,
                       lx * x_scale, ly * y_scale, lz * z_scale), paint, label)
                  for (x, y, z, lx, ly, lz), paint, label in structures)
    anchor = np.array([mount[0] * vessel.length_m, mount[1] * vessel.beam_m, height])
    return VesselGeometry(hull, deck, rails, boxes, anchor, marking,
                          height + reference_top * z_scale, color)


# The reference scene shows a 0.40×0.30 m plate at 16× scale.
DEMO_SCALE = 16.0
PLATE_LENGTH_M = 0.40
PLATE_BEAM_M = 0.30
PLATE = box(0, 0, 1.6, PLATE_LENGTH_M * DEMO_SCALE, PLATE_BEAM_M * DEMO_SCALE, 0.3)
CUP = box(0, 0, 2.55, 1.1, 1.1, 1.6)
POST = box(0, 0, 0.75, 0.6, 0.6, 1.5)
POST_TOP = np.array([0.0, 0.0, 1.5])


@dataclass(frozen=True)
class ScenePose:
    ship: np.ndarray
    base: np.ndarray
    pivot: np.ndarray
    outer: np.ndarray
    plate: np.ndarray


def scene_pose(ship_deg, joint_deg, *, vessel: VesselSpec = GUNNERUS_SPEC, device_scale: float = DEMO_SCALE):
    ship = ship_rotation(*ship_deg)
    base = ship @ vessel_geometry(vessel).mount
    return ScenePose(
        ship, base, base + ship @ (POST_TOP * device_scale / DEMO_SCALE),
        ship @ roll_joint_rotation(joint_deg[0]),
        ship @ gimbal_rotation(*joint_deg),
    )


def _tf(vertices, frame, translation=0):
    return vertices @ frame.T + translation


def _mesh(shape, frame, origin, color, name, *, legend=False, opacity=1.0):
    vertices, i, j, k = shape
    xyz = _tf(vertices, frame, origin)
    return go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], i=i, j=j, k=k, color=color, name=name,
        opacity=opacity, flatshading=True, showlegend=legend, hoverinfo="skip",
        lighting=dict(ambient=0.65, diffuse=0.65, roughness=0.8, specular=0.22),
        lightposition=dict(x=-20, y=15, z=35),
    )


def _line(xyz, color, name, *, width=4, legend=False, dash="solid"):
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                        line=dict(color=color, width=width, dash=dash), name=name,
                        showlegend=legend, hoverinfo="skip")


PLATE_LOCAL = (PLATE[0] - POST_TOP, *PLATE[1:])
PAYLOAD_LOCAL = (CUP[0] - POST_TOP, *CUP[1:])
PLATE_EDGE = np.array([[-3.2, -2.4, .26], [3.2, -2.4, .26], [3.2, 2.4, .26],
                       [-3.2, 2.4, .26], [-3.2, -2.4, .26]])
LEVEL = np.array([[-3.95, -3.35, .54], [3.95, -3.35, .54], [3.95, 3.35, .54],
                  [-3.95, 3.35, .54], [-3.95, -3.35, .54]])
RING_ANGLE = np.linspace(-np.pi * 0.90, np.pi * 0.90, 34)
OUTER_RING = np.column_stack((np.zeros_like(RING_ANGLE), 2.62 * np.sin(RING_ANGLE),
                              -1.05 * np.cos(RING_ANGLE)))
INNER_RING = np.column_stack((2.64 * np.sin(RING_ANGLE), np.zeros_like(RING_ANGLE),
                              -0.77 * np.cos(RING_ANGLE)))


def _scene_ranges(vessel, geometry, result, pivot, close, scale, wave_height):
    if close:
        half = np.array([5.65, 5.65, 5.10]) * scale
        return [tuple(r) for r in np.column_stack((pivot - half, pivot + half))]

    radius = 4.6 * scale
    x = max(vessel.length_m / 2, abs(geometry.mount[0]) + radius)
    y = max(vessel.beam_m / 2, abs(geometry.mount[1]) + radius)
    top = max(geometry.top_m, geometry.rails[0][0, 2], geometry.mount[2] + radius)
    roll, pitch = np.deg2rad(np.max(np.abs(result.motion.angles_deg), axis=0))
    height = max(top, vessel.draft_m)
    z_pad = x * np.sin(pitch) + y * np.sin(roll) + max(wave_height, .075 * vessel.beam_m)
    x += (height + y * np.sin(roll)) * np.sin(pitch) + .13 * vessel.length_m
    y += height * np.sin(roll) + .55 * vessel.beam_m
    return [(-x, x), (-y, y), (-vessel.draft_m - z_pad, top + z_pad)]


def _scaled_mesh(shape, scale):
    return shape[0] * scale, *shape[1:]


def scene_figure(
    result: SimulationResult, index: int, *, close: bool, show_reference: bool,
    show_off: bool, sea: str, selected: str, vessel: VesselSpec = GUNNERUS_SPEC,
    device_scale: float = DEMO_SCALE,
) -> go.Figure:
    motion = result.motion
    device_factor = device_scale / DEMO_SCALE
    geometry = vessel_geometry(vessel)
    att = scene_pose(motion.angles_deg[index], result.joint_deg[index], vessel=vessel,
                     device_scale=device_scale)
    h = SEA_STATES[sea][0] * .7
    ranges = _scene_ranges(vessel, geometry, result, att.pivot, close, device_factor, h)
    palette = [[0, "#203F5B"], [0.40, "#216582"], [0.73, "#328AA0"], [1, "#97C7D1"]]
    fig = go.Figure()
    if not close or ranges[2][0] <= h:
        x, y, z = ocean_surface(motion.source_elapsed_s[index], sea, x_range=ranges[0], y_range=ranges[1])
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, colorscale=palette, cmin=-h, cmax=h, showscale=False,
            hoverinfo="skip", name="가상 수면", lighting=dict(ambient=.85, diffuse=.4, roughness=.8),
        ))
    fig.add_trace(_mesh(geometry.hull, att.ship, 0, geometry.color, "선체"))
    fig.add_trace(_mesh(geometry.deck, att.ship, 0, "#B6C9D0", "갑판"))
    for cabin, paint, label in geometry.structures:
        fig.add_trace(_mesh(cabin, att.ship, 0, paint, label))
    for rail in geometry.rails:
        fig.add_trace(_line(_tf(rail, att.ship), "#F3F7F8", "갑판 테두리", width=4))
    if geometry.deck_marking is not None:
        fig.add_trace(_line(_tf(geometry.deck_marking, att.ship), "#F7F1D3", "선미 헬기갑판 표시", width=4))
    fig.add_trace(_mesh(_scaled_mesh(POST, device_factor), att.ship, att.base, "#25384B", "짐벌 지지대"))
    fig.add_trace(_line(_tf(OUTER_RING * device_factor, att.outer, att.pivot), "#455A69", "외부 Roll 축",
                        width=7, legend=close))
    fig.add_trace(_line(_tf(INNER_RING * device_factor, att.plate, att.pivot), "#587987", "내부 Pitch 축",
                        width=6, legend=close))
    if show_off and result.controller_on:
        fig.add_trace(_line(_tf(PLATE_EDGE * device_factor, att.ship, att.pivot),
                            "#E69656", "제어 OFF 상판", width=5, dash="dash", legend=True))
    fig.add_trace(_mesh(_scaled_mesh(PLATE_LOCAL, device_factor), att.plate, att.pivot,
                        COLORS[selected], selected + " 상판", legend=True))
    fig.add_trace(_mesh(_scaled_mesh(PAYLOAD_LOCAL, device_factor), att.plate, att.pivot,
                        "#F2BE61", "계측장비", legend=True))
    if show_reference:
        fixed = LEVEL * device_factor + att.pivot
        fig.add_trace(go.Mesh3d(x=fixed[:4, 0], y=fixed[:4, 1], z=fixed[:4, 2],
                                i=[0, 0], j=[1, 2], k=[2, 3], color="#C0F6EC",
                                opacity=0.19, showlegend=False, hoverinfo="skip", name="수평 기준면"))
        fig.add_trace(_line(fixed, "#449A90", "수평 기준면", width=4, dash="dot", legend=True))

    panel_bg = "#E8F1F6"
    if close:
        camera = dict(eye=dict(x=-1.43, y=1.30, z=0.72), center=dict(x=0, y=0, z=0.02))
    else:
        camera = dict(eye=dict(x=-1.0, y=0.96, z=0.63), center=dict(x=0, y=0, z=-0.03))
    span = max(high - low for low, high in ranges)
    axes = [dict(title=title, range=extent, gridcolor="#C3D6E2", backgroundcolor=panel_bg,
                 showbackground=True, color=MUTED, zeroline=False, nticks=6, tickfont=dict(size=10))
            for title, extent in zip(("선수 방향 (m)", "좌현 방향 (m)", "높이 (m)"), ranges)]
    view_key = (vessel.kind, vessel.hull_variant, vessel.length_m, vessel.beam_m, vessel.draft_m,
                vessel.deck_above_water_m, geometry.top_m, close, device_scale)
    fig.update_layout(
        height=430, autosize=True, uirevision=str(view_key), paper_bgcolor=panel_bg,
        margin=dict(l=0, r=0, t=4, b=0), font=dict(color=INK, size=12),
        scene=dict(xaxis=axes[0], yaxis=axes[1], zaxis=axes[2], aspectmode="manual",
                   aspectratio={key: (high - low) / span for key, (low, high) in zip("xyz", ranges)},
                   camera=camera, bgcolor=panel_bg, dragmode="orbit"),
        legend=dict(orientation="h", y=0.015, x=.025, bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#CEE1EB", borderwidth=1, font=dict(size=11, color=INK)),
    )
    return fig


def _line_chart(fig: go.Figure, height: int = 365):
    fig.update_layout(
        font=dict(color=INK, size=12), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        height=height, margin=dict(l=15, r=15, t=48, b=20), hovermode="x unified",
        legend=dict(orientation="h", y=1.11, x=1, xanchor="right", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _sample_indexes(n: int, limit: int = 1500):
    return np.unique(np.linspace(0, n - 1, min(n, limit), dtype=int))


def attitude_chart(results: dict[str, SimulationResult], *, cursor: int | None = None):
    source = next(iter(results.values())).motion
    n = len(source.time_s) if cursor is None else cursor + 1
    indexes = _sample_indexes(n)
    t = source.time_s[indexes]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Roll · 좌우", "Pitch · 앞뒤"))
    for axis_i, axis in enumerate(AXES):
        col = axis_i + 1
        if OFF not in results:
            fig.add_trace(go.Scatter(x=t, y=source.angles_deg[indexes, axis_i],
                                     line=dict(color=AMBER, width=1.8, dash="dot"),
                                     name="선박", legendgroup="ship", showlegend=col == 1,
                                     mode="lines+markers" if n == 1 else "lines"), row=1, col=col)
        for label, run in results.items():
            fig.add_trace(go.Scatter(x=t, y=run.plate_deg[indexes, axis_i], name=label,
                                     line=dict(color=COLORS[label], width=2), legendgroup=label,
                                     showlegend=col == 1, mode="lines+markers" if n == 1 else "lines",
                                     hovertemplate="%{y:.3f}°<extra>%{fullData.name}</extra>"), row=1, col=col)
        fig.add_hrect(y0=-1, y1=1, fillcolor="#A9DDD3", opacity=.22, line_width=0, row=1, col=col)
        if cursor is not None:
            fig.add_vline(x=float(source.time_s[cursor]), line_color=TEAL, line_width=1,
                          line_dash="dot", row=1, col=col)
        fig.update_xaxes(title_text="실험 시간 (s)", row=1, col=col)
        fig.update_yaxes(title_text="각도 (°)", row=1, col=col)
    return _line_chart(fig, 285 if cursor is not None else 350)


def motor_chart(results: dict[str, SimulationResult], signal: str):
    fields = {
        "토크": (("command_nm", "PID 명령", "dot"), ("torque_nm", "적용 토크", "solid")),
        "회전각": (("joint_deg", "모터", "solid"),),
        "각속도": (("rate_deg_s", "모터", "solid"),),
    }
    units = {"토크": "N·m", "회전각": "°", "각속도": "°/s"}
    fig = make_subplots(rows=1, cols=2, subplot_titles=AXES)
    for index in range(2):
        for name, result in results.items():
            if not result.controller_on:
                continue
            i = _sample_indexes(len(result.motion.time_s))
            for key, detail, dash in fields[signal]:
                label = f"{name} · {detail}" if signal == "토크" else name
                fig.add_trace(go.Scatter(x=result.motion.time_s[i], y=getattr(result, key)[i, index],
                                         name=label, legendgroup=label, showlegend=index == 0,
                                         line=dict(color=COLORS[name], width=1.6, dash=dash)),
                              row=1, col=index + 1)
        fig.update_xaxes(title_text="실험 시간 (s)", row=1, col=index + 1)
        fig.update_yaxes(title_text=units[signal], row=1, col=index + 1)
    return _line_chart(fig, 330)


def optimization_chart(records):
    x = [record.number for record in records]
    y = [record.score for record in records]
    fig = go.Figure([
        go.Scatter(x=x, y=y, mode="markers", name="개별 실험", marker=dict(color="#7796B0", opacity=.63, size=5)),
        go.Scatter(x=x, y=np.minimum.accumulate(y), mode="lines", line_shape="hv",
                   name="누적 최저 J", line=dict(color=TEAL, width=2.7)),
    ])
    fig.update_xaxes(title_text="완료한 실험", rangemode="tozero")
    fig.update_yaxes(title_text="목적함수 J")
    return _line_chart(fig, 320)
