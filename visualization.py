"""Plotly view of the supplied hull and the joint states calculated by the engine."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ship_model import SEA_STATES, gimbal_rotation, ocean_surface, roll_joint_rotation, ship_rotation
from simulation import SimulationResult

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


def hull_mesh():
    xs = np.linspace(-18.1, 18.1, 35)
    V = []
    for x in xs:
        half = 4.8 * np.sqrt(max(0.01, 1 - abs(x / 18.1) ** 2.3))
        V += [[x, -half, 1.15], [x, -half * 0.88, -0.25], [x, 0, -1.35], [x, half * 0.88, -0.25], [x, half, 1.15]]
    I, J, K = [], [], []
    for i in range(len(xs) - 1):
        for j in range(4):
            a = i * 5 + j
            c = (i + 1) * 5 + j
            I += [a, a + 1]; J += [a + 1, c + 1]; K += [c, c]
    return np.array(V, float), np.array(I), np.array(J), np.array(K)


HULL = hull_mesh()
DECKHOUSE = [box(4.0, 0, 2.25, 9.0, 5.4, 2.2), box(5.8, 0, 4.05, 4.8, 4.8, 1.7),
             box(0.3, 0, 3.65, 1.0, 1.0, 2.5), box(0.3, 0, 5.15, 2.2, 0.35, 0.25),
             box(-3.2, 0, 2.75, 1.8, 1.5, 2.8)]
PIVOT_SHIP = np.array([-11.5, 0.0, 1.15])          # 선미 갑판 위 짐벌 회전 중심
PLATE = box(0, 0, 1.6, 6.4, 4.8, 0.3)               # 상판 (실제 40×30cm를 확대 표시, 4:3 비율)
CUP = box(0, 0, 2.55, 1.1, 1.1, 1.6)                # 상판 위 적재물
POST = box(0, 0, 0.75, 0.6, 0.6, 1.5)               # 짐벌 지지대 (선박에 고정)


POST_TOP = np.array([0.0, 0.0, 1.5])


@dataclass(frozen=True)
class ScenePose:
    ship: np.ndarray
    base: np.ndarray
    pivot: np.ndarray
    outer: np.ndarray
    plate: np.ndarray


def scene_pose(ship_deg, joint_deg) -> ScenePose:
    ship = ship_rotation(*ship_deg)
    base = ship @ PIVOT_SHIP
    return ScenePose(
        ship, base, base + ship @ POST_TOP,
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


def _deck():
    v = HULL[0]
    indexes = np.arange(0, len(v) - 5, 5)
    i, j, k = np.concatenate((indexes, indexes + 4)), np.concatenate((indexes + 4, indexes + 9)), np.concatenate(
        (indexes + 5, indexes + 5)
    )
    return v, i, j, k


DECK = _deck()
PORT_RAIL = HULL[0][4::5] + [0, 0, 0.12]
STARBOARD_RAIL = HULL[0][::5] + [0, 0, 0.12]
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


def scene_figure(
    result: SimulationResult, index: int, *, close: bool, show_reference: bool,
    show_off: bool, sea: str, selected: str,
) -> go.Figure:
    motion = result.motion
    time = motion.source_elapsed_s[index]
    att = scene_pose(motion.angles_deg[index], result.joint_deg[index])
    x, y, z = ocean_surface(time, sea)
    h = SEA_STATES[sea][0] * .7
    palette = [[0, "#203F5B"], [0.40, "#216582"], [0.73, "#328AA0"], [1, "#97C7D1"]]
    fig = go.Figure(go.Surface(
        x=x, y=y, z=z, colorscale=palette, cmin=-h, cmax=h, showscale=False,
        hoverinfo="skip", name="가상 수면", lighting=dict(ambient=.85, diffuse=.4, roughness=.8),
    ))
    fig.add_trace(_mesh(HULL, att.ship, 0, "#385878", "선체"))
    fig.add_trace(_mesh(DECK, att.ship, 0, "#B6C9D0", "갑판"))
    for cabin in DECKHOUSE:
        fig.add_trace(_mesh(cabin, att.ship, 0, "#F4F3E9", "상부 구조물"))
    for rail in (PORT_RAIL, STARBOARD_RAIL):
        fig.add_trace(_line(_tf(rail, att.ship), "#F3F7F8", "갑판 테두리", width=4))
    fig.add_trace(_mesh(POST, att.ship, att.base, "#25384B", "짐벌 지지대"))
    fig.add_trace(_line(_tf(OUTER_RING, att.outer, att.pivot), "#455A69", "외부 Roll 축",
                        width=7, legend=close))
    fig.add_trace(_line(_tf(INNER_RING, att.plate, att.pivot), "#587987", "내부 Pitch 축",
                        width=6, legend=close))
    if show_off and result.controller_on:
        fig.add_trace(_line(_tf(PLATE_EDGE, att.ship, att.pivot), "#E69656", "제어 OFF 상판",
                            width=5, dash="dash", legend=True))
    fig.add_trace(_mesh(PLATE_LOCAL, att.plate, att.pivot, COLORS[selected], selected + " 상판", legend=True))
    fig.add_trace(_mesh(PAYLOAD_LOCAL, att.plate, att.pivot, "#F2BE61", "계측장비", legend=True))
    if show_reference:
        fixed = LEVEL + att.pivot
        fig.add_trace(go.Mesh3d(x=fixed[:4, 0], y=fixed[:4, 1], z=fixed[:4, 2],
                                i=[0, 0], j=[1, 2], k=[2, 3], color="#C0F6EC",
                                opacity=0.19, showlegend=False, hoverinfo="skip", name="수평 기준면"))
        fig.add_trace(_line(fixed, "#449A90", "수평 기준면", width=4, dash="dot", legend=True))

    panel_bg = "#E8F1F6"
    if close:
        ranges = [(-17.2, -5.8), (-6.0, 6.0), (-2.4, 9.0)]
        camera = dict(eye=dict(x=-1.43, y=1.30, z=0.72), center=dict(x=0, y=0, z=0.02))
    else:
        ranges = [(-23.0, 23.0), (-15.0, 15.0), (-4.6, 9.4)]
        camera = dict(eye=dict(x=-1.0, y=0.96, z=0.63), center=dict(x=0, y=0, z=-0.03))
    span = max(high - low for low, high in ranges)
    axes = [dict(title=title, range=extent, gridcolor="#C3D6E2", backgroundcolor=panel_bg,
                 showbackground=True, color=MUTED, zeroline=False, nticks=6, tickfont=dict(size=10))
            for title, extent in zip(("선수 방향 (m)", "좌현 방향 (m)", "높이 (m)"), ranges)]
    fig.update_layout(
        height=430, autosize=True, uirevision=f"gimbal-{close}", paper_bgcolor=panel_bg,
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
