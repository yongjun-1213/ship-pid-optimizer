"""Playback actions and synchronized vessel and attitude diagrams."""

from time import monotonic

import streamlit as st

from playback import Playback
from ui_experiments import PLOT_CONFIG, comparison, data_summary, experiment_picker, model_identity
from ui_vessel import default_vessel_for, dimensions_text
from vessel_specs import VESSEL_KINDS
from visualization import (AI_PID, DEMO_SCALE, MANUAL_PID, OFF, PLATE_BEAM_M,
                           PLATE_LENGTH_M, attitude_chart, scene_figure)


def render_scene(runs, model, applied, sea, *, vessels=None):
    header, picker = st.columns([3, 1])
    header.subheader("3D 시뮬레이션")
    with picker:
        exp = experiment_picker(runs, "scene_experiment")
    motion = runs[exp]
    vessel = (vessels[exp] if vessels is not None else default_vessel_for(motion.window.source)).spec
    data_summary(motion)
    profile = "개략 모형" if vessel.provenance != "user" else "사용자 지정 3D 제원"
    st.caption(f"{profile} · {vessel.name} · {VESSEL_KINDS[vessel.kind]} · {dimensions_text(vessel)}")
    st.caption("파도는 합성 수면입니다. 실측 입력 모드에서는 관측된 Roll·Pitch만 선박 자세에 반영합니다."
               if not motion.wave else "파도 성분으로 생성한 합성 Roll·Pitch를 재생합니다.")
    signature = (model_identity(motion, model), model.manual, model.enabled,
                 applied.study_id if applied else None, applied.trial_number if applied else None)
    if st.session_state.get("playback_signature") != signature:
        st.session_state.playback_signature = signature
        st.session_state.playback = Playback(motion.dt, len(motion.time_s))
    cursor = st.session_state.playback
    results = comparison(motion, model.manual, model.motor, model.sensor, model.pid,
                         applied.gains if applied else None)
    view = st.columns([1.27, 1.48, 1.32, .83, .98, 1.06], vertical_alignment="bottom")
    modes = list(results) if model.enabled else [OFF]
    request = st.session_state.pop("scene_request", None)
    preferred = request or st.session_state.get("scene_focus") or MANUAL_PID
    if preferred not in modes:
        preferred = MANUAL_PID if model.enabled else OFF
    if request is not None or st.session_state.get("scene_focus") not in modes:
        st.session_state.scene_focus = preferred
    selected = view[0].selectbox("상판 제어", modes, key="scene_focus")
    close = view[1].selectbox("카메라", ("선박 전체", "수평유지장치 확대"), key="scene_camera") == "수평유지장치 확대"
    device_scale = view[2].selectbox(
        "장치 표시 크기", (1.0, DEMO_SCALE), index=1, key="scene_device_scale",
        format_func={1.0: "실제 크기 (1×)", DEMO_SCALE: "시연용 확대 (16×)"}.get,
        help="실제 상판 기준은 0.40 × 0.30 m입니다. 표시 크기는 모든 선박에 동일하게 적용됩니다.",
    )
    speed = view[3].selectbox("재생 속도", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0], index=2,
                              format_func=lambda s: f"{s:g}×", key="scene_speed")
    show_ref = view[4].toggle("수평 기준면", value=True, key="show_reference")
    show_off = view[5].toggle("제어 OFF 윤곽", value=True, key="show_off")
    initially_running = cursor.playing

    @st.fragment(run_every=0.26 if initially_running else None)
    def playback_view():
        play, pause, stop, restart, step = st.columns(5)
        is_end = cursor.index >= cursor.samples - 1
        if play.button("재생", type="primary", disabled=cursor.playing or is_end, width="stretch"):
            cursor.play(monotonic())
            st.rerun()
        if pause.button("일시정지", disabled=not cursor.playing, width="stretch"):
            cursor.pause()
            st.rerun()
        if stop.button("정지", width="stretch"):
            cursor.stop()
            st.rerun()
        if restart.button("처음부터 재생", width="stretch"):
            cursor.restart(monotonic())
            st.rerun()
        if step.button("한 스텝", disabled=cursor.playing or is_end, width="stretch"):
            cursor.step()
            st.rerun()

        cursor.tick(monotonic(), speed)
        if initially_running and not cursor.playing:
            st.rerun()
        index, run = cursor.index, results[selected]
        counter, *panels = st.columns([1.35, 1, 1, 1, 1])
        status = "재생 중" if cursor.playing else "구간 종료" if index == cursor.samples - 1 else (
            "시작 위치" if index == 0 else "일시정지"
        )
        counter.metric("실험 시간 (초)", f"{cursor.time_s:.3f}")
        for column, title, angle in zip(
            panels, ("선박 Roll", "선박 Pitch", "상판 Roll", "상판 Pitch"),
            (*motion.angles_deg[index], *run.plate_deg[index]),
        ):
            column.metric(title, f"{angle:+.2f}°")
        scene_sea = motion.wave or sea
        figure = scene_figure(run, index, close=close, show_reference=show_ref,
                              show_off=show_off, sea=scene_sea, selected=selected,
                              vessel=vessel, device_scale=device_scale)
        st.plotly_chart(figure, key="motion_scene", width="stretch", config=PLOT_CONFIG)
        st.caption(f"{status} · 마우스로 회전 · 휠로 확대 · "
                   f"상판 {PLATE_LENGTH_M * device_scale:.2f} × {PLATE_BEAM_M * device_scale:.2f} m 표시 "
                   f"({device_scale:g}×)")
        move, go = st.columns([3, 1], vertical_alignment="bottom")
        target = move.number_input("이동할 실험 시간 (초)", 0.0, float(motion.time_s[-1]),
                                   step=motion.dt, format="%.3f",
                                   key=f"seek_{motion.digest[:16]}")
        if go.button("시간 이동", width="stretch"):
            cursor.seek(target)
            st.rerun()

        shown = {OFF: results[OFF], selected: run}
        st.plotly_chart(attitude_chart(shown, cursor=index), key="motion_attitudes",
                        width="stretch", config=PLOT_CONFIG)

    playback_view()
