"""Streamlit controls for the background TPE worker."""

import pandas as pd
import streamlit as st

from evaluation import ObjectiveWeights
from optimizer import MAX_TRIALS, OptimizationJob, StudyPlan, decode_pid, default_bounds
from ui_data import EXPERIMENTS
from ui_experiments import AppliedPID, PLOT_CONFIG, comparison, data_summary, model_identity, plan_identity
from visualization import AI_PID, MANUAL_PID, OFF, optimization_chart

STATUS = {"ready": "대기", "running": "실험 진행 중", "stopping": "현재 실험 마무리 중",
          "paused": "중단됨", "finished": "완료", "failed": "오류"}
WEIGHTS = {
    "roll": "Roll RMS", "pitch": "Pitch RMS", "peak": "최대 전체 경사",
    "effort": "토크·속도 사용량", "vibration": "진동 지표", "constraints": "물리 한계 도달",
}
GAIN_UNITS = {"kp": "N·m/rad", "ki": "N·m/(rad·s)", "kd": "N·m·s/rad"}


def _label(name):
    field, _, axis = name.partition("_")
    return (axis.capitalize() + " " if axis else "") + field.capitalize()


def _study_parameters(train, model):
    a, b, c = st.columns([1.2, 1, 1])
    mode = a.selectbox("탐색 방식", [3, 6], format_func={3: "3변수 · 축 공통", 6: "6변수 · 축별 독립"}.get,
                       key="opt_mode")
    total = b.selectbox("실험 횟수", [50, 100, 300, 500, "직접 입력"], index=1, key="opt_trials")
    seed = c.number_input("Optuna 난수 시드", 0, 2**32 - 1, 7, key="opt_seed")
    if total == "직접 입력":
        total = b.number_input("총 실험 횟수", 1, MAX_TRIALS, 150, 1, key="opt_trials_custom")
    bounds = {}
    with st.expander("PID 탐색 범위"):
        left, right = st.columns(2)
        for name, (lower, upper) in default_bounds(mode).items():
            base = name.split("_")[0]
            lo = left.number_input(f"{_label(name)} 하한", min_value=0.0, value=lower, step=.1,
                                   format="%.4f", key=f"opt_lo_{name}", help=GAIN_UNITS[base])
            hi = right.number_input(f"{_label(name)} 상한", min_value=0.0, value=upper, step=.1,
                                    format="%.4f", key=f"opt_hi_{name}", help=GAIN_UNITS[base])
            bounds[name] = (float(lo), float(hi))
    weights = {}
    with st.expander("목적함수 가중치"):
        defaults = vars(ObjectiveWeights())
        left, right = st.columns(2)
        for index, (key, label) in enumerate(WEIGHTS.items()):
            column = left if index % 2 == 0 else right
            weights[key] = float(column.number_input(label, min_value=0.0, value=defaults[key], step=.01,
                                                     format="%.3f", key=f"opt_weight_{key}"))
        st.caption("각 RMS는 1°, 최대 전체 경사는 5°를 기준으로 나눕니다. 구동량은 정격 토크·속도 대비 "
                   "비율, 진동은 각가속도 RMS, 제약은 포화 및 각도·속도 한계에 닿은 스텝 비율입니다.")
    try:
        return StudyPlan(train, model.motor, model.sensor, model.pid, mode, bounds,
                         ObjectiveWeights(**weights), int(seed)), int(total)
    except ValueError as error:
        st.warning(str(error))
        return None, int(total)


def _validation_summary(runs, model, best_pid):
    rows = []
    for exp, motion in runs.items():
        if exp == "A":
            continue
        results = comparison(motion, model.manual, model.motor, model.sensor, model.pid, best_pid)
        for label in (OFF, MANUAL_PID, AI_PID):
            metrics = results[label].metrics
            rows.append({
                "구간": EXPERIMENTS[exp], "입력": motion.window.source, "제어": label,
                "Roll RMS (°)": metrics.roll.rms_deg, "Pitch RMS (°)": metrics.pitch.rms_deg,
                "±1° 이내 (%)": metrics.within_one_deg_pct,
            })
    if rows:
        st.markdown("**학습에 사용하지 않은 구간의 확인 결과**")
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")


def _progress(job: OptimizationJob, runs, model):
    train = runs["A"]
    live = job.snapshot().running

    @st.fragment(run_every=0.45 if live else None)
    def progress():
        snapshot = job.snapshot()
        if live and not snapshot.running:
            st.rerun()
        best = snapshot.best
        m1, m2, m3 = st.columns(3)
        m1.metric("완료한 실험", f"{len(snapshot.records):,} / {snapshot.target:,}")
        m2.metric("최저 J", f"{best.score:.5f}" if best else "—")
        m3.metric("소요 시간", f"{snapshot.elapsed_s:.1f}초")
        st.progress(len(snapshot.records) / max(1, snapshot.target), text=STATUS[snapshot.status])
        if snapshot.running:
            if st.button("최적화 중단", key="stop_study", disabled=snapshot.status == "stopping"):
                job.stop()
                st.rerun()
        if snapshot.error:
            st.error(snapshot.error)
        if snapshot.save_error:
            st.warning("서버에 자동 저장할 수 없습니다. 체크포인트 JSON을 다운로드할 수 있습니다.")
        if best:
            pid = decode_pid(best.params, job.plan.mode)
            df = pd.DataFrame([{"축": name, "Kp": g.kp, "Ki": g.ki, "Kd": g.kd}
                               for name, g in [("Roll", pid.roll), ("Pitch", pid.pitch)]])
            st.markdown(f"**가장 낮은 점수를 기록한 이득 · 실험 {best.number:,}회차**")
            st.dataframe(df, hide_index=True, width="stretch")
            compatible = plan_identity(job.plan) == model_identity(train, model)
            apply, view, download = st.columns(3)
            if apply.button("최적 PID 적용", type="primary", disabled=not compatible, width="stretch"):
                st.session_state.applied_pid = AppliedPID.from_trial(
                    job.plan, best, vessel_metadata=job.vessel_metadata,
                )
                st.session_state.scene_request = AI_PID
                st.rerun()
            applied = st.session_state.get("applied_pid")
            is_applied = compatible and applied and applied.study_id == job.plan.signature()
            if view.button("3D에서 보기", disabled=not is_applied, width="stretch"):
                st.session_state.nav_request = "3D 시뮬레이션"
                st.rerun()
            download.download_button("체크포인트 JSON", job.checkpoint(), mime="application/json",
                                     file_name="pid_tpe_checkpoint.json", on_click="ignore", width="stretch")
            if is_applied:
                st.caption(f"실험 {applied.trial_number:,}회차 이득을 적용했습니다. 성능 분석에서 검증 구간을 선택할 수 있습니다.")
            if not compatible:
                st.warning("이 작업을 시작할 때 고정한 학습 구간이나 장치 설정과 현재 설정이 다릅니다.")
            st.plotly_chart(optimization_chart(snapshot.records), config=PLOT_CONFIG,
                            width="stretch", key="optimizer_record_plot")
            if compatible and not snapshot.running:
                _validation_summary(runs, model, pid)

    progress()


def render_optimizer(runs, model, *, vessel=None):
    train = runs["A"]
    st.subheader("AI 기반 PID 자동 최적화")
    st.caption("Optuna TPE가 A 구간을 반복 계산하고 목적함수 J가 가장 낮은 조합을 찾습니다.")
    for key, motion in runs.items():
        data_summary(motion, name=EXPERIMENTS[key])
    if "B" not in runs:
        st.warning("현재 독립 검증 B 구간이 없습니다. 사이드바에서 학습과 겹치지 않는 구간을 지정할 수 있습니다.")
    plan, total = _study_parameters(train, model)
    job = st.session_state.get("optimization_job")
    snapshot = job.snapshot() if job else None
    busy = snapshot is not None and snapshot.running
    same_plan = plan is not None and job is not None and job.plan.signature() == plan.signature()
    start_col, resume_col = st.columns(2)
    with start_col:
        if st.button("최적화 시작", key="start_study", type="primary", disabled=busy or plan is None,
                     width="stretch"):
            new_job = OptimizationJob(plan, vessel_metadata=vessel.manifest() if vessel is not None else None)
            new_job.start(total)
            st.session_state.optimization_job = new_job
            st.rerun()
    with resume_col:
        if st.button("기록에서 이어서 실행", disabled=(busy or not same_plan or len(snapshot.records) >= total),
                     width="stretch"):
            job.start(total)
            st.rerun()
    with st.expander("체크포인트 불러오기"):
        checkpoint = st.file_uploader("저장한 실험 기록 JSON", type=["json"], key="optimizer_checkpoint")
        if st.button("실험 기록 복원", disabled=busy or plan is None or checkpoint is None):
            try:
                st.session_state.optimization_job = OptimizationJob.restore(plan, checkpoint.getvalue())
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        st.caption("복원에는 같은 학습 구간, 데이터 매핑, 장치·센서 설정, 탐색 범위, 가중치와 시드가 필요합니다. "
                   "이어 하려면 완료 횟수보다 큰 목표 횟수를 선택하세요.")
    if job is not None:
        _progress(job, runs, model)
