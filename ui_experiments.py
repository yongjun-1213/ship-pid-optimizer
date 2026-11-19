"""Compare experiments and export the underlying performance data."""

from dataclasses import asdict, dataclass
import json

import pandas as pd
import streamlit as st

from data_loader import MotionInput
from evaluation import ObjectiveWeights, objective_score, objective_terms, rms_reduction
from optimizer import StudyPlan, TrialRecord, decode_pid
from pid_controller import TwoAxisPID
from simulation import prepare_simulation, simulate
from ui_data import EXPERIMENTS
from ui_settings import ModelSettings
from visualization import AI_PID, AXES, MANUAL_PID, OFF, attitude_chart, motor_chart

PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": True}


@dataclass(frozen=True)
class AppliedPID:
    study_id: str
    training: dict
    trial_number: int
    score: float
    gains: TwoAxisPID
    weights: ObjectiveWeights
    identity: tuple

    @classmethod
    def from_trial(cls, plan: StudyPlan, trial: TrialRecord):
        return cls(plan.signature(), plan.metadata(), trial.number, trial.score,
                   decode_pid(trial.params, plan.mode), plan.weights, plan_identity(plan))


def plan_identity(plan: StudyPlan):
    return plan.motion.digest, plan.motor, plan.sensor, plan.pid


def model_identity(motion: MotionInput, model: ModelSettings):
    return motion.digest, model.motor, model.sensor, model.pid


def applied_for(train: MotionInput | None, model: ModelSettings):
    applied = st.session_state.get("applied_pid")
    return applied if train is not None and applied and applied.identity == model_identity(train, model) else None


@st.cache_data(show_spinner=False, max_entries=18)
def comparison(motion, manual, motor, sensor, pid, ai):
    prepared = prepare_simulation(motion, sensor)
    results = {
        OFF: simulate(manual, prepared, motor, pid_settings=pid, controller_on=False),
        MANUAL_PID: simulate(manual, prepared, motor, pid_settings=pid),
    }
    if ai is not None:
        results[AI_PID] = simulate(ai, prepared, motor, pid_settings=pid)
    return results


def experiment_picker(runs, key):
    if key in st.session_state and st.session_state[key] not in runs:
        st.session_state[key] = next(iter(runs))
    return st.selectbox("실험 구간", list(runs), format_func=EXPERIMENTS.get, key=key)


def data_summary(motion: MotionInput, *, name: str | None = None):
    window = motion.window
    manifest = window.manifest()
    times = (f"{manifest['start_utc'][:23].replace('T', ' ')} ~ "
             f"{manifest['end_utc'][:23].replace('T', ' ')} UTC") if window.origin else (
                 f"기록의 {window.observed_start:.3f}–{window.observed_end:.3f}초"
             )
    lead = (f"{name} · " if name else "") + window.source + " · " + times
    st.caption(f"{lead} · 계산 {motion.time_s[-1]:.3f}초")


def _metric_table(results):
    rows = []
    off, manual = results[OFF].metrics, results[MANUAL_PID].metrics
    for label, result in results.items():
        for axis in AXES:
            attr = axis.lower()
            m = getattr(result.metrics, attr)
            a = getattr(off, attr)
            u = getattr(manual, attr)
            rows.append({
                "비교": label, "축": axis, "RMS (°)": m.rms_deg,
                "평균 절대 오차 (°)": m.mae_deg, "최대 오차 (°)": m.peak_deg,
                "OFF 대비 RMS 감소 (%)": rms_reduction(a.rms_deg, m.rms_deg),
                "수동 대비 RMS 감소 (%)": rms_reduction(u.rms_deg, m.rms_deg) if label == AI_PID else None,
                "모터 누적 회전 (°)": m.travel_deg, "모터 구동량 (J)": m.absolute_work_j,
            })
    return pd.DataFrame(rows).round(3)


def _constraint_table(results):
    rows = []
    for label, result in results.items():
        for axis in AXES:
            m = getattr(result.metrics, axis.lower())
            rows.append({
                "비교": label, "축": axis,
                "토크 포화 횟수": m.torque_saturation_events,
                "각속도 제한 횟수": m.rate_limit_events,
                "회전각 제한 횟수": m.angle_limit_events,
                "토크 포화 스텝": m.torque_saturation_samples,
                "각속도 제한 스텝": m.rate_limit_samples,
                "회전각 제한 스텝": m.angle_limit_samples,
                "물리 한계 초과 스텝": m.violation_samples,
                "최대 적용 토크 (N·m)": round(m.torque_peak_nm, 4),
                "최대 각속도 (°/s)": round(m.rate_peak_deg_s, 3),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=8)
def _export_csv(results):
    frames = [run.frame().assign(controller=label) for label, run in results.items()]
    merged = pd.concat(frames, ignore_index=True)
    return merged.to_csv(index=False).encode("utf-8-sig")


def _export_json(exp, motion, data, model, results, applied):
    weights = applied.weights if applied else ObjectiveWeights()
    payload = {
        "schema": 1,
        "experiment": exp,
        "window": motion.window.manifest(),
        "simulation_duration_s": float(motion.time_s[-1]),
        "motion_kind": "synthetic" if data.wave else "attitude_record",
        "source_file": data.filename,
        "data_quality": asdict(data.quality),
        "dt_s": motion.dt,
        "interpolation": motion.interpolation,
        "mapping": asdict(motion.mapping) if motion.mapping else None,
        "model": asdict(model),
        "objective_weights": asdict(weights),
        "applied_optimization": {
            "training": applied.training, "trial": applied.trial_number,
            "training_score": applied.score, "pid_gains": asdict(applied.gains)
        } if applied else None,
        "results": {},
    }
    for label, result in results.items():
        terms = objective_terms(result.metrics, model.motor)
        payload["results"][label] = {"metrics": result.metrics.to_dict(),
                                     "objective_terms": terms, "objective": objective_score(terms, weights)}
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def render_performance(runs, sources, model, applied):
    st.subheader("성능 분석")
    exp = experiment_picker(runs, "performance_experiment")
    motion = runs[exp]
    data_summary(motion)
    if exp != "A":
        st.caption("이 성능 평가는 학습 구간에 포함되지 않은 원본 관측점으로 계산합니다." if not motion.wave
                   else "선택한 구간은 학습 구간과 분리된 합성 입력입니다.")
    if applied is None:
        st.info("자동 최적화 화면에서 최적 PID를 적용하면 이 구간의 결과에 추가됩니다.")
    results = comparison(motion, model.manual, model.motor, model.sensor, model.pid,
                         applied.gains if applied else None)
    weights = applied.weights if applied else ObjectiveWeights()
    panels = st.columns(len(results))
    for column, (label, result) in zip(panels, results.items()):
        with column:
            st.metric(f"{label} · ±1° 이내", f"{result.metrics.within_one_deg_pct:.1f}%",
                      help="Roll과 Pitch 절댓값이 모두 1° 이하인 스텝의 비율입니다.")
            terms = objective_terms(result.metrics, model.motor)
            st.caption(f"최대 상판 경사 {result.metrics.peak_tilt_deg:.3f}° · J {objective_score(terms, weights):.4f}")
    st.plotly_chart(attitude_chart(results), config=PLOT_CONFIG, width="stretch", key="performance_attitudes")
    st.markdown("**상판 오차와 모터 구동량**")
    st.dataframe(_metric_table(results), hide_index=True, width="stretch")
    st.caption("제어 OFF 상판 각도는 선박 각도와 같습니다. 구동량은 적용 토크 × 각속도의 절댓값을 시간 적분한 값입니다.")
    st.markdown("**모터 명령과 응답**")
    signal = st.segmented_control("그래프 항목", ["토크", "회전각", "각속도"],
                                  default="토크", key="motor_signal") or "토크"
    st.plotly_chart(motor_chart(results, signal), config=PLOT_CONFIG, width="stretch", key=f"motor_chart_{signal}")
    with st.expander("모터 제한과 포화 기록"):
        st.dataframe(_constraint_table(results), hide_index=True, width="stretch")
        st.caption("횟수는 한 축에서 연속된 제한 구간을 한 번으로 셉니다. 물리 한계 초과 스텝은 제한 적용 후의 실제 상태를 검사합니다.")
    st.markdown("**결과 다운로드**")
    x, y = st.columns(2)
    x.download_button("시계열 CSV", _export_csv(results), file_name=f"stabilizer_{exp}_timeseries.csv",
                      mime="text/csv", on_click="ignore", width="stretch")
    y.download_button("성능·설정 JSON", _export_json(exp, motion, sources[exp], model, results, applied),
                      file_name=f"stabilizer_{exp}_report.json", mime="application/json",
                      on_click="ignore", width="stretch")
