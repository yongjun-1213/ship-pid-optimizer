"""Data mapping and provenance controls for the Streamlit app."""

from io import BytesIO
from math import floor

import pandas as pd
import streamlit as st

from data_loader import (ACE, GUNNERUS, SOURCE_FILES, SOURCE_LINKS, UPLOAD, VIRTUAL,
                         ColumnMapping, MotionData, clean_motion, detect_columns,
                         read_csv, resample_window, source_bytes, validate_holdouts,
                         source_mapping)
from ship_model import SEA_STATES, virtual_motion

EXPERIMENTS = {
    "A": "A · 학습",
    "B": "B · 독립 검증",
    "C": "C · 추가 검증",
}
POSITIVE = {
    "roll": {1: "좌현이 올라감", -1: "우현이 올라감"},
    "pitch": {1: "선수가 올라감", -1: "선수가 내려감"},
}
SOURCE_OPTIONS = (GUNNERUS, ACE, UPLOAD, VIRTUAL)
SOURCE_LABELS = {
    GUNNERUS: "R/V Gunnerus · 실측",
    ACE: "R/V Akademik Tryoshnikov · 실측",
    UPLOAD: UPLOAD,
    VIRTUAL: "가상 파도 · 합성 입력",
}


@st.cache_data(show_spinner=False, max_entries=8)
def _parse_bytes(payload: bytes):
    return read_csv(payload)


@st.cache_data(show_spinner=False, max_entries=16)
def _clean(frame, mapping, source, filename, max_angle, max_gap):
    return clean_motion(frame, mapping, source=source, filename=filename,
                        max_angle=max_angle, max_gap=max_gap)


@st.cache_data(show_spinner=False, max_entries=36)
def _resample(data, segment, start, end, dt, method):
    return resample_window(data, segment, start, end, dt, method=method)


@st.cache_data(show_spinner=False, max_entries=3)
def _virtual(sea):
    return virtual_motion(sea)


def _axis_directions(key):
    values = []
    for axis, choices in POSITIVE.items():
        label = f"원본 양의 {axis.capitalize()} 방향"
        values.append(st.radio(
            label, list(choices), format_func=lambda choice, labels=choices: labels[choice],
            index=None, key=f"{key}_{axis}_sign",
        ))
    return values


def _pick(label, choices, guess, key, *, labels=None, allow_empty=True):
    return st.selectbox(label, choices, index=(choices.index(guess) if guess in choices else
                                               (None if allow_empty else 0)),
                        key=key, placeholder="열을 선택하세요",
                        format_func=(lambda value: labels.get(value, value)) if labels else str)


def _upload_mapping(frame, key) -> ColumnMapping | None:
    columns = frame.columns.tolist()
    guess = detect_columns(columns)
    st.caption("정규화 좌표계의 양의 방향은 Roll = 좌현 상승, Pitch = 선수 상승입니다.")
    time = _pick("시간 열 또는 날짜 열", columns, guess["time"], key + "_time")
    roll = _pick("Roll 열", columns, guess["roll"], key + "_roll")
    pitch = _pick("Pitch 열", columns, guess["pitch"], key + "_pitch")
    known_time = str(time).lower()
    explicit_time = ("datetime" if (guess["clock"] and time == guess["time"]) or
                     "utc" in known_time or known_time == "datetime" else
                     next((unit for unit in ("ms", "us", "s") if known_time.endswith("_" + unit)), None))
    clock_units = {"datetime": "UTC 날짜·시각", "s": "초 (s)", "ms": "밀리초 (ms)", "us": "마이크로초 (µs)"}
    time_unit = st.selectbox(
        "시간 형식 또는 단위", list(clock_units), format_func=clock_units.get,
        index=list(clock_units).index(explicit_time) if explicit_time else None,
        placeholder="시간 형식 선택", key=key + "_time_unit",
    )
    clock = None
    if time_unit == "datetime":
        clock = _pick("별도 시계 열 (선택)", [None, *columns], guess["clock"], key + "_clock",
                      labels={None: "시간 열에 날짜와 시각이 함께 있음"}, allow_empty=False)
    units = []
    for axis, name in (("Roll", roll), ("Pitch", pitch)):
        normalized = str(name).lower()
        unit = "rad" if "rad" in normalized else "deg" if ("deg" in normalized or "°" in normalized) else None
        units.append(st.selectbox(
            f"{axis} 각도 단위", ["deg", "rad"], format_func={"deg": "도 (°)", "rad": "라디안 (rad)"}.get,
            index=["deg", "rad"].index(unit) if unit else None, key=key + "_" + axis.lower() + "_unit",
            placeholder="각도 단위 선택",
        ))
    signs = _axis_directions(key)
    if not all([time, roll, pitch, time_unit, *units, *signs]):
        st.info("시간과 자세 열, 단위, 각 축의 양의 방향을 지정하면 구간을 선택할 수 있습니다.")
        return None
    return ColumnMapping(time, roll, pitch, time_unit, *units, *signs, clock)


def _quality_panel(data: MotionData, raw: bytes | None, key: str):
    q = data.quality
    count = len(data.seconds)
    st.caption(f"원본 간격 중앙값 {q.observation_dt:.4g}초 · {1 / q.observation_dt:.3g} Hz · 유효 {count:,}행")
    st.caption(f"시간 오류 {q.bad_time_rows:,}행 · 결측/범위 초과 {q.bad_angle_rows:,}행 · "
               f"중복 제거 {q.duplicate_rows:,}행 · 긴 공백 {q.split_gaps:,}개")
    if data.origin:
        st.caption("기록의 시작 시각 " + pd.Timestamp(data.origin).strftime("%Y-%m-%d %H:%M:%S.%f")[:23] + " UTC")
    with st.expander("데이터 보기·다운로드"):
        cleaned = data.cleaned_frame()
        st.dataframe(cleaned.head(10), hide_index=True, width="stretch")
        st.download_button("정제 데이터 CSV", cleaned.to_csv(index=False).encode("utf-8-sig"),
                           file_name="cleaned_attitude.csv", mime="text/csv", key=f"{key}_clean", on_click="ignore")
        if raw is not None:
            st.download_button("원본 기록 다운로드", raw, file_name=data.filename,
                               key=f"{key}_raw", on_click="ignore")
        st.caption(f"{q.max_gap:.3g}초 이하의 공백만 선택 범위 내에서 보간합니다.")


def _source_data(source: str, key: str, sea: str, *, options: bool) -> MotionData | None:
    if source == VIRTUAL:
        result = _virtual(sea)
        _quality_panel(result, None, key)
        st.caption("파도 성분에서 계산한 10 Hz 합성 자세입니다.")
        return result
    try:
        if source == UPLOAD:
            upload = st.file_uploader("자세 파일 선택", type=["csv", "gz", "txt"], key=f"{key}_upload",
                                      help="CSV, CSV.GZ 또는 쉼표로 구분된 HYDRINS 일별 TXT 파일")
            if upload is None:
                st.info("시간, Roll, Pitch가 포함된 CSV 또는 CSV.GZ 파일을 올려 주세요.")
                return None
            raw, filename = upload.getvalue(), upload.name
            frame = _parse_bytes(raw)
            with st.expander("열과 단위 지정", expanded=True):
                st.dataframe(frame.head(4), hide_index=True, width="stretch")
                mapping = _upload_mapping(frame, key)
            if mapping is None:
                return None
        else:
            raw, filename = source_bytes(source), SOURCE_FILES[source]
            frame = _parse_bytes(raw)
            st.caption(f"[관측 자료 출처]({SOURCE_LINKS[source]})")
            if source == ACE:
                st.caption("HYDRINS PC UTC 시각과 platform_roll / platform_pitch (°)를 사용합니다. "
                           "2017-01-26의 15분 발췌이며, 원본 메타데이터에서 축의 양의 방향은 확인되지 않았습니다.")
                signs = _axis_directions(key)
                if not all(signs):
                    st.info("실험에 사용할 두 축의 부호를 선택해 주세요. 선택은 결과 JSON에 기록됩니다.")
                    return None
                mapping = source_mapping(source, *signs)
            else:
                mapping = source_mapping(source)
                st.caption("Seapath 2 Hz · 2023-10-31 · Roll 양수는 좌현 상승, Pitch 양수는 선수 상승")

        angle, gap = 70.0, 0.0
        if options:
            with st.expander("결측·이상치 처리"):
                angle = st.number_input("자세 절댓값 상한 (°)", 1.0, 88.0, 70.0, 1.0, key=f"{key}_angle")
                gap = st.number_input("보간 허용 간격 (초)", 0.0, 1000.0, 0.0, 0.05,
                                      key=f"{key}_gap",
                                      help="0이면 원본 간격 중앙값의 3.05배. 초과 공백에서는 연속 구간을 분리합니다.")
        result = _clean(frame, mapping, source, filename, float(angle), float(gap) or None)
        _quality_panel(result, raw, key)
        return result
    except (ValueError, OSError, pd.errors.ParserError) as error:
        st.error(f"자세 데이터를 불러올 수 없습니다. {error}")
        return None


def _default_bounds(data, exp, lo, hi, external=False):
    start = lo
    span = hi - lo
    if data.source == GUNNERUS and lo <= 0.01 and hi >= 3_000 and not external:
        start = {"A": 0.0, "B": 1560.0, "C": 2850.0}[exp]
        return start, start + 60.0
    if external:
        return start, min(hi, start + 60.0)
    span_fraction = {"A": 0.0, "B": .40, "C": .75}[exp]
    duration = span if exp == "A" and span < data.quality.observation_dt * 6 else min(60.0, 0.25 * span)
    start = lo + span_fraction * span
    return start, min(start + duration, hi)


def _window(data: MotionData, exp: str, dt: float, method: str, external=False):
    eligible = [i for i, (lo, hi) in enumerate(data.segments)
                if hi - lo >= 2 and data.seconds[hi - 1] - data.seconds[lo] >= dt]
    if not eligible:
        st.error("제어 주기를 채울 수 있는 연속 관측 구간이 없습니다.")
        return None
    prefix = f"{exp}_{data.record_id[:14]}_{data.quality.max_gap:.6g}_{data.quality.bad_angle_rows}"
    segment = st.selectbox("연속 구간", eligible, key=f"{prefix}_segment",
                           format_func=lambda i: f"{i + 1}번 · {data.seconds[data.segments[i][0]]:.3f}–"
                                                 f"{data.seconds[data.segments[i][1] - 1]:.3f}초",
                           disabled=len(eligible) == 1)
    a, b = data.segments[segment]
    lo, hi = float(data.seconds[a]), float(data.seconds[b - 1])
    start0, end0 = _default_bounds(data, exp, lo, hi, external=external)
    key = f"{prefix}_{segment}_{lo:.6f}_{hi:.6f}"
    f = "%.3f" if data.quality.observation_dt >= 0.001 else "%.6f"
    scale = max(0.001, min(0.5, data.quality.observation_dt))
    left, right = st.columns(2)
    start = left.number_input("시작 (초)", min_value=lo, max_value=hi, value=start0,
                              step=scale, format=f, key=f"{key}_start")
    end = right.number_input("종료 (초)", min_value=lo, max_value=hi, value=end0,
                              step=scale, format=f, key=f"{key}_end")
    try:
        result = _resample(data, segment, start, end, dt, method)
        st.caption(f"원본 {result.window.observations:,}행 → 제어 {len(result.time_s):,}점 · "
                   f"계산 길이 {result.time_s[-1]:.3f}초")
        return result
    except ValueError as error:
        st.warning(str(error))
        return None


def render_data():
    st.header("선박 운동 데이터")
    source = st.selectbox("자세 입력", SOURCE_OPTIONS, format_func=SOURCE_LABELS.get, key="source_picker")
    sea = st.selectbox("해상 시나리오" if source == VIRTUAL else "3D 가상 수면",
                       list(SEA_STATES), index=1, key="sea")
    data = _source_data(source, "primary_" + str(SOURCE_OPTIONS.index(source)), sea, options=True)
    if data is None:
        return {}, {}
    with st.expander("제어 주기와 보간"):
        dt = float(st.number_input("제어 주기 (초)", 0.005, 0.2, 0.02, 0.005, format="%.3f"))
        method = st.selectbox("보간", ("pchip", "linear"), format_func={"pchip": "PCHIP", "linear": "선형"}.get)
        st.caption("보간은 제어 시각에 맞춰 값을 계산합니다. 원본 관측 주파수와 정보량은 그대로입니다.")
    st.subheader("실험 구간")
    st.caption("시작·종료 값은 각 기록의 시작 시각 기준입니다. 구간에 포함된 원본 관측점만으로 보간합니다.")
    runs, sources = {}, {}
    with st.expander(EXPERIMENTS["A"], expanded=True):
        if train := _window(data, "A", dt, method):
            runs["A"], sources["A"] = train, data
    enough = len(data.seconds) >= 8
    for exp in ("B", "C"):
        with st.expander(EXPERIMENTS[exp]):
            enabled = st.checkbox("이 검증 사용", value=enough, key=f"enable_{exp}_{data.record_id[:14]}")
            if not enabled:
                continue
            candidate = data
            if exp == "C":
                labels = {"local": "같은 기록의 다른 구간", **SOURCE_LABELS}
                choice = st.radio("추가 검증 입력", ("local", *(name for name in (ACE, GUNNERUS) if name != source)),
                                      format_func=labels.get, key=f"external_{data.record_id[:14]}")
                if choice != "local":
                    candidate = _source_data(choice, "extra_" + str(SOURCE_OPTIONS.index(choice)), sea, options=True)
            if candidate is None:
                continue
            window = _window(candidate, exp, dt, method, external=candidate is not data)
            if window and "A" in runs:
                try:
                    validate_holdouts(runs["A"].window, *(run.window for code, run in runs.items() if code != "A"),
                                      window.window)
                    runs[exp], sources[exp] = window, candidate
                except ValueError as error:
                    st.error(str(error))
    return runs, sources
