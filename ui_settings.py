"""Model settings and JSON import/export for the Streamlit app."""

from dataclasses import asdict, dataclass
import json

import streamlit as st

from motor_model import MotorAxis, MotorParams
from pid_controller import Gains, PIDSettings, TwoAxisPID
from simulation import SensorParams


@dataclass(frozen=True)
class ModelSettings:
    manual: TwoAxisPID = TwoAxisPID()
    motor: MotorParams = MotorParams()
    sensor: SensorParams = SensorParams()
    pid: PIDSettings = PIDSettings()
    enabled: bool = True

    def to_bytes(self):
        return json.dumps({"schema": 1, **asdict(self)}, ensure_ascii=False,
                          indent=2, allow_nan=False).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload):
        try:
            data = json.loads(payload)
            if data["schema"] != 1 or not isinstance(data["enabled"], bool):
                raise ValueError("설정 파일 버전을 확인해 주세요.")
            m = data["motor"]
            return cls(
                TwoAxisPID(**{axis: Gains(**values) for axis, values in data["manual"].items()}),
                MotorParams(m["payload_kg"], MotorAxis(**m["roll"]), MotorAxis(**m["pitch"])),
                SensorParams(**data["sensor"]), PIDSettings(**data["pid"]), data["enabled"],
            )
        except (KeyError, TypeError, OverflowError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError("이 앱에서 저장한 PID·모터 설정 JSON을 선택해 주세요.") from error


def _number(label, default, step, key, *, minimum=0.0, maximum=None, help=None):
    return float(st.number_input(label, min_value=minimum, max_value=maximum, value=float(default),
                                 step=step, format="%.4f", key=key, help=help))


def _axis_motor(axis: MotorAxis, key):
    entries = (
        ("bare_inertia_kg_m2", "장치 자체 관성 (kg·m²)", .005, .0001),
        ("payload_radius_m", "적재물 유효 회전 반경 (m)", .005, 0.0),
        ("damping_nm_s_rad", "점성 감쇠 (N·m·s/rad)", .01, 0.0),
        ("torque_nm", "최대 구동 토크 (N·m)", .1, .001),
        ("delay_s", "명령 전달 지연 (초)", .005, 0.0),
        ("response_s", "토크 응답 시간상수 (초)", .005, 0.0),
        ("max_angle_deg", "최대 상대 회전각 (±°)", 1.0, .1),
        ("max_rate_deg_s", "최대 상대 각속도 (°/s)", 5.0, .1),
    )
    values = {}
    for field, label, step, minimum in entries:
        maximum = 89.9 if field == "max_angle_deg" else 10.0 if field == "delay_s" else None
        values[field] = _number(label, getattr(axis, field), step, f"{key}_{field}", minimum=minimum,
                                maximum=maximum)
    return MotorAxis(**values)


def _gain_values(gains: Gains, key: str):
    labels = ("Kp (N·m/rad)", "Ki (N·m/(rad·s))", "Kd (N·m·s/rad)")
    return Gains(*(_number(label, value, 0.05, f"{key}_{i}")
                   for i, (label, value) in enumerate(zip(labels, asdict(gains).values()))))


def render_model():
    model = st.session_state.model_settings
    rev = st.session_state.get("model_form_revision", 0)
    prefix = f"model_{rev}"
    st.subheader("PID 제어 설정")
    st.caption("Roll과 Pitch는 상판 추정각을 0°로 제어합니다. 각 PID는 모터 토크를 명령합니다.")
    with st.form(f"{prefix}_form", border=True):
        enabled = st.toggle("시뮬레이터 제어 활성화", value=model.enabled, key=f"{prefix}_on")
        left, right = st.columns(2)
        with left:
            st.markdown("**Roll PID**")
            roll = _gain_values(model.manual.roll, prefix + "_roll")
        with right:
            st.markdown("**Pitch PID**")
            pitch = _gain_values(model.manual.pitch, prefix + "_pitch")
        with st.expander("적분 제한과 미분 필터"):
            a, b = st.columns(2)
            with a:
                derivative = _number("미분 필터 시간상수 (초)", model.pid.derivative_tau_s, .01, prefix + "_d")
            with b:
                integral = _number("적분 토크 상한 / 최대 토크", model.pid.integral_limit_fraction,
                                   .05, prefix + "_i", maximum=1.0)
            st.caption("포화 또는 기계 한계 쪽으로 명령이 더 증가하면 적분을 잠급니다.")
        with st.expander("모터 · 가상 모델 파라미터"):
            mass = _number("적재물 질량 (kg)", model.motor.payload_kg, .1, prefix + "_mass")
            l, r = st.columns(2)
            with l:
                st.markdown("**Roll 모터**")
                mr = _axis_motor(model.motor.roll, prefix + "_mr")
            with r:
                st.markdown("**Pitch 모터**")
                mp = _axis_motor(model.motor.pitch, prefix + "_mp")
            st.caption(f"합성 관성 Roll {mr.inertia(mass):.5f}, Pitch {mp.inertia(mass):.5f} kg·m². "
                       "장치 관성에 적재물 질량 × 유효 회전 반경²을 더합니다.")
        with st.expander("모의 IMU"):
            a, b = st.columns(2)
            with a:
                angle = _number("중력 기울기 잡음 표준편차 (°)", model.sensor.angle_noise_deg,
                                .05, prefix + "_an")
                gyro = _number("자이로 잡음 표준편차 (°/s)", model.sensor.gyro_noise_deg_s,
                               .05, prefix + "_gn")
            with b:
                bias = _number("자이로 바이어스 표준편차 (°/s)", model.sensor.gyro_bias_deg_s,
                               .05, prefix + "_bias")
                fusion = _number("상보필터 시간상수 (초)", model.sensor.fusion_tau_s,
                                 .05, prefix + "_fusion")
            seed = int(st.number_input("IMU 난수 시드", 0, 2**32 - 1, model.sensor.seed, 1, key=prefix + "_seed"))
            st.caption("가상 상판 IMU에 중력 기울기와 자이로 오차를 모의합니다.")
        submit = st.form_submit_button("PID와 장치 설정 적용", type="primary")
    if submit:
        try:
            settings = ModelSettings(TwoAxisPID(roll, pitch), MotorParams(mass, mr, mp),
                                     SensorParams(angle, gyro, bias, fusion, seed),
                                     PIDSettings(derivative, integral), enabled)
            st.session_state.model_settings = settings
            st.success("설정을 적용했습니다. 새 계산은 각 실험 구간의 0초부터 시작합니다.")
            model = settings
        except ValueError as error:
            st.error(str(error))
    with st.expander("설정 JSON 저장 또는 불러오기"):
        st.download_button("현재 설정 JSON", model.to_bytes(), file_name="pid_model_settings.json",
                           mime="application/json", on_click="ignore")
        saved = st.file_uploader("PID·모터 설정 파일", type=["json"], key="model_import")
        if saved is not None and st.button("파일 설정 적용"):
            try:
                st.session_state.model_settings = ModelSettings.from_bytes(saved.getvalue())
                st.session_state.model_form_revision = rev + 1
                st.rerun()
            except ValueError as error:
                st.error(str(error))
