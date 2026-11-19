"""Smoke tests for the app's real entry point and Streamlit state transitions."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _select(app, label):
    return next(widget for widget in app.selectbox if widget.label == label)


def _radio(app, label):
    return next(widget for widget in app.radio if widget.label == label)


def _metric(app, label):
    return next(widget for widget in app.metric if widget.label == label)


def test_streamlit_entrypoint_replay_and_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("STABILIZER_RUN_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert _metric(app, "실험 시간 (초)").value == "0.000"
    _button(app, "한 스텝").click().run()
    assert _metric(app, "실험 시간 (초)").value == "0.020"
    app.run()
    assert _metric(app, "실험 시간 (초)").value == "0.020"
    _button(app, "정지").click().run()
    assert _metric(app, "실험 시간 (초)").value == "0.000"
    app.button_group(key="nav").set_value("AI 자동 최적화").run()
    _select(app, "실험 횟수").set_value("직접 입력").run()
    app.number_input(key="opt_trials_custom").set_value(12).run()
    _button(app, "최적화 시작").click().run()
    app.session_state.optimization_job.wait(12)
    app.run()
    assert _metric(app, "완료한 실험").value == "12 / 12"
    assert not app.exception
    _button(app, "최적 PID 적용").click().run()
    _button(app, "3D에서 보기").click().run()
    assert _select(app, "상판 제어").value == "AI 최적 PID"
    _button(app, "한 스텝").click().run()
    before = tuple(_metric(app, label).value for label in ("실험 시간 (초)", "선박 Roll", "선박 Pitch", "상판 Roll"))
    original = app.session_state.optimization_job.plan
    next(box for box in app.checkbox if box.label == "3D 제원 직접 입력").check().run()
    next(item for item in app.number_input if item.label == "선박 길이 (m)").set_value(72.5)
    _button(app, "3D 제원 적용").click().run()
    after = tuple(_metric(app, label).value for label in ("실험 시간 (초)", "선박 Roll", "선박 Pitch", "상판 Roll"))
    assert after == before
    assert any("72.50 m" in element.value for element in app.caption)
    assert app.session_state.optimization_job.plan is original
    assert _select(app, "상판 제어").value == "AI 최적 PID"
    app.button_group(key="nav").set_value("성능 분석").run()
    _select(app, "실험 구간").set_value("B").run()
    assert _metric(app, "AI 최적 PID · ±1° 이내").value.endswith("%")
    assert len(app.get("download_button")) >= 2
    assert not app.exception


def test_dropdown_for_ace_and_virtual_modes():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.selectbox(key="source_picker").set_value("R/V Akademik Tryoshnikov").run()
    assert not app.metric
    assert any("133.53 m" in element.value for element in app.caption)
    _radio(app, "원본 양의 Roll 방향").set_value(1).run()
    _radio(app, "원본 양의 Pitch 방향").set_value(-1).run()
    assert _metric(app, "선박 Roll").value == "-1.83°"
    assert _metric(app, "선박 Pitch").value == "+1.40°"
    assert not app.exception
    app.selectbox(key="source_picker").set_value("가상 파도").run()
    assert _metric(app, "실험 시간 (초)").value == "0.000"
    assert not app.exception


def test_other_vessel_selected_as_independent_experiment():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.button_group(key="nav").set_value("성능 분석").run()
    _radio(app, "추가 검증 입력").set_value("R/V Akademik Tryoshnikov").run()
    _radio(app, "원본 양의 Roll 방향").set_value(1).run()
    _radio(app, "원본 양의 Pitch 방향").set_value(-1).run()
    _select(app, "실험 구간").set_value("C").run()
    assert any("2017-01-26" in element.value for element in app.caption)
    assert len(app.get("download_button")) >= 2
    assert not app.exception
