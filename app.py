"""Korean Streamlit interface for the vessel stabilization study."""

import streamlit as st

from ui_data import render_data
from ui_experiments import applied_for, render_performance
from ui_optimizer import render_optimizer
from ui_scene import render_scene
from ui_settings import ModelSettings, render_model

PAGES = ("3D 시뮬레이션", "PID 설정", "AI 자동 최적화", "성능 분석")

st.set_page_config(page_title="선박 2축 수평유지 시뮬레이터", page_icon="⚓", layout="wide")
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { font-family: "Noto Sans CJK KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif; }
[data-testid="stSidebar"] { background: #EDF4F8; border-right: 1px solid #DCE8EE; }
[data-testid="stSidebar"] [data-testid="stHeading"] h2 { font-size: 1.3rem; padding-top: .1rem; }
[data-testid="stSidebar"] [data-testid="stHeading"] h3 { font-size: 1.06rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #50697E; }
[data-testid="stMainBlockContainer"] { max-width: 1460px; padding-top: 2.25rem; padding-bottom: 3rem; }
.app-head { border-bottom: 1px solid #D8E5ED; padding: .2rem 0 1.12rem; }
.app-head h1 { margin: 0; padding: 0; color: #132F46; font-size: clamp(1.65rem, 2.5vw, 2.55rem);
    letter-spacing: -.055em; line-height: 1.28; }
.app-head p { margin: .52rem 0 0; color: #526C82; font-size: .92rem; }
[data-testid="stMetric"] { background: #FFF; border: 1px solid #DCE8EF; border-radius: .6rem;
    border-top: 2px solid #62A9A3; padding: .68rem .78rem; }
[data-testid="stMetricLabel"] p { font-size: .86rem; color: #476078; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-size: 1.5rem; }
[data-testid="stPlotlyChart"] { border: 1px solid #DCE8EF; border-radius: .65rem; overflow: hidden; }
[data-testid="stSegmentedControl"] button { min-height: 2.4rem; }
[data-testid="stProgressBar"] { height: .6rem; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<header class="app-head">
    <h1>선박 2축 수평유지 시뮬레이터</h1>
    <p>선박 Roll·Pitch 관측 기록으로 상판을 제어하고 PID 이득을 자동 탐색합니다.</p>
</header>
""", unsafe_allow_html=True)


def pause_playback():
    cursor = st.session_state.get("playback")
    if cursor is not None:
        cursor.pause()


# Keep tuning and camera widgets when switching views.
for key in list(st.session_state):
    if key.startswith(("opt_", "scene_", "show_", "seek_", "performance_experiment", "motor_signal")):
        st.session_state[key] = st.session_state[key]

if "model_settings" not in st.session_state:
    st.session_state.model_settings = ModelSettings()
if request := st.session_state.pop("nav_request", None):
    st.session_state.nav = request
page = st.segmented_control("화면 선택", PAGES, default=PAGES[0], key="nav",
                            label_visibility="collapsed", on_change=pause_playback) or PAGES[0]
with st.sidebar:
    runs, sources, vessels = render_data()

model = st.session_state.model_settings
if page == "PID 설정":
    render_model(vessels["A"])
elif "A" not in runs:
    pause_playback()
    st.info("사이드바에서 운동 데이터를 불러오고 실험 A 구간을 지정해 주세요.")
else:
    applied = applied_for(runs["A"], model)
    if page == "3D 시뮬레이션":
        render_scene(runs, model, applied, st.session_state.get("sea", "보통 파도"), vessels=vessels)
    elif page == "AI 자동 최적화":
        render_optimizer(runs, model, vessel=vessels["A"].spec)
    elif page == "성능 분석":
        render_performance(runs, sources, model, applied, vessels=vessels)
