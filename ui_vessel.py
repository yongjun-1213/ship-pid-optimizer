"""Sidebar vessel particulars and scene-only user dimensions."""

from dataclasses import dataclass
from hashlib import sha256

import streamlit as st

from data_loader import UPLOAD
from vessel_specs import DEFAULT_VESSEL, VESSEL_KINDS, VESSEL_SOURCES, VESSELS, VesselSpec


@dataclass(frozen=True)
class VesselSelection:
    spec: VesselSpec
    reference: VesselSpec
    settings_key: str | None = None

    def apply_import(self, vessel: VesselSpec):
        if self.settings_key is not None:
            st.session_state[self.settings_key] = vessel
            revision = f"{self.settings_key}_revision"
            st.session_state[revision] = st.session_state.get(revision, 0) + 1


def default_vessel_for(source: str) -> VesselSelection:
    spec = VESSELS.get(source, DEFAULT_VESSEL)
    return VesselSelection(spec, spec)


def dimensions_text(vessel: VesselSpec) -> str:
    return (f"{vessel.length_label} {vessel.length_m:,.2f} m · {vessel.beam_label} {vessel.beam_m:,.2f} m"
            f" · {vessel.draft_label} {vessel.draft_m:,.2f} m")


def show_particulars(vessel: VesselSpec):
    st.text(vessel.name)
    st.caption(VESSEL_KINDS[vessel.kind])
    st.caption(dimensions_text(vessel))
    if vessel.displacement_t is not None:
        st.caption(f"{vessel.displacement_label} {vessel.displacement_t:,.2f}".rstrip("0").rstrip(".") + " t")
    if vessel.height_above_water_m is not None:
        st.caption(f"수면 위 높이 {vessel.height_above_water_m:,.2f} m")
    if vessel.provenance == "published":
        st.caption(" · ".join(f"[{label}]({url})" for label, url in VESSEL_SOURCES[vessel.name]))


def _dimensions_form(key: str, defaults: VesselSpec | None, reference: VesselSpec):
    with st.form(key, border=False):
        name = st.text_input("선박 이름", value=defaults.name if defaults else "", key=f"{key}_name")
        kinds = list(VESSEL_KINDS)
        kind = st.selectbox("선박 종류", kinds, format_func=VESSEL_KINDS.get,
                            index=kinds.index(defaults.kind) if defaults else 2, key=f"{key}_kind")
        length = st.number_input("선박 길이 (m)", min_value=0.01, step=0.5, format="%.2f",
                                 value=defaults.length_m if defaults else None, key=f"{key}_length")
        beam = st.number_input("선박 폭 (m)", min_value=0.01, step=0.5, format="%.2f",
                               value=defaults.beam_m if defaults else None, key=f"{key}_beam")
        draft = st.number_input("선박 흘수 (m)", min_value=0.01, step=0.1, format="%.2f",
                                value=defaults.draft_m if defaults else None, key=f"{key}_draft")
        height = st.number_input("수면 위 선박 높이 (선택, m)", min_value=0.01, step=0.5, format="%.2f",
                                 value=defaults.height_above_water_m if defaults else None, key=f"{key}_height")
        submit = st.form_submit_button("3D 제원 적용", width="stretch")
    if not submit:
        return None
    if not name.strip() or any(value is None for value in (length, beam, draft)):
        st.error("선박 이름, 길이, 폭, 흘수를 입력해 주세요.")
        return None
    try:
        return VesselSpec(name.strip(), kind, length, beam, draft, height,
                          hull_variant=reference.hull_variant if kind == "research" else "standard")
    except ValueError as error:
        st.error(str(error))
        return None


def render_vessel_info(source: str, upload=None) -> VesselSelection:
    reference = VESSELS.get(source, DEFAULT_VESSEL)
    with st.expander("선박 정보 및 제원", expanded=True):
        if source == UPLOAD and upload is None:
            st.caption("운동 파일을 선택하면 선박 제원을 직접 입력할 수 있습니다.")
            return VesselSelection(reference, reference)
        scope = sha256(upload.getbuffer()).hexdigest() if source == UPLOAD else source
        settings_key = f"vessel_saved_{scope}"
        revision = st.session_state.get(f"{settings_key}_revision", 0)
        form_key = f"{settings_key}_{revision}"
        saved = st.session_state.get(settings_key)
        if source != UPLOAD:
            show_particulars(reference)
            edit = st.checkbox("3D 제원 직접 입력", value=saved is not None, key=f"{form_key}_edit")
            if not edit:
                return VesselSelection(reference, reference, settings_key)
        elif saved is None:
            st.caption("제원을 입력하면 해당 비율로 표시합니다. 입력 전에는 기본 모형을 사용합니다.")
        if updated := _dimensions_form(form_key, saved or (reference if source != UPLOAD else None), reference):
            st.session_state[settings_key] = saved = updated
        if saved is not None:
            st.caption(f"3D 적용 제원 · {VESSEL_KINDS[saved.kind]}")
            st.caption(dimensions_text(saved))
        return VesselSelection(saved or reference, reference, settings_key)
