"""Ship dimensions for the scene and experiment records."""

from dataclasses import asdict, dataclass
from math import isfinite

VESSEL_KINDS = {"research": "연구선", "cargo": "화물선", "other": "기타 선박"}


@dataclass(frozen=True)
class VesselSpec:
    name: str
    kind: str
    length_m: float
    beam_m: float
    draft_m: float
    height_above_water_m: float | None = None
    depth_m: float | None = None
    displacement_t: float | None = None
    provenance: str = "user"
    length_label: str = "입력 길이"
    beam_label: str = "선폭"
    draft_label: str = "흘수"
    displacement_label: str = "배수량"
    hull_variant: str = "standard"

    def __post_init__(self):
        if not self.name.strip() or self.kind not in VESSEL_KINDS:
            raise ValueError("선박 이름과 종류를 확인해 주세요.")
        required = (self.length_m, self.beam_m, self.draft_m)
        optional = (self.height_above_water_m, self.depth_m)
        if (not all(value is not None and isfinite(value) and value > 0 for value in required)
                or any(value is not None and (not isfinite(value) or value <= 0) for value in optional)):
            raise ValueError("선박 길이, 선폭, 흘수, 높이는 양수(m)로 입력해 주세요.")
        if self.depth_m is not None and self.depth_m <= self.draft_m:
            raise ValueError("선체 깊이는 흘수보다 커야 합니다.")

    @property
    def deck_above_water_m(self) -> float:
        deck = (self.depth_m - self.draft_m if self.depth_m is not None else 0.55 * self.draft_m)
        if self.height_above_water_m is not None:
            deck = min(deck, 0.65 * self.height_above_water_m)
        return deck

    def manifest(self) -> dict:
        details = asdict(self)
        if self.provenance == "published":
            details["sources"] = [
                {"name": title, "url": url} for title, url in VESSEL_SOURCES[self.name]
            ]
        return details


GUNNERUS_SPEC = VesselSpec(
    "R/V Gunnerus", "research", 36.25, 9.60, 2.70,
    depth_m=4.20, displacement_t=543.79, provenance="published", length_label="전장",
    beam_label="형폭", draft_label="형흘수", displacement_label="배수량 (2023 실험 제원)",
)
TRYOSHNIKOV_SPEC = VesselSpec(
    "R/V Akademik Tryoshnikov", "research", 133.53, 23.25, 8.50,
    depth_m=13.50, displacement_t=16539, provenance="published", length_label="전장",
    beam_label="최대 선폭", draft_label="설계 수선 흘수", hull_variant="polar",
)
VESSELS = {spec.name: spec for spec in (GUNNERUS_SPEC, TRYOSHNIKOV_SPEC)}
VESSEL_SOURCES = {
    GUNNERUS_SPEC.name: (
        ("NTNU 선박 제원", "https://www.ntnu.edu/documents/1262202806/0/Specifications%2BRV%2BGUNNERUS%2B20260305.pdf/8430b710-1719-2cfe-2bce-a4f4d9253252?t=1772726476143"),
        ("파도 차폐 실험 제원", "https://github.com/shiplab/gunnerus/tree/main/operational/wave-shielding-2023#the-case-in-gunnerus-terms"),
    ),
    TRYOSHNIKOV_SPEC.name: (
        ("AARI 운항기관 제원", "https://www.aari.ru/the-fleet/rv-%C2%ABakademik-tryoshnikov%C2%BB"),
        ("러시아 선급", "https://rs-class.org/c/getves.php?imo=9548536"),
    ),
}
DEFAULT_VESSEL = VesselSpec("기본 모형", "other", 36.25, 9.60, 2.70,
                            provenance="illustrative", length_label="모형 길이")


def user_vessel_from_manifest(data: dict | None, reference: VesselSpec) -> VesselSpec | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("설정 파일의 선박 제원을 확인해 주세요.")
    if data.get("provenance") != "user":
        return None
    try:
        name, kind = data["name"].strip(), data["kind"]
        height = data.get("height_above_water_m")
        # A polar profile belongs to the active Tryoshnikov visual template.
        return VesselSpec(name, kind, float(data["length_m"]), float(data["beam_m"]),
                          float(data["draft_m"]), float(height) if height is not None else None,
                          hull_variant=reference.hull_variant if kind == "research" else "standard")
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("설정 파일의 선박 제원을 확인해 주세요.") from error
