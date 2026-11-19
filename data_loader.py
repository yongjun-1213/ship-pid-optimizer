"""Read and resample measured attitudes without modifying the input files."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

DATA_DIR = Path(__file__).resolve().parent / "data"
GUNNERUS = "R/V Gunnerus"
ACE = "R/V Akademik Tryoshnikov"
VIRTUAL = "가상 파도"
UPLOAD = "CSV / CSV.GZ 업로드"
SOURCE_LINKS = {
    GUNNERUS: "https://github.com/shiplab/gunnerus/tree/main/operational/wave-shielding-2023",
    ACE: "https://zenodo.org/records/3733228",
}
SOURCE_FILES = {
    GUNNERUS: "gunnerus_seapath_2hz.csv.gz",
    ACE: "ace_hydrins_20170126_excerpt.csv.gz",
}
TIME_UNITS = {"s": 1.0, "ms": 0.001, "us": 0.000001}


@dataclass(frozen=True)
class ColumnMapping:
    time: str
    roll: str
    pitch: str
    time_unit: str
    roll_unit: str
    pitch_unit: str
    roll_sign: int
    pitch_sign: int
    clock: str | None = None


@dataclass(frozen=True)
class DataQuality:
    input_rows: int
    bad_time_rows: int
    bad_angle_rows: int
    duplicate_rows: int
    observation_dt: float
    max_gap: float
    split_gaps: int


@dataclass(frozen=True)
class MotionData:
    source: str
    record_id: str
    filename: str
    seconds: np.ndarray
    angles_deg: np.ndarray
    segments: tuple[tuple[int, int], ...]  # Half-open indexes of valid observations.
    quality: DataQuality
    origin: str | None = None
    mapping: ColumnMapping | None = None
    wave: str | None = None

    def cleaned_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({
            "source_offset_s": self.seconds,
            "roll_deg": self.angles_deg[:, 0],
            "pitch_deg": self.angles_deg[:, 1],
        })
        if self.origin:
            frame["time_utc"] = pd.Timestamp(self.origin) + pd.to_timedelta(self.seconds, unit="s")
        return frame.assign(
            segment=np.repeat(np.arange(1, len(self.segments) + 1),
                              [b - a for a, b in self.segments])
        )


@dataclass(frozen=True)
class Window:
    source: str
    record_id: str
    observed_start: float
    observed_end: float
    origin: str | None
    observations: int

    def manifest(self) -> dict:
        data = asdict(self)
        if self.origin:
            origin = pd.Timestamp(self.origin)
            data["start_utc"] = (origin + pd.to_timedelta(self.observed_start, unit="s")).isoformat()
            data["end_utc"] = (origin + pd.to_timedelta(self.observed_end, unit="s")).isoformat()
        return data

    def overlaps(self, other: "Window") -> bool:
        if self.record_id == other.record_id and self.origin == other.origin:
            return max(self.observed_start, other.observed_start) <= min(
                self.observed_end, other.observed_end
            )
        # The two built-in records come from distinct vessels and years. Also guard
        # UTC overlaps for an uploaded copy/extract of either record.
        if self.origin and other.origin:
            a, b = self.manifest(), other.manifest()
            return max(pd.Timestamp(a["start_utc"]), pd.Timestamp(b["start_utc"])) <= min(
                pd.Timestamp(a["end_utc"]), pd.Timestamp(b["end_utc"])
            )
        return False


@dataclass(frozen=True)
class MotionInput:
    time_s: np.ndarray
    angles_deg: np.ndarray
    window: Window
    dt: float
    source_elapsed_s: np.ndarray
    wave: str | None
    digest: str
    mapping: ColumnMapping | None
    interpolation: str

    def frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({
            "simulation_time_s": self.time_s,
            "source_offset_s": self.source_elapsed_s,
            "ship_roll_deg": self.angles_deg[:, 0],
            "ship_pitch_deg": self.angles_deg[:, 1],
        })
        if self.window.origin:
            frame["time_utc"] = (
                pd.Timestamp(self.window.origin) + pd.to_timedelta(self.source_elapsed_s, unit="s")
            )
        return frame


def _immutable(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array, dtype=np.float64)
    array.flags.writeable = False
    return array


def read_csv(payload: bytes) -> pd.DataFrame:
    """Parse compressed/plain CSV, including daily ACE files with .txt extensions."""
    compression = "gzip" if payload.startswith(b"\x1f\x8b") else None
    options = dict(compression=compression, sep=None, engine="python")
    for encoding in ("utf-8-sig", "cp949", "latin1"):
        try:
            frame = pd.read_csv(BytesIO(payload), encoding=encoding, **options)
            if frame.shape[1] < 3:
                raise ValueError("시간·Roll·Pitch에 해당하는 최소 세 개의 열이 필요합니다.")
            frame.columns = frame.columns.str.strip()
            if not frame.columns.is_unique:
                raise ValueError("중복된 열 이름을 구분할 수 있게 수정해 주세요.")
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError("텍스트 인코딩을 확인해 주세요.")


def detect_columns(columns) -> dict[str, str | None]:
    matches = {key: None for key in ("time", "roll", "pitch", "clock")}
    by_name = {re.sub(r"[^a-z0-9가-힣]+", "", str(c).casefold()): c for c in columns}
    aliases = {
        "time": ("timeutc", "timestamputc", "timestamp", "datetime", "time", "times",
                 "timems", "timeus", "sourceoffsets", "simulationtimes", "elapsedseconds",
                 "elapsed", "seconds", "t", "시간", "pcdateutc"),
        "roll": ("platformroll", "rolldeg", "rollrad", "roll", "shiprolldeg", "rollangle", "롤"),
        "pitch": ("platformpitch", "pitchdeg", "pitchrad", "pitch", "shippitchdeg", "pitchangle", "피치"),
        "clock": ("pctimehhmmsssss", "pctimehhmmssss", "pctimeutc"),
    }
    for key, candidates in aliases.items():
        matches[key] = next((by_name[name] for name in candidates if name in by_name), None)
    return matches


@lru_cache(maxsize=2)
def source_bytes(source: str) -> bytes:
    return (DATA_DIR / SOURCE_FILES[source]).read_bytes()


def source_mapping(source: str, roll_sign: int | None = None, pitch_sign: int | None = None):
    if source == GUNNERUS:
        return ColumnMapping("time_utc", "roll_deg", "pitch_deg", "datetime", "deg", "deg", 1, 1)
    if source == ACE and roll_sign in (-1, 1) and pitch_sign in (-1, 1):
        return ColumnMapping("pc_date_utc", "platform_roll", "platform_pitch", "datetime", "deg", "deg",
                             roll_sign, pitch_sign, "pc_time_hhmmsssss")
    raise ValueError("원본 Roll과 Pitch의 양의 방향을 선택해 주세요.")


def _parse_time(frame: pd.DataFrame, mapping: ColumnMapping):
    raw = frame[mapping.time]
    if mapping.time_unit == "datetime":
        raw = raw.astype("string").str.strip()
        if mapping.clock:
            raw = raw + "T" + frame[mapping.clock].astype("string").str.strip()
        else:
            raw = raw.mask(raw.str.fullmatch(r"[+-]?\d+(\.\d+)?", na=False))
        dates = pd.to_datetime(raw, format="mixed", errors="coerce", utc=True)
        # Subtract integer nanoseconds before conversion to float.
        first = dates.min()
        if pd.isna(first):
            raise ValueError("UTC 날짜·시각을 읽을 수 없습니다. 시간 열과 단위를 확인해 주세요.")
        ticks = dates.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        valid = dates.notna().to_numpy()
        offsets = np.full(len(frame), np.nan)
        offsets[valid] = (ticks[valid] - first.value) / 1e9
        return offsets, first.isoformat()
    if mapping.time_unit not in TIME_UNITS or mapping.clock:
        raise ValueError("숫자 시간에는 초, 밀리초, 마이크로초 중 단일 시간 열을 지정해 주세요.")
    raw = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw)
    offsets = np.full(len(frame), np.nan)
    if valid.any():
        offsets[valid] = (raw[valid] - raw[valid].min()) * TIME_UNITS[mapping.time_unit]
    return offsets, None


def clean_motion(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    source: str = UPLOAD,
    filename: str = "",
    max_angle: float = 70.0,
    max_gap: float | None = None,
) -> MotionData:
    names = [mapping.time, mapping.roll, mapping.pitch] + ([mapping.clock] if mapping.clock else [])
    if len(set(names)) != len(names) or set(names).difference(frame.columns):
        raise ValueError("시간, Roll, Pitch에 서로 다른 유효한 열을 지정해 주세요.")
    if {mapping.roll_unit, mapping.pitch_unit}.difference({"deg", "rad"}):
        raise ValueError("Roll과 Pitch의 각도 단위를 지정해 주세요.")
    if mapping.roll_sign not in (-1, 1) or mapping.pitch_sign not in (-1, 1):
        raise ValueError("Roll과 Pitch의 양의 방향을 지정해 주세요.")
    if not 0 < max_angle < 90 or (max_gap is not None and (not np.isfinite(max_gap) or max_gap <= 0)):
        raise ValueError("각도 한계는 0°~90° 사이, 보간 허용 간격은 양수여야 합니다.")

    seconds, origin = _parse_time(frame, mapping)
    angles = np.column_stack([
        pd.to_numeric(frame[mapping.roll], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(frame[mapping.pitch], errors="coerce").to_numpy(dtype=float),
    ])
    angles *= [180 / np.pi if mapping.roll_unit == "rad" else 1,
               180 / np.pi if mapping.pitch_unit == "rad" else 1]
    valid_time = np.isfinite(seconds)
    native_times = np.unique(seconds[valid_time])
    if len(native_times) < 2:
        raise ValueError("서로 다른 유효 시각이 2개 이상 필요합니다.")
    observation_dt = float(np.median(np.diff(native_times)))
    gap_limit = float(max_gap if max_gap is not None else 3.05 * observation_dt)
    valid_angle = np.isfinite(angles).all(axis=1) & (np.abs(angles) <= max_angle).all(axis=1)
    valid = valid_time & valid_angle
    order = np.flatnonzero(valid)
    order = order[np.argsort(seconds[order], kind="stable")]
    _, first = np.unique(seconds[order], return_index=True)
    unique = order[first]
    if len(unique) < 2:
        raise ValueError("정제 후 유효한 Roll/Pitch 관측값이 2개 미만입니다.")
    times = _immutable(seconds[unique])
    breaks = np.flatnonzero(np.diff(times) > gap_limit) + 1
    edges = [0, *breaks.tolist(), len(times)]
    segments = tuple(zip(edges[:-1], edges[1:]))
    quality = DataQuality(len(frame), int((~valid_time).sum()),
                          int((valid_time & ~valid_angle).sum()), len(order) - len(unique),
                          observation_dt, gap_limit, len(breaks))
    # Compute identity independently of quality settings and angle direction.
    identity = sha256()
    if source in SOURCE_FILES:
        identity.update(source.encode())
    else:
        identity.update((origin or "").encode())
        identity.update(np.column_stack((seconds[valid_time], angles[valid_time])).tobytes())
    angles *= [mapping.roll_sign, mapping.pitch_sign]
    return MotionData(source, identity.hexdigest(), filename, times, _immutable(angles[unique]),
                      segments, quality, origin, mapping)


def resample_window(
    data: MotionData,
    segment: int,
    start_s: float,
    end_s: float,
    dt: float = 0.02,
    *,
    method: str = "pchip",
    max_steps: int = 100_000,
) -> MotionInput:
    if not np.isfinite([dt, start_s, end_s]).all() or dt <= 0 or end_s <= start_s:
        raise ValueError("시작 시각은 종료 시각보다 작고 제어 주기는 양수여야 합니다.")
    if not 0 <= segment < len(data.segments) or method not in ("pchip", "linear"):
        raise ValueError("연속 구간 또는 보간 방법을 확인해 주세요.")
    lo, hi = data.segments[segment]
    segment_times = data.seconds[lo:hi]
    if start_s < segment_times[0] - 1e-8 or end_s > segment_times[-1] + 1e-8:
        raise ValueError("선택한 연속 구간 안에서 시작·종료 시각을 지정해 주세요.")
    left = np.searchsorted(segment_times, start_s - 1e-9, side="left")
    right = np.searchsorted(segment_times, end_s + 1e-9, side="right")
    times = segment_times[left:right]
    values = data.angles_deg[lo + left:lo + right]
    if len(times) < 2:
        raise ValueError("실험 구간 안에 원본 관측 시각이 2개 이상 있어야 합니다.")
    relative = times - times[0]
    steps = int(np.floor(relative[-1] / dt + 1e-10))
    if steps < 1 or steps > max_steps:
        raise ValueError(f"실험당 제어 스텝은 1~{max_steps:,} 범위입니다. 구간 또는 제어 주기를 조정해 주세요.")
    grid = np.arange(steps + 1, dtype=float) * dt
    targets = np.minimum(grid, relative[-1])
    if method == "linear":
        sampled = np.column_stack([np.interp(targets, relative, values[:, i]) for i in range(2)])
    else:
        sampled = PchipInterpolator(relative, values, axis=0, extrapolate=False)(targets)
    window = Window(data.source, data.record_id, float(times[0]), float(times[-1]), data.origin, len(times))
    digest = sha256(repr((window, data.mapping, dt, method, data.wave)).encode() + sampled.tobytes()).hexdigest()
    return MotionInput(_immutable(grid), _immutable(sampled), window, dt, _immutable(grid + times[0]),
                       data.wave, digest, data.mapping, method)


def validate_holdouts(train: Window, *holdouts: Window) -> None:
    windows = (train, *holdouts)
    for index, window in enumerate(windows):
        if any(window.overlaps(other) for other in windows[:index]):
            raise ValueError("학습·검증 구간이 원본 관측 시각을 공유합니다. 서로 겹치지 않는 구간을 선택해 주세요.")
