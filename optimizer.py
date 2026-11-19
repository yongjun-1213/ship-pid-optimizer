"""Run Optuna in a worker, saving completed trials as an atomic JSON checkpoint."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite, isclose
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from uuid import uuid4
import json
import os
import tempfile

import optuna
from optuna.distributions import FloatDistribution

from data_loader import MotionInput
from evaluation import ObjectiveWeights, objective_score, objective_terms
from motor_model import MotorParams
from pid_controller import Gains, PIDSettings, TwoAxisPID
from simulation import SensorParams, prepare_simulation, simulate

CHECKPOINT_VERSION = 1
MAX_TRIALS = 2000
AXIS_GAINS = ("kp", "ki", "kd")
COMPONENTS = ("roll", "pitch", "peak", "effort", "vibration", "constraints", "violations")


def parameters(mode):
    return AXIS_GAINS if mode == 3 else tuple(
        f"{gain}_{axis}" for axis in ("roll", "pitch") for gain in AXIS_GAINS
    )


def default_bounds(mode: int) -> dict[str, tuple[float, float]]:
    defaults = {"kp": (0.0, 24.0), "ki": (0.0, 8.0), "kd": (0.0, 1.6)}
    return {name: defaults[name.split("_")[0]] for name in parameters(mode)}


def decode_pid(values: dict[str, float], mode: int) -> TwoAxisPID:
    if mode == 3:
        return TwoAxisPID.shared(Gains(**{name: values[name] for name in AXIS_GAINS}))
    return TwoAxisPID(*(Gains(**{name: values[f"{name}_{axis}"] for name in AXIS_GAINS})
                        for axis in ("roll", "pitch")))


def _json(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2).encode("utf-8")


@dataclass(frozen=True)
class StudyPlan:
    motion: MotionInput
    motor: MotorParams
    sensor: SensorParams
    pid: PIDSettings
    mode: int
    bounds: dict[str, tuple[float, float]]
    weights: ObjectiveWeights
    seed: int

    def __post_init__(self):
        if self.mode not in (3, 6) or set(self.bounds) != set(parameters(self.mode)):
            raise ValueError("탐색 변수 수와 이득 범위를 확인해 주세요.")
        if not all(isfinite(lo) and isfinite(hi) and 0 <= lo < hi for lo, hi in self.bounds.values()):
            raise ValueError("이득 범위의 하한은 0 이상이고 상한보다 작아야 합니다.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Optuna 시드는 0~4,294,967,295 범위의 정수입니다.")

    def metadata(self) -> dict:
        return {
            "motion_digest": self.motion.digest,
            "training": self.motion.window.manifest(),
            "dt_s": self.motion.dt,
            "interpolation": self.motion.interpolation,
            "mapping": asdict(self.motion.mapping) if self.motion.mapping else None,
            "motor": asdict(self.motor),
            "sensor": asdict(self.sensor),
            "pid_settings": asdict(self.pid),
            "mode": self.mode,
            "bounds": self.bounds,
            "weights": asdict(self.weights),
            "seed": self.seed,
        }

    def signature(self) -> str:
        return sha256(_json(self.metadata())).hexdigest()


@dataclass(frozen=True)
class TrialRecord:
    number: int
    score: float
    params: dict[str, float]
    terms: dict[str, float]


@dataclass(frozen=True)
class JobSnapshot:
    status: str
    records: tuple[TrialRecord, ...]
    target: int
    elapsed_s: float
    error: str | None
    save_error: str | None

    @property
    def best(self) -> TrialRecord | None:
        return min(self.records, key=lambda row: row.score, default=None)

    @property
    def running(self) -> bool:
        return self.status in ("running", "stopping")


def _run_path() -> Path:
    root = Path(os.environ.get("STABILIZER_RUN_DIR", Path(tempfile.gettempdir()) / "ship-pid-studies"))
    return root / f"{uuid4().hex}.json"


class OptimizationJob:
    def __init__(self, plan: StudyPlan, path: Path | None = None):
        self.plan = plan
        self.path = path if path is not None else _run_path()
        self._lock = Lock()
        self._stop_requested = Event()
        self._thread: Thread | None = None
        self._records: list[TrialRecord] = []
        self._best: TrialRecord | None = None
        self._target = 0
        self._status = "ready"
        self._elapsed = 0.0
        self._started: float | None = None
        self._error: str | None = None
        self._save_error: str | None = None

    def start(self, total_trials: int):
        with self._lock:
            if self._status in ("running", "stopping"):
                raise ValueError("이미 최적화가 진행 중입니다.")
            if not len(self._records) < total_trials <= MAX_TRIALS:
                raise ValueError(f"총 목표 횟수는 완료 횟수보다 크고 {MAX_TRIALS:,} 이하여야 합니다.")
            self._stop_requested.clear()
            self._status, self._target, self._error = "running", total_trials, None
            self._started = perf_counter()
            self._save_locked()
            self._thread = Thread(target=self._run, name="ship-pid-tpe", daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            if self._status == "running":
                self._status = "stopping"
                self._stop_requested.set()

    def wait(self, seconds: float | None = None):
        if self._thread is not None:
            self._thread.join(seconds)

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(self._status, tuple(self._records), self._target, self._elapsed_now(),
                               self._error, self._save_error)

    def _elapsed_now(self) -> float:
        return self._elapsed + (perf_counter() - self._started if self._started is not None else 0.0)

    def _checkpoint(self) -> dict:
        return {
            "schema": CHECKPOINT_VERSION,
            "optuna_version": optuna.__version__,
            "signature": self.plan.signature(),
            "plan": self.plan.metadata(),
            "trial_target": self._target,
            "elapsed_s": self._elapsed_now(),
            "status": self._status,
            "records": [asdict(record) for record in self._records],
            "best": asdict(self._best) if self._best else None,
        }

    def checkpoint(self) -> bytes:
        with self._lock:
            return _json(self._checkpoint())

    def _save_locked(self):
        partial = self.path.with_suffix(".json.part")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(_json(self._checkpoint()))
            partial.replace(self.path)
            self._save_error = None
        except OSError as error:
            self._save_error = str(error)

    @classmethod
    def restore(cls, plan: StudyPlan, payload: bytes) -> "OptimizationJob":
        try:
            saved = json.loads(payload)
            if (saved["schema"] != CHECKPOINT_VERSION or saved["signature"] != plan.signature()
                    or saved["optuna_version"] != optuna.__version__):
                raise ValueError("체크포인트의 학습 구간·설정 또는 Optuna 버전이 현재 실험과 다릅니다.")
            job = cls(plan)
            if len(saved["records"]) > MAX_TRIALS:
                raise ValueError("복원할 실험 기록 개수가 너무 많습니다.")
            elapsed = float(saved["elapsed_s"])
            target = int(saved["trial_target"])
            if not isfinite(elapsed) or elapsed < 0 or not len(saved["records"]) <= target <= MAX_TRIALS:
                raise ValueError("체크포인트의 진행 횟수 또는 시간이 올바르지 않습니다.")
            for index, data in enumerate(saved["records"], 1):
                params = {name: float(value) for name, value in data["params"].items()}
                terms = {name: float(value) for name, value in data["terms"].items()}
                score = float(data["score"])
                if (data["number"] != index or set(params) != set(plan.bounds)
                        or any(not lo <= params[name] <= hi for name, (lo, hi) in plan.bounds.items())
                        or set(terms) != set(COMPONENTS)
                        or not all(isfinite(v) and v >= 0 for v in terms.values())
                        or not isfinite(score) or not isclose(score, objective_score(terms, plan.weights))):
                    raise ValueError("체크포인트에 손상된 실험 기록이 있습니다.")
                job._records.append(TrialRecord(index, score, params, terms))
            job._best = min(job._records, key=lambda row: row.score, default=None)
            job._target, job._elapsed = target, elapsed
            job._status = "paused" if len(job._records) < target else "finished"
            return job
        except (KeyError, TypeError, AttributeError, OverflowError, json.JSONDecodeError) as error:
            raise ValueError("이 프로그램에서 저장한 체크포인트 JSON을 선택해 주세요.") from error

    def _run(self):
        failure = None
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            prepared = prepare_simulation(self.plan.motion, self.plan.sensor)
            mode, bounds = self.plan.mode, self.plan.bounds
            order = parameters(mode)
            distributions = {name: FloatDistribution(*bounds[name]) for name in order}
            for record in self._records:
                study.add_trial(optuna.trial.create_trial(
                    params=record.params, distributions=distributions, value=record.score))
            while len(self._records) < self._target and not self._stop_requested.is_set():
                index = len(self._records)
                # A seed per trial preserves the proposal sequence after checkpoint restore.
                study.sampler = optuna.samplers.TPESampler(seed=(self.plan.seed + index * 104729) % 2**32)
                trial = study.ask()
                values = {name: trial.suggest_float(name, *bounds[name]) for name in order}
                gains = decode_pid(values, mode)
                result = simulate(gains, prepared, self.plan.motor, pid_settings=self.plan.pid)
                terms = objective_terms(result.metrics, self.plan.motor)
                score = objective_score(terms, self.plan.weights)
                study.tell(trial, score)
                with self._lock:
                    row = TrialRecord(index + 1, score, values, terms)
                    self._records.append(row)
                    if self._best is None or score < self._best.score:
                        self._best = row
                    self._save_locked()
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        finally:
            with self._lock:
                self._elapsed, self._started = self._elapsed_now(), None
                self._error = failure
                if failure:
                    self._status = "failed"
                else:
                    self._status = "finished" if len(self._records) == self._target else "paused"
                self._save_locked()
