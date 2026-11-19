import json
from dataclasses import replace
from threading import Event

import pytest

import optimizer
from evaluation import ObjectiveWeights, objective_score, objective_terms
from motor_model import MotorParams
from optimizer import OptimizationJob, StudyPlan, decode_pid, default_bounds
from pid_controller import PIDSettings
from simulation import SensorParams, simulate


def plan_for(train, mode=3):
    return StudyPlan(train, MotorParams(), SensorParams(), PIDSettings(), mode, default_bounds(mode),
                     ObjectiveWeights(), 7)


def completed(plan, path, count):
    job = OptimizationJob(plan, path)
    job.start(count)
    job.wait(15)
    snapshot = job.snapshot()
    assert snapshot.error is None and snapshot.save_error is None
    assert snapshot.status == "finished"
    assert len(snapshot.records) == count
    return job


@pytest.mark.parametrize("mode", [3, 6])
def test_optuna_real_data_and_best_pid_application(train, tmp_path, mode):
    plan = plan_for(train, mode)
    result = completed(plan, tmp_path / "progress.json", 13).snapshot()
    assert len(result.best.params) == mode
    assert result.best.score == min(row.score for row in result.records)
    assert len({tuple(row.params.values()) for row in result.records}) == 13
    best_pid = decode_pid(result.best.params, mode)
    if mode == 3:
        assert best_pid.roll == best_pid.pitch
    study_run = simulate(best_pid, train)
    assert result.best.score == objective_score(objective_terms(study_run.metrics, plan.motor), plan.weights)


def test_cancel_save_and_resume_reproducibly(train, tmp_path, monkeypatch):
    plan = plan_for(train)
    first, proceed = Event(), Event()
    original = optimizer.simulate

    def paused_simulate(*args, **kwargs):
        if not first.is_set():
            first.set()
            assert proceed.wait(8)
        return original(*args, **kwargs)

    monkeypatch.setattr(optimizer, "simulate", paused_simulate)
    partial = OptimizationJob(plan, tmp_path / "partial.json")
    partial.start(14)
    assert first.wait(5)
    partial.stop()
    assert partial.snapshot().status == "stopping"
    proceed.set()
    partial.wait(10)
    assert partial.snapshot().status == "paused"
    assert len(partial.snapshot().records) == 1
    restored = OptimizationJob.restore(plan, partial.path.read_bytes())
    with pytest.raises(ValueError, match="현재 실험"):
        OptimizationJob.restore(replace(plan, seed=8), partial.path.read_bytes())
    restored.start(14)
    restored.wait(15)
    continuous = completed(plan, tmp_path / "continuous.json", 14)
    assert restored.snapshot().records == continuous.snapshot().records
    assert restored.snapshot().best == continuous.snapshot().best


def test_job_records_vessel_without_changing_checkpoint_identity(train, tmp_path):
    from vessel_specs import GUNNERUS_SPEC, TRYOSHNIKOV_SPEC

    plan = plan_for(train)
    jobs = [OptimizationJob(plan, tmp_path / f"{i}.json", vessel_metadata=spec.manifest() if spec else None)
            for i, spec in enumerate((None, GUNNERUS_SPEC, TRYOSHNIKOV_SPEC))]
    assert len({json.loads(job.checkpoint())["signature"] for job in jobs}) == 1
    assert jobs[0].restore(plan, jobs[0].checkpoint()).vessel_metadata is None
    jobs[1].start(12)
    jobs[1].wait(12)
    assert jobs[1].snapshot().status == "finished"
    restored = jobs[2].restore(plan, jobs[1].checkpoint())
    assert restored.vessel_metadata["length_m"] == 36.25
    assert restored.snapshot().records == jobs[1].snapshot().records
