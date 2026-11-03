"""Archive #176: real spawned workers repair authenticated v10 work on CPU."""
import json
from pathlib import Path

from oscar_ascend.rotations import load_active_artifact
from tools import optr_parallel, prepare_optr_rotations
from test_optr_recipe3_pipeline import NAME, case, forbidden
from test_optr_trust_region_pipeline import (
    LEGACY_DIGEST, VERSION, _forbid_new_fit_or_capture, _make_rejected_source,
)


def _forbidden_worker_fit(*_args, **_kwargs):
    raise AssertionError("authenticated source repair must not initialize an NPU or refit")


def _source_repair_worker(spec, index, cpu_threads, pipe):
    """Test-only path adaptation; execute the original production worker."""
    from oscar_ascend import optr_sequential
    from tools import prepare_optr_recipe3, prepare_optr_recipe3_base
    from tools.optr_fit_worker import fit_worker

    assert spec["testing"] is True
    prepare_optr_recipe3_base.ROOT = Path(spec["base_config"]["rotation_artifact"]).parents[4]
    optr_sequential.fitting_device = _forbidden_worker_fit
    prepare_optr_recipe3.fit_perhead = _forbidden_worker_fit
    fit_worker(spec, index, cpu_threads, pipe)


def _events(output):
    return [json.loads(line.split("OPTR_PROGRESS ", 1)[1])
            for line in output.splitlines() if line.startswith("[oscar] OPTR_PROGRESS ")]


def test_real_spawned_source_worker_publishes_then_zero_worker_resume(case, monkeypatch, capsys):
    config, output, _, source_files = _make_rejected_source(case, monkeypatch)
    _forbid_new_fit_or_capture(monkeypatch)
    original_workers = optr_parallel.run_layer_workers

    def run_workers(spec, names, **kwargs):
        assert spec["testing"] is True and spec["source_checkpoint"] is not None
        return original_workers(spec, names, **kwargs, worker_target=_source_repair_worker)

    monkeypatch.setattr(optr_parallel, "run_layer_workers", run_workers)
    capsys.readouterr()
    result = prepare_optr_rotations.prepare(config, output=output, testing=True,
        parallel_testing=True, capture_runner=forbidden, base_runner=forbidden)
    assert result["accepted_layers"] == [NAME]
    events = _events(capsys.readouterr().out)
    starts = [row for row in events if row.get("phase") == "fit_worker_started"]
    exits = [row for row in events if row.get("phase") == "fit_worker_exited"]
    assert len(starts) == len(exits) == 1
    assert starts[0]["worker"] == exits[0]["worker"] == 0
    assert starts[0]["pid"] == exits[0]["pid"] and exits[0]["returncode"] == 0
    assert any(row.get("phase") == "fit_worker_ready" and row["fitting_device"] == "cpu"
               for row in events)
    assert any(row.get("phase") == "recipe3_source_reused" and row["refit"] is False
               for row in events)
    assert any(row.get("phase") == "recipe3_repair_locked_before_test" for row in events)
    cleanup = next(row for row in events if row.get("phase") == "fit_workers_cleanup")
    assert cleanup["cleanup_complete"] is True
    artifact = load_active_artifact(config, path=output, device="cpu", testing=True)
    certificate = artifact["refinement_certificate"]
    assert certificate["policy"]["policy_version"] == VERSION
    assert certificate["independent_test"]["status"] == "passed"
    assert certificate["layers"][NAME]["source_fit_provenance"]["fitting_implementation_sha256"] == LEGACY_DIGEST
    completed = next((output.parent / "fit-checkpoints").glob("*/layer-3.json"))
    identity = json.loads(completed.read_text())["identity"]
    assert identity["source_fit_snapshot_sha256"] == certificate["source_fit_snapshot_sha256"]
    completed_bytes = completed.read_bytes()
    assert all(path.read_bytes() == content for path, content in source_files.items())

    output.unlink()
    (output.parent / ".optr.complete.json").unlink()
    # An empty pending queue must return before multiprocessing is initialized.
    monkeypatch.setattr(optr_parallel.mp, "get_context", forbidden)
    capsys.readouterr()
    resumed = prepare_optr_rotations.prepare(config, output=output, testing=True,
        parallel_testing=True, capture_runner=forbidden, base_runner=forbidden)
    assert resumed["accepted_layers"] == [NAME]
    events = _events(capsys.readouterr().out)
    assert next(row for row in events if row.get("phase") == "parallel_fit_plan")["workers"] == 0
    assert any(row.get("phase") == "layer_checked_resumed" for row in events)
    assert not any(row.get("phase") == "fit_worker_started" for row in events)
    assert completed.read_bytes() == completed_bytes
    assert load_active_artifact(config, path=output, device="cpu", testing=True)["refinement_certificate"] == certificate
    assert all(path.read_bytes() == content for path, content in source_files.items())
