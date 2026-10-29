"""Archive #148/#170/#172/#173: failed independent tests remain failed evidence.

The synthetic metrics isolate reporting/publication behavior. They are not
model activations, NPU results, or task accuracy measurements.
"""
import copy
import hashlib
import json
import sys

import pytest
import torch

from oscar_ascend.optr_artifacts import matrix_digest
from oscar_ascend.optr_checkpoint import FitCheckpoints
from tools import prepare_optr_rotations as prepare


NAME = "model.layers.3.self_attn.attn"
PARTITIONS = ("single_split", "captured_runtime_frames")


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def metrics(split, errors):
    rows = []
    for index, error in enumerate(errors):
        request_id = split + ("-a" if index < 2 else "-b")
        partition = PARTITIONS[index % 2]
        rows.append({"request_id": request_id + ":" + partition, "partition": partition,
                     "sample_sha256": hashlib.sha256(request_id.encode()).hexdigest(),
                     "squared_error": error, "teacher_energy": 4., "normalized_error": error / 4.,
                     "frames": [{"frame_id": 0, "context": 0, "current_end": 1, "query_positions": [0],
                                 "squared_error": error, "teacher_energy": 4., "normalized_error": error / 4.,
                                 "terminal_query": {"position": 0, "squared_error": error,
                                                    "teacher_energy": 4., "normalized_error": error / 4.}}]})
    error = sum(row["squared_error"] for row in rows)
    energy = sum(row["teacher_energy"] for row in rows)
    return {"squared_error": error, "teacher_energy": energy,
            "normalized_error": error / energy, "requests": rows}


def test_terminal_only_regression_is_printed_and_persisted(tmp_path):
    from test_optr_frame_gate import metric
    from oscar_ascend.optr_policy import optr_policy
    before, after = metric(split="test"), metric((3., .5), split="test")
    after["requests"][0]["frames"][0]["terminal_query"].update(
        squared_error=2.5, normalized_error=1.25)
    record = {"test": {"baseline": before, "selected": after}}
    checkpoints = FitCheckpoints(tmp_path, {"test_only": True})
    events = []
    report = prepare._report_independent_test_failure(
        NAME, 3, record, optr_policy({}), checkpoints, lambda **row: events.append(row))
    row = report["regressed_requests"][0]
    assert row["regressed_metrics"] == []
    frame = row["regressed_frames"][0]
    assert frame["regressed_metrics"] == []
    assert frame["regressed_terminal_metrics"] == ["squared_error", "normalized_error"]
    assert report["artifact_publish_allowed"] and not report["test_used_for_selection"]
    assert json.loads((checkpoints.root / "layer-3-independent-test-failed.json").read_text())["regressed_requests"] == [row]
    assert any(event.get("phase") == "independent_test_request_regressed" and
               event["regressed_frames"] == [frame] for event in events)


def samples(heads=2):
    return {split: [[{"request_id": split}] for _ in range(heads)]
            for split in ("train", "validation", "test")}


def baseline(dim=4):
    return {"layer_id": 3, "Rk": -torch.eye(dim).repeat(2, 1, 1),
            "Rv": -torch.eye(dim).repeat(2, 1, 1), "k_mean": torch.zeros(2, dim)}


def fitter(rk, rv, _mean, *_args, **_kwargs):
    return rk.roll(1, dims=-1), rv, {"accepted": True, "fixture": "validation_candidate"}


def install_metrics(monkeypatch, seed, test_errors):
    checked = []
    values = {"validation": {"baseline": metrics("validation", [4., 4., 4., 4.]),
                             "selected": metrics("validation", [2., 2., 2., 2.])},
              "test": {"baseline": metrics("test", [4., 4., 4., 4.]),
                       "selected": metrics("test", test_errors)}}
    def measure(constants, data, _policy, **_options):
        split = data[0][0]["request_id"]
        choice = "baseline" if torch.equal(constants["Rk"], seed["Rk"]) else "selected"
        checked.append((split, choice))
        return copy.deepcopy(values[split][choice])
    monkeypatch.setattr(prepare, "_metrics", measure)
    return values, checked


@pytest.mark.parametrize("test_errors", [
    [4.123456789012345, 4., 4., 4.765432109876543],
    [1., 1., 4.123456789012345, 1.],  # Total improves but one request fails.
])
def test_quality_failure_keeps_exact_metrics_identity_and_rejected_requests(tmp_path, monkeypatch, capsys, test_errors):
    seed = baseline()
    values, checked = install_metrics(monkeypatch, seed, test_errors)
    identity = {"profile": {"recipe_version": 5}, "base_artifact_sha256": "a" * 64,
                "trace_manifest_sha256": "b" * 64, "fitting_implementation_sha256": "c" * 64,
                "test_only": True}
    checkpoints = FitCheckpoints(tmp_path / "fit-checkpoints", identity)
    policy = {"validation_partitions": list(PARTITIONS), "steps": 16}
    selected, record = prepare.refine_layer(NAME, seed, samples(), policy, fitter=fitter, checkpoints=checkpoints)
    path = checkpoints.root / "layer-3-independent-test-failed.json"
    report = json.loads(path.read_text())
    assert report["record"] == record
    assert torch.equal(selected["Rk"], seed["Rk"].roll(1, dims=-1))
    assert report["identity"] == checkpoints.identity == identity
    assert report["policy"] == policy and not report["test_policy_unchanged"]
    assert report["artifact_publish_allowed"] and report["experimental"]
    assert not report["test_used_for_selection"]
    record = report["record"]
    assert record["validation"] == values["validation"] and record["test"] == values["test"]
    assert record["head_decisions"] == [{"accepted": True, "fixture": "validation_candidate"}] * 2
    assert record["base_constants_sha256"] == matrix_digest({NAME: seed})
    selected = {**seed, "Rk": seed["Rk"].roll(1, dims=-1)}
    assert record["selected_constants_sha256"] == matrix_digest({NAME: selected})
    expected_bad = [row for row in values["test"]["selected"]["requests"] if row["squared_error"] > 4.]
    assert [row["selected"] for row in report["regressed_requests"]] == expected_bad
    assert all(row["partition"] in PARTITIONS for row in report["regressed_requests"])
    assert checked == [("validation", "baseline"), ("validation", "selected"),
                       ("test", "baseline"), ("test", "selected")]
    assert checkpoints.load("layer-3") is None  # Failure file is never a completed layer pointer.
    assert checkpoints.load("layer-3-head-0") is not None  # Existing head work is diagnostic/reusable only.
    output = capsys.readouterr().out
    assert "independent_test_failed" in output and "independent_test_request_regressed" in output
    for row in expected_bad:
        assert row["request_id"] in output and repr(row["squared_error"]) in output
    if sum(test_errors) < 16:
        assert report["total"]["regressed_metrics"] == []
        assert report["regressed_requests"]  # Average improvement never relabels test as passed.


@pytest.mark.parametrize("failure", [True, False])
def test_prepare_publishes_experimental_constants_and_preserves_test_outcome(tmp_path, monkeypatch, failure):
    from test_optr_integration import artifact, configuration
    from oscar_ascend import optr_traces
    config = configuration(tmp_path)
    template = artifact(config)
    seed = template["layers"][NAME]
    seed_artifact = {"global_kv_heads": 2, "layers": {NAME: seed}}
    seed_path = tmp_path / "seed.pt"
    torch.save(seed_artifact, seed_path)
    monkeypatch.setattr(prepare, "_base_config", lambda *_: {"rotation_artifact": str(seed_path)})
    monkeypatch.setattr(prepare, "_prepare_base", lambda *_a, **_k: seed_artifact)
    monkeypatch.setattr(optr_traces, "load_trace_manifest", lambda *_a, **_k: {"fixture": "trace"})
    monkeypatch.setattr(optr_traces, "load_layer_samples", lambda *_a, **_k: samples())
    install_metrics(monkeypatch, seed, [1., 1., 4.25, 1.] if failure else [2., 2., 2., 2.])
    output = tmp_path / "refined" / "calibrated.pt"
    published = []
    original_publish = prepare.publish_artifact
    def publish(value, destination):
        published.append(destination)
        return original_publish(value, destination)
    monkeypatch.setattr(prepare, "publish_artifact", publish)
    options = {"output": output, "testing": True, "capture_runner": lambda *_: {"fixture": "capture"},
               "fitter": fitter}
    result = prepare.prepare(config, **options)
    assert result["accepted_layers"] == [NAME] and output.is_file()
    assert published == [output]
    assert list(output.parent.glob("fit-checkpoints/*/layer-3.json"))
    assert result["experimental"] and result["model_quality"] == "not_measured"
    assert result["independent_test"] == ("failed" if failure else "passed")
    from oscar_ascend.rotations import load_active_artifact, prepare_device_rotations
    loaded = load_active_artifact(config, path=output, device="cpu", testing=True)
    cert = loaded["refinement_certificate"]
    assert cert["schema_version"] == 2 and cert["experimental"]
    assert cert["independent_test"] == {"status": result["independent_test"],
                                        "regressed_layers": [NAME] if failure else []}
    for rank in (0, 1):
        matrices = prepare_device_rotations(loaded, device="cpu", testing=True, tp_rank=rank, tp_size=2)[NAME]
        assert torch.equal(matrices.key_transposed.T, seed["Rk"][rank].roll(1, dims=-1))
    def forbidden(*_a, **_kw):
        pytest.fail("completed experimental PT must load without refitting")
    monkeypatch.setattr(prepare, "_prepare_base", forbidden)
    cached = prepare.prepare(config, output=output, testing=True)
    assert cached["reused"] and cached["independent_test"] == result["independent_test"]
    assert cached["experimental"]
    if failure:
        files = list(output.parent.glob("fit-checkpoints/*/layer-3-independent-test-failed.json"))
        assert len(files) == 1
        report = json.loads(files[0].read_text())
        assert report["identity"]["profile"] == template["calibration_profile"]
        assert report["identity"]["test_only"] is True
    else:
        assert not list(output.parent.rglob("*-independent-test-failed.json"))


def test_report_write_failure_still_raises_quality_gate_value_error(tmp_path, monkeypatch):
    from tools import phase
    seed = baseline()
    install_metrics(monkeypatch, seed, [5., 4., 4., 4.])
    checkpoints = FitCheckpoints(tmp_path, {"test_only": True})
    # Finish head checkpoints first; only diagnostic persistence is failed.
    for head in range(2):
        rk, rv, decision = fitter(seed["Rk"][head], seed["Rv"][head], None)
        checkpoints.save(f"layer-3-head-{head}", {"Rk": rk, "Rv": rv, "decision": decision})
    def unavailable(*_args):
        raise OSError("diagnostic filesystem unavailable")
    monkeypatch.setattr(phase, "atomic_json", unavailable)
    with pytest.raises(prepare.IndependentTestRegression, match="artifact not published") as caught:
        prepare.refine_layer(NAME, seed, samples(), {}, fitter=fitter, checkpoints=checkpoints)
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.report["status"] == "independent_test_failed"
    assert checkpoints.load("layer-3") is None


def failure_record():
    seed = baseline()
    return {"accepted": True, "head_decisions": [{"accepted": True}] * 2,
            "base_constants_sha256": matrix_digest({NAME: seed}),
            "selected_constants_sha256": matrix_digest({NAME: {**seed, "Rk": seed["Rk"].roll(1, dims=-1)}}),
            "validation": {"baseline": metrics("validation", [4.] * 4),
                           "selected": metrics("validation", [2.] * 4)},
            "test": {"baseline": metrics("test", [4.] * 4),
                     "selected": metrics("test", [1., 1., 4.25, 1.])}}


@pytest.mark.parametrize("sink", ["stdout", "callback"])
@pytest.mark.parametrize("disk_failure", [False, True])
def test_failed_output_cannot_mask_quality_gate_or_prevent_prior_persistence(tmp_path, monkeypatch, sink, disk_failure):
    from tools import phase
    checkpoints = FitCheckpoints(tmp_path, {"test_only": True, "source": "quality-failure-fixture"})
    path = checkpoints.root / "layer-3-independent-test-failed.json"
    record = failure_record()
    order = []
    original = phase.atomic_json
    disk_error = OSError("quality report disk unavailable")
    output_error = BrokenPipeError("stdout pipe closed") if sink == "stdout" else RuntimeError("progress callback unavailable")
    def persist(destination, report):
        order.append("persist")
        if disk_failure:
            raise disk_error
        original(destination, report)
    def failed_output(*_args, **_kwargs):
        order.append("output")
        assert path.exists() is (not disk_failure)
        raise output_error
    class ClosedStdout:
        write = staticmethod(failed_output)
        def flush(self):
            pass
    with monkeypatch.context() as patch:
        patch.setattr(phase, "atomic_json", persist)
        if sink == "stdout":
            patch.setattr(sys, "stdout", ClosedStdout())
        emit = prepare._progress if sink == "stdout" else failed_output
        with pytest.raises(prepare.IndependentTestRegression, match="independent test report failed; artifact not published") as caught:
            prepare._report_independent_test_failure(NAME, 3, record, {}, checkpoints, emit)
    error = caught.value
    assert order == ["persist", "output"]
    assert error.report["record"] == record and error.report["artifact_publish_allowed"]
    assert error.report["identity"] == checkpoints.identity
    assert error.__cause__ is (disk_error if disk_failure else output_error)
    assert error.reporting_errors == ((disk_error, output_error) if disk_failure else (output_error,))
    if not disk_failure:
        assert json.loads(path.read_text()) == error.report
    else:
        assert not path.exists()
    assert checkpoints.load("layer-3") is None
