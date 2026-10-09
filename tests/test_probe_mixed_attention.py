"""Archive #130/#139/#143/#148-151: mixed probe CPU contracts only."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tools import probe_mixed_attention as mixed


def test_shape_plan_is_31_q4_plus_one_real_16260_prefill():
    cold, history = mixed.shape_plan()
    assert cold.name == "mixed_cold_first_chunk"
    assert history.name == "mixed_history_30k"
    contexts = tuple((20000, 23000, 27000, 30000)[i % 4] for i in range(31))
    for spec in (cold, history):
        assert spec.qlens == (4,) * 31 + (16260,)
        assert len(spec.qlens) == 32 and sum(spec.qlens) == 16384
        assert spec.contexts[:31] == contexts
        assert spec.splits == 1 and spec.dim == 256 and spec.kv_heads == 1
    assert cold.contexts[-1] == 0
    assert history.contexts[-1] == 13740
    assert history.contexts[-1] + history.qlens[-1] == 30000


def test_cold_zero_context_and_derived_padded_decode_preserve_qkv_and_holes():
    torch = pytest.importorskip("torch")
    def small(context):
        spec = mixed.reuse.Shape("small", (4, 4, 20), (17, 65, context),
                                 64, 1, 1, False)
        logical = hashlib.sha256()
        def observe(request, _start, q, old_k, old_v, ck, cv):
            if request < 2:
                logical.update(mixed._logical_request_hash(
                    request, q, old_k, old_v, ck, cv))
        fixture = mixed.reuse.make_fixture(torch, spec, on_request=observe)
        fixture["logical_decode_input_sha256"] = logical.hexdigest()
        return fixture
    cold, warm = small(0), small(13)
    assert cold["logical_decode_input_sha256"] == warm["logical_decode_input_sha256"]
    for key in ("q", "ck", "cv"):
        assert torch.equal(cold["cpu"][key][:8], warm["cpu"][key][:8])
    pure = mixed.derive_decode_fixture(torch, warm, requests=2,
                                       padded_tokens=16, splits=3)
    assert pure["tokens"] == 16 and pure["actual_tokens"] == 8
    assert pure["spec"].name == "decode_from_small_n16_s3"
    assert pure["cpu"]["starts"].tolist() == [0, 4, 8]
    assert pure["cpu"]["slots"][8:].tolist() == [-1] * 8
    assert len(pure["expected"]) == 8
    assert pure["cpu"]["raw"] is warm["cpu"]["raw"]
    assert pure["cpu"]["wk"] is warm["cpu"]["wk"]
    for key in ("q", "ck", "cv"):
        assert torch.equal(pure["cpu"][key][:8], warm["cpu"][key][:8])


def test_preparation_preserves_source0_leader_evidence_and_uses_hadamard(monkeypatch):
    torch = pytest.importorskip("torch")
    spec = mixed.reuse.Shape("tiny", (1,), (17,), 64, 1, 1, False)
    fixture = mixed.reuse.make_fixture(torch, spec)
    tensors = {key: tensor.clone() for key, tensor in fixture["cpu"].items()}
    buffers = {"cv": {}, "rotation_status": torch.full((1, spec.heads), -99,
                                                    dtype=torch.int32)}
    calls = []
    monkeypatch.setattr(mixed.reuse, "_prepare", lambda *_: {
        "task_sha256": "tasks", "source0_leaders": 7,
        "source0_distinct_kvend": 2})
    class Ops:
        def rotate_out(self, query, rk, output, status, hadamard, slots):
            calls.append((hadamard, tuple(query.shape), tuple(slots.shape)))
            output.copy_(query.float() @ rk.T)
            status.zero_()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None),
                        raising=False)
    result = mixed._prepare_rotation(torch, Ops(), fixture, tensors, buffers, 20)
    assert result["task_sha256"] == "tasks"
    assert result["source0_leaders"] == 7
    assert result["source0_distinct_kvend"] == 2
    assert calls == [(True, (1, 6, 64), (1,))]


def test_mixed_pipeline_stage_order_and_pure_decode_omits_current(monkeypatch):
    torch = pytest.importorskip("torch")
    from oscar_ascend.integration import current_attention as current
    stages = []
    class Event:
        clock = 0
        def __init__(self, **_kwargs):
            pass
        def record(self, _stream):
            Event.clock += 1
            self.tick = Event.clock
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return float(other.tick - self.tick)
    stream = object()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        current_stream=lambda: stream, Event=Event), raising=False)
    monkeypatch.setattr(mixed.reuse, "_poison", lambda *_: None)
    monkeypatch.setattr(mixed.reuse, "_launch",
                        lambda *_args, **_kwargs: stages.append("cv:unified"))
    monkeypatch.setattr(current, "suppress_current_source_tasks",
                        lambda *_: stages.append("suppress"))
    monkeypatch.setattr(current, "guard_current_slots",
                        lambda *_: stages.append("guard"))
    monkeypatch.setattr(current, "native_current_partial",
                        lambda q, _k, _v, _lengths, **_kw:
                        (stages.append("current") or torch.zeros_like(q),
                         torch.zeros((*q.shape[:2], 1), dtype=torch.float32)))
    monkeypatch.setattr(current, "write_current_partial",
                        lambda *_: stages.append("write"))
    monkeypatch.setattr(mixed, "_current_source_written", lambda *_: None)
    spec = mixed.reuse.Shape("stage", (1,), (17,), 64, 1, 1, False)
    fixture = {"spec": spec, "tokens": 1, "actual_tokens": 1,
               "scale": 0.125}
    tensors = {"q": torch.zeros((1, 6, 64), dtype=torch.bfloat16),
               "ck": torch.zeros((1, 1, 64), dtype=torch.bfloat16),
               "cv": torch.zeros((1, 1, 64), dtype=torch.bfloat16),
               "slots": torch.zeros(1, dtype=torch.int64)}
    cv = {"tasks": torch.zeros((3, 16), dtype=torch.int64),
          "partial": torch.zeros((1, 6, 3, 64), dtype=torch.float32),
          "lse": torch.zeros((1, 6, 3), dtype=torch.float32)}
    buffers = {"cv": cv, "output": torch.zeros((6, 64)),
               "output_lse": torch.zeros(6),
               "merge_status": torch.zeros(6, dtype=torch.int32)}
    ops = SimpleNamespace(merge_lse_out=lambda *_: stages.append("merge"))
    row = mixed._run_once(torch, ops, fixture, tensors, buffers, 20,
                          mixed=True, pristine_tasks=cv["tasks"].clone(),
                          cumulative=(1,))
    assert stages == ["suppress", "cv:unified", "guard", "current", "write", "merge"]
    assert row["total_ms"] > row["cv_ms"] > 0
    stages.clear()
    row = mixed._run_once(torch, ops, fixture, tensors, buffers, 20, mixed=False)
    assert stages == ["cv:unified", "merge"]
    assert row["current_ms"] == 0.0


def test_failure_reports_rc2_and_result_filename_cannot_alias_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(mixed, "probe", lambda *_: (_ for _ in ()).throw(
        mixed.MixedAttentionProbeError("native current ABI mismatch")))
    output = tmp_path / "mixed-attention-report.json"
    assert mixed.main(["--output", str(output)]) == 2
    assert json.loads(output.read_text())["status"] == "failed"
    with pytest.raises(SystemExit) as error:
        mixed.main(["--output", str(tmp_path / "mixed-attention.json")])
    assert error.value.code == 2
