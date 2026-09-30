"""Host contracts for the paired striped writer/reader device gate.

Archive #126/#145/#148-154: preserve device error, graph and evidence boundaries.
These tests exercise case geometry, ABI ordering and fail-closed behavior.
They do not claim NPU completion, graph replay or speed acceptance.
"""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from contextlib import contextmanager

import pytest
import torch

from oscar_ascend.ops.cv_dispatch import STRIPED_CV_OPS
from tools import probe_striped_cache as striped
from tools import probe_history_reuse as reuse


def test_case_plan_exercises_real_q4_q1_graph_and_all_reader_variants():
    cases = striped.case_plan()
    names = [case.shape.name for case in cases]
    assert len(cases) == 12 and len(set(names)) == 12
    assert {case.shape.name for case in cases if case.graph} == {
        "q4_32_20k30k", "q1_32_20k30k"}
    assert {case.shape.name for case in cases if case.timed} == {
        "q4_32_20k30k", "q1_32_20k30k", "q4_32_65k", "q1_32_65k",
        "q4_padded16k", "c16_mature20k"}
    assert {case.shape.name for case in cases if case.shape.corrupt_metadata} == {
        "invalid_live_metadata"}
    assert {case.shape.name for case in cases if case.shape.corrupt_dead_tail} == {
        "invalid_dead_tail"}
    assert {case.new for case in cases if case.shape.corrupt_metadata or
            case.shape.corrupt_dead_tail} == {"attention_cv_striped_decode_out"}
    assert all(case.shape.dim == 256 and case.new in STRIPED_CV_OPS for case in cases)
    q4 = cases[0].shape
    q1 = cases[1].shape
    assert sum(q4.qlens) == 128 and q4.padded_tokens is None
    assert sum(q1.qlens) == 32 and q1.padded_tokens == 128 and q1.slot_context


def test_reader_abi_inserts_stats_only_for_clustered_symbols():
    shape = reuse.Shape("abi", (4,), (511,), 256, 1, 3, False)
    fixture = {"spec": shape, "blocks": 1, "stride": 2304 * 136, "scale": 256**-0.5}
    tensors = {key: object() for key in ("q", "qr", "ck", "cv", "rv", "raw",
                                         "table", "wk", "wv", "tags")}
    buffers = {key: object() for key in ("tasks", "partial", "lse", "status",
                                         "workspace", "cluster_stats")}
    calls = []

    class Ops:
        def __getattr__(self, name):
            return lambda *args: calls.append((name, args))

    for name, cluster in (("attention_cv_striped_decode_out", 1),
                          ("attention_cv_striped_q1_out", 1),
                          ("attention_cv_striped_cluster4_out", 4),
                          ("attention_cv_striped_cluster16_out", 16)):
        striped._launch(Ops(), name, tensors, fixture, buffers, 20, cluster)
        called, args = calls[-1]
        assert called == name
        assert args[0] is tensors["q"] and args[5] is tensors["raw"]
        assert args[10] is buffers["tasks"] and args[14] is buffers["workspace"]
        if cluster > 1:
            assert len(args) == 26 and args[15] is buffers["cluster_stats"]
        else:
            assert len(args) == 25 and buffers["cluster_stats"] not in args


def test_invalid_live_error_compares_task_and_value_class_without_nan_bits():
    old = {"status": torch.tensor([[3, 0], [0, 0]], dtype=torch.int32),
           "partial": torch.tensor([float("nan"), 1.0, -0.0]),
           "lse": torch.tensor([float("nan"), 2.0])}
    new = {"status": torch.tensor([[0, 3], [0, 0]], dtype=torch.int32),
           "partial": torch.tensor([float("nan"), 1.0, -0.0]),
           "lse": torch.tensor([float("nan"), 2.0])}
    assert striped._check_invalid(torch, old, new)["error_task_parity"] == "passed"
    new["partial"][1] = 2.0
    with pytest.raises(RuntimeError, match="finite partial bits"):
        striped._check_invalid(torch, old, new)
    new["partial"][1] = 1.0
    new["status"].zero_()
    with pytest.raises(Exception, match="error|invalid"):
        striped._check_invalid(torch, old, new)


def test_writer_reader_retargets_unique_existing_source0_owner():
    shape = reuse.Shape("striped_store", (385,), (641,), 256, 1, 1, True)
    fixture = {"spec": shape, "tokens": 385, "actual_tokens": 385}
    tasks = torch.zeros((385 * 3, 16), dtype=torch.int64)
    row = tasks.view(385, 1, 3, 1, 16)[378, 0, 0, 0]
    row[0], row[1], row[7] = 378, 7, 0
    before = tasks.clone()
    task_id, begin, end, owner, count = striped._retarget_reader_task(tasks, fixture)
    assert (task_id, begin, end, owner, count) == (378 * 3, 641, 1026, 378, 7)
    assert row[3:5].tolist() == [641, 1026] and row[8] == 1026
    before[task_id, 3:5] = tasks[task_id, 3:5]
    before[task_id, 8] = tasks[task_id, 8]
    assert torch.equal(before, tasks)  # no second leader or unrelated task changed


def test_no_npu_is_explicit_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(striped.reuse, "_active_device", lambda _target: None)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        set_device=lambda _index: None, is_available=lambda: False), raising=False)
    with pytest.raises(RuntimeError, match="target NPU required"):
        striped.probe(Path(striped.ROOT / "configs/target.json"),
                      Path(striped.ROOT / "configs/acceptance.json"))


def test_store_graph_gate_mutates_fixed_kv_and_replays_eager_equivalently(monkeypatch):
    active = {"graph": None}

    class Graph:
        def replay(self):
            self.fn()

    @contextmanager
    def capture(graph, **_kwargs):
        active["graph"] = graph
        try:
            yield
        finally:
            active["graph"] = None

    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        NPUGraph=Graph, graph=capture, synchronize=lambda: None), raising=False)
    monkeypatch.setattr(striped, "raw_to_striped", lambda _fixture:
                        torch.zeros(136, dtype=torch.uint8))
    fixture = {"tokens": 1, "blocks": 1, "stride": 136,
               "spec": SimpleNamespace(recent_tokens=256)}
    tensors = {"ck": torch.tensor([2.0]), "cv": torch.tensor([3.0]),
               "wk": torch.zeros(1), "wv": torch.zeros(1),
               "tags": torch.zeros(1, dtype=torch.int64),
               "slots": torch.zeros(1, dtype=torch.int64)}
    buffers = {"positions": torch.zeros(1, dtype=torch.int64)}

    class Ops:
        def rotate_clip_store_striped_out(self, k, v, _rk, _rv, _slots,
                                           _positions, raw, wk, wv, tags, status,
                                           *_attrs):
            def write():
                raw[0] = int((k + v).item()) & 255
                wk[0], wv[0], tags[0] = k[0], v[0], 1
                status.zero_()
            if active["graph"] is not None:
                active["graph"].fn = write
            else:
                write()

    evidence = striped._store_changed_input_graph(
        torch, Ops(), torch.device("cpu"), fixture, tensors, buffers,
        torch.eye(1), torch.eye(1))
    assert evidence == {"capture": "passed", "replay": "passed",
                        "same_address_changed_key_value": True,
                        "eager_parity": "bitwise_passed"}
