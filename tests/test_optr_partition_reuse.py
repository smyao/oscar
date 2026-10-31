"""Archive #170/#173/#175/#176: reuse only identical captured split arithmetic.

D256/GQA6/TP4 synthetic CPU captures exercise real replay and projection.
Call counts measure eliminated work, not NPU or model performance.
"""
import json

import pytest
import torch

from oscar_ascend import optr_calibration as calibration
from oscar_ascend import optr_traces as traces
from oscar_ascend.optr_policy import optr_policy
from tools import prepare_optr_rotations as prepare


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def capture():
    from test_optr_calibration import trace
    heads, dim, length = 4, 256, 338
    generator = torch.Generator().manual_seed(914)
    rk = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q
    rv = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q
    constants = {"Rk": rk.repeat(heads, 1, 1), "Rv": rv.repeat(heads, 1, 1),
                 "k_mean": torch.zeros(heads, dim)}
    samples = []
    for head in range(heads):
        sample = trace(914 + head, "validation", length=length, dim=dim, queries=4, group=6)
        positions = torch.tensor([0, 335, 336, 337])
        logits = torch.einsum("mgd,ld->mgl", sample["q"].float(), sample["k"].float()) / dim ** .5
        logits.masked_fill_(torch.arange(length)[None, None] > positions[:, None, None], -torch.inf)
        sample.update(positions=positions, contexts=torch.tensor([0, 0, 336, 337]),
            current_ends=torch.tensor([336, 336, 337, 338]), frame_ids=torch.tensor([0, 0, 1, 2]),
            history_splits=torch.ones(4, dtype=torch.int64), token_ids=torch.arange(length),
            sample_sha256=f"{914 + head:064x}", wo=sample["wo"].bfloat16().float(),
            teacher=torch.einsum("mgl,ld->mgd", logits.softmax(-1), sample["v"].float()).bfloat16())
        samples.append([sample])
    return constants, samples, optr_policy({})


def serialized(metric):
    return json.dumps(metric, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_one_split_reuses_real_d256_gqa6_tp4_replay_with_identical_bytes(monkeypatch):
    constants, samples, policy = capture()
    counts = {"replay": 0, "projection": 0}
    replay, project = calibration.replay_sample, calibration.project_output

    def counted_replay(*args, **kwargs):
        counts["replay"] += 1
        return replay(*args, **kwargs)

    def counted_project(*args, **kwargs):
        counts["projection"] += 1
        return project(*args, **kwargs)

    monkeypatch.setattr(calibration, "replay_sample", counted_replay)
    monkeypatch.setattr(calibration, "project_output", counted_project)
    with monkeypatch.context() as bypass:
        bypass.setattr(prepare, "_single_partition_equivalent", lambda *_: False)
        reference = prepare._metrics(constants, samples, policy)
    assert counts == {"replay": 8, "projection": 16}
    counts.update(replay=0, projection=0)
    events = []
    actual = prepare._metrics(constants, samples, policy, progress=lambda **row: events.append(row))
    assert counts == {"replay": 4, "projection": 8}
    assert serialized(actual) == serialized(reference)
    assert events == [{"event": "partition_checked", "partition": partition}
                      for partition in policy["validation_partitions"]]
    for view in [actual, *actual["projection_checks"].values()]:
        left, right = view["requests"]
        assert left["request_id"] == "validation:single_split"
        assert right["request_id"] == "validation:captured_runtime_frames"
        assert left["sample_sha256"] == right["sample_sha256"]
        assert left["frames"] == right["frames"] and left["frames"] is not right["frames"]
        assert len(left["frames"]) == 3 and len(left["frames"][0]["queries"]) == 2
        assert view["squared_error"] == left["squared_error"] + right["squared_error"]


def test_any_captured_multiple_split_executes_both_real_partitions(monkeypatch):
    constants, samples, policy = capture()
    samples[-1][0]["history_splits"][-1] = 2
    partitions = []
    aggregate = traces.aggregate_layer_error

    def counted(*args, **kwargs):
        partitions.append(args[2]["replay_partition"])
        return aggregate(*args, **kwargs)

    monkeypatch.setattr(traces, "aggregate_layer_error", counted)
    actual = prepare._metrics(constants, samples, policy)
    assert partitions == ["single_split", "captured_runtime_frames"]
    with monkeypatch.context() as bypass:
        bypass.setattr(prepare, "_single_partition_equivalent", lambda *_: False)
        reference = prepare._metrics(constants, samples, policy)
    assert serialized(actual) == serialized(reference)


@pytest.mark.parametrize("fault", ["missing_splits", "missing_ends", "split_dtype", "split_shape",
                                   "zero_split", "end_dtype", "end_shape", "end_bounds"])
def test_reuse_cannot_hide_invalid_or_missing_captured_metadata(monkeypatch, fault):
    constants, samples, policy = capture()
    sample = samples[-1][0]
    if fault == "missing_splits": sample.pop("history_splits")
    elif fault == "missing_ends": sample.pop("current_ends")
    elif fault == "split_dtype": sample["history_splits"] = sample["history_splits"].float()
    elif fault == "split_shape": sample["history_splits"] = sample["history_splits"][:1]
    elif fault == "zero_split": sample["history_splits"][0] = 0
    elif fault == "end_dtype": sample["current_ends"] = sample["current_ends"].float()
    elif fault == "end_shape": sample["current_ends"] = sample["current_ends"][:1]
    else: sample["current_ends"][-1] = 337
    with monkeypatch.context() as bypass:
        bypass.setattr(prepare, "_single_partition_equivalent", lambda *_: False)
        with pytest.raises(ValueError) as original:
            prepare._metrics(constants, samples, policy)
    with pytest.raises(ValueError) as actual:
        prepare._metrics(constants, samples, policy)
    assert str(actual.value) == str(original.value)


def test_reordered_or_single_partition_plan_still_evaluates_requested_views(monkeypatch):
    constants, samples, policy = capture()
    order = list(reversed(policy["validation_partitions"]))
    partitions = []
    aggregate = traces.aggregate_layer_error

    def counted(*args, **kwargs):
        partitions.append(args[2]["replay_partition"])
        return aggregate(*args, **kwargs)

    monkeypatch.setattr(traces, "aggregate_layer_error", counted)
    result = prepare._metrics(constants, samples, {**policy, "validation_partitions": order})
    assert partitions == order
    assert [row["partition"] for row in result["requests"]] == order
    partitions.clear()
    prepare._metrics(constants, samples, {**policy, "validation_partitions": [order[0]]})
    assert partitions == [order[0]]


def test_invalid_single_split_policy_is_not_bypassed():
    constants, samples, policy = capture()
    with pytest.raises(ValueError, match="invalid fixed source split count"):
        prepare._metrics(constants, samples, {**policy, "history_splits": 2})
