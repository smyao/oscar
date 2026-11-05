"""Archive #170/#175-177: faster offline replay cannot weaken any quality gate.

Compare complete decisions and the independent test against separate original
partition replays. CPU synthetic evidence is not production model accuracy.
"""
import copy
import json
import weakref

import pytest
import torch

from oscar_ascend import optr_calibration as calibration
from oscar_ascend import optr_trust_region as repair
from oscar_ascend.optr_artifacts import matrix_digest
from oscar_ascend.optr_policy import optr_policy
from test_optr_calibration import trace
from test_optr_native_residual import case
from test_optr_trust_region import basic_case, rotation
from tools.prepare_optr_native_residual import refine_layer
from tools.prepare_optr_rotations import IndependentTestRegression, _metrics


PARTITIONS = ["single_split", "captured_runtime_frames"]


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def serialized(metric):
    return json.dumps(metric, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def separate_partitions(constants, head, samples, policy, partitions):
    """The original request loop, with no prepared-history argument."""
    return {partition: repair._project_head(constants, head, samples,
            {**policy, "replay_partition": partition}) for partition in partitions}


@pytest.mark.parametrize("dim,splits", [(64, 2), (128, 3), (256, 2)])
def test_complete_metrics_reuse_history_once_per_request_and_release_it(monkeypatch, dim, splits):
    baseline, samples, policy = basic_case(dim=dim, heads=4, splits=splits)
    for head in samples:
        other = copy.deepcopy(head[0])
        other.update(request_id="validation_other", sample_sha256="e" * 64)
        other["v"].neg_()
        other["teacher"].neg_()
        head.append(other)
    candidate = {**baseline, "Rv": baseline["Rv"].clone()}
    candidate["Rv"][2] = candidate["Rv"][2] @ rotation(dim, .0025)
    expected_before = _metrics(baseline, samples, policy)
    expected_after = _metrics(candidate, samples, policy)
    prepare, codec = calibration._replay_history, calibration.quantize_dequantize
    histories, codec_calls = [], []

    def record(*args, **kwargs):
        history = prepare(*args, **kwargs)
        histories.append([weakref.ref(value) for value in history.values()])
        return history

    def count(*args, **kwargs):
        codec_calls.append(1)
        return codec(*args, **kwargs)

    monkeypatch.setattr(calibration, "_replay_history", record)
    monkeypatch.setattr(calibration, "quantize_dequantize", count)
    cache = repair.ProjectionCache(baseline, samples, policy)
    assert serialized(cache.before) == serialized(expected_before)
    assert len(histories) == 8 and len(codec_calls) == 16
    assert all(reference() is None for history in histories for reference in history)
    actual = cache.measure(candidate, 2)
    assert serialized(actual) == serialized(expected_after)
    assert len(histories) == 10 and len(codec_calls) == 20
    assert all(reference() is None for history in histories for reference in history)


@pytest.mark.parametrize("splits", [1, 2])
def test_complete_native_selection_and_independent_test_are_byte_identical(monkeypatch, splits):
    baseline, train, validation, _ = case(splits=splits)
    test = copy.deepcopy(validation)
    for head in test:
        for sample in head:
            sample.update(request_id="independent_test", split="test", sample_sha256="f" * 64)
    policy = optr_policy({"optr_baseline": "recipe3",
                          "optr_recipe3_optimizer": "joint_native_residual_v1"})
    policy.update(sink_tokens=2, recent_tokens=4, query_batch_size=2)
    samples = {"train": train, "validation": validation, "test": test}
    name = "model.layers.3.self_attn.attn"
    events = []
    actual_constants, actual = refine_layer(name, baseline, samples, policy,
                                            progress=lambda **row: events.append(row))
    with monkeypatch.context() as original:
        original.setattr(repair, "_project_head_partitions", separate_partitions)
        expected_constants, expected = refine_layer(name, baseline, samples, policy, progress=lambda **_: None)
    # Wall-clock telemetry is the sole intentionally different record field.
    actual["native_residual_selection"].pop("elapsed_seconds")
    expected["native_residual_selection"].pop("elapsed_seconds")
    assert serialized(actual) == serialized(expected)
    assert matrix_digest({name: actual_constants}) == matrix_digest({name: expected_constants})
    for key in ("Rk", "Rv", "k_mean"):
        assert torch.equal(actual_constants[key].view(torch.uint8), expected_constants[key].view(torch.uint8))
    assert actual["native_residual_selection"]["test_used_for_selection"] is False
    assert any(row.get("independent_test") == "passed" for row in events)
    if splits == 1:
        assert actual["accepted"]  # Exercise a changed PT, not only seed self-comparison.


def test_differentiable_partition_calls_keep_independent_backward_graphs(monkeypatch):
    sample = trace(seed=32, length=48, dim=8, queries=4)
    sample["history_splits"] = torch.full((4,), 2, dtype=torch.int64)
    rk = rotation(8, .12).requires_grad_()
    rv = rotation(8, -.07).requires_grad_()
    mean = torch.linspace(-.1, .1, 8).requires_grad_()
    policy = {"query_batch_size": 2, "sink_tokens": 2, "recent_tokens": 4}
    original = [calibration.replay_sample(rk, rv, mean, sample,
                {**policy, "replay_partition": partition}) for partition in PARTITIONS]
    for output in original:
        output.square().sum().backward()
    gradients = [value.grad.clone() for value in (rk, rv, mean)]
    for value in (rk, rv, mean):
        value.grad = None

    def forbidden(*args, **kwargs):
        pytest.fail("gradient replay must not share a history graph")

    monkeypatch.setattr(calibration._PreparedReplayHistory, "__init__", forbidden)
    actual = calibration.replay_sample_partitions(rk, rv, mean, sample, policy, PARTITIONS)
    for expected, partition in zip(original, PARTITIONS):
        assert torch.equal(actual[partition], expected)
        actual[partition].square().sum().backward()  # No retain_graph needed.
    assert all(torch.equal(value.grad, expected) for value, expected in zip((rk, rv, mean), gradients))


def test_independent_test_regression_is_still_rejected_with_identical_evidence(monkeypatch):
    baseline, train, validation, _ = case(splits=1)
    policy = optr_policy({"optr_baseline": "recipe3",
                          "optr_recipe3_optimizer": "joint_native_residual_v1"})
    policy.update(sink_tokens=2, recent_tokens=4, query_batch_size=2)
    test = copy.deepcopy(validation)
    for head, samples in enumerate(test):
        for sample in samples:
            sample.update(request_id="independent_test", split="test", sample_sha256="f" * 64)
            sample["teacher"] = calibration.replay_sample(baseline["Rk"][head], baseline["Rv"][head],
                baseline["k_mean"][head], sample, {**policy, "replay_partition": "single_split"}).bfloat16()
    samples = {"train": train, "validation": validation, "test": test}
    with pytest.raises(IndependentTestRegression, match="not published") as actual:
        refine_layer("model.layers.3.self_attn.attn", baseline, samples, policy, progress=lambda **_: None)
    with monkeypatch.context() as original:
        original.setattr(repair, "_project_head_partitions", separate_partitions)
        with pytest.raises(IndependentTestRegression, match="not published") as expected:
            refine_layer("model.layers.3.self_attn.attn", baseline, samples, policy, progress=lambda **_: None)
    for failure in (actual, expected):
        failure.value.report["record"]["native_residual_selection"].pop("elapsed_seconds")
    assert serialized(actual.value.report) == serialized(expected.value.report)


@pytest.mark.parametrize("fault", ["matrix", "matrix_mutation", "request", "request_id", "sample_mutation", "clip", "opaque"])
def test_prepared_history_cannot_cross_candidate_request_or_clip(fault):
    sample = trace(dim=8)
    rk, rv, mean = torch.eye(8), torch.eye(8), torch.zeros(8)
    policy = calibration._policy({})
    prepared = calibration._PreparedReplayHistory(rk, rv, mean, sample, policy)
    if fault == "matrix":
        rk = rk.clone()
    elif fault == "matrix_mutation":
        rk.add_(.00001)
    elif fault == "request":
        sample = dict(sample)
    elif fault == "request_id":
        sample["request_id"] = "different"
    elif fault == "sample_mutation":
        sample["k"].neg_()
    elif fault == "clip":
        policy["k_clip"] = .95
    else:
        prepared = prepared.values
    with pytest.raises(ValueError, match="prepared replay history"):
        calibration.replay_sample(rk, rv, mean, sample, policy, _history=prepared)


def test_inference_tensors_keep_the_original_replay_without_version_cache(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unversioned inference tensors cannot enter prepared history")

    monkeypatch.setattr(calibration._PreparedReplayHistory, "__init__", forbidden)
    with torch.inference_mode():
        sample = trace(dim=8)
        sample["history_splits"] = torch.full((4,), 2, dtype=torch.int64)
        identity, mean = torch.eye(8), torch.zeros(8)
        actual = calibration.replay_sample_partitions(identity, identity, mean, sample, {}, PARTITIONS)
        for partition in PARTITIONS:
            expected = calibration.replay_sample(identity, identity, mean, sample, {"replay_partition": partition})
            assert torch.equal(actual[partition], expected)


@pytest.mark.parametrize("dim,splits,changed", [(64, 1, [0, 2, 3]), (256, 2, [0, 2]), (64, 3, [0, 1, 2, 3])])
def test_multihead_cache_matches_full_metrics_in_original_head_order_and_releases_work(monkeypatch, dim, splits, changed):
    baseline, samples, policy = basic_case(dim=dim, heads=4, splits=splits)
    candidate = {**baseline, "Rk": baseline["Rk"].clone(), "Rv": baseline["Rv"].clone()}
    for head in changed:
        key = "Rk" if head % 2 == 0 else "Rv"
        candidate[key][head] = candidate[key][head] @ rotation(dim, .0025 * (head + 1))
    # Joint candidates can change a head's FP32 constants without crossing a
    # BF16 operand boundary. That head must remain eligible for baseline reuse.
    if 1 not in changed:
        candidate["Rk"][1, 0, 0] = torch.nextafter(candidate["Rk"][1, 0, 0], torch.tensor(float("inf")))
        assert torch.equal(candidate["Rk"][1].bfloat16(), baseline["Rk"][1].bfloat16())
    expected = _metrics(candidate, samples, policy)
    cache = repair.ProjectionCache(baseline, samples, policy)
    original = {part: [[value.clone() for value in head] for head in outputs]
                for part, outputs in cache.outputs.items()}
    replay, project = repair._project_head_partitions, calibration.project_output
    order, projections = [], []

    def counted(constants, head, *args, **kwargs):
        order.append(head)
        return replay(constants, head, *args, **kwargs)

    def track(*args, **kwargs):
        output = project(*args, **kwargs)
        projections.append(weakref.ref(output))
        return output

    monkeypatch.setattr(repair, "_project_head_partitions", counted)
    monkeypatch.setattr(calibration, "project_output", track)
    actual = cache.measure_many(candidate, changed)
    assert serialized(actual) == serialized(expected)
    assert order == changed
    assert all(reference() is None for reference in projections)
    for partition, outputs in cache.outputs.items():
        assert all(torch.equal(value, previous) for head, old in zip(outputs, original[partition])
                   for value, previous in zip(head, old))


@pytest.mark.parametrize("splits", [1, 2])
def test_multihead_api_single_head_is_identical_to_existing_measure(splits):
    baseline, samples, policy = basic_case(dim=64, heads=4, splits=splits)
    candidate = {**baseline, "Rv": baseline["Rv"].clone()}
    candidate["Rv"][2] = candidate["Rv"][2] @ rotation(64, .0025)
    cache = repair.ProjectionCache(baseline, samples, policy)
    assert serialized(cache.measure_many(candidate, [2])) == serialized(cache.measure(candidate, 2))


@pytest.mark.parametrize("heads", [[], [1, 0], [0, 0], [-1], [True], [4], (0,), [0, 2.]])
def test_multihead_api_rejects_noncanonical_or_invalid_head_list(heads):
    baseline, samples, policy = basic_case(heads=4)
    cache = repair.ProjectionCache(baseline, samples, policy)
    with pytest.raises(ValueError, match="changed_heads"):
        cache.measure_many(baseline, heads)


@pytest.mark.parametrize("key", ["Rk", "Rv", "k_mean"])
def test_multihead_api_rejects_undeclared_operand_changes(key):
    baseline, samples, policy = basic_case(heads=4)
    cache = repair.ProjectionCache(baseline, samples, policy)
    candidate = {**baseline, key: baseline[key].clone()}
    if key == "k_mean":
        candidate[key][1] += 1
    else:
        candidate[key][1] = candidate[key][1] @ rotation(8, .1)
    with pytest.raises(ValueError, match="undeclared head"):
        cache.measure_many(candidate, [0, 2])
