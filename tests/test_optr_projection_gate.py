"""Archive #170/#173/#175/#176: BF16 projection boundaries can reverse ranking.

All operands in the counterexamples are exactly BF16-representable. D256,
GQA6, TP4 padding embeds a two-coordinate example without introducing a
GEMM/reduction-tree ambiguity. These are CPU arithmetic contracts, not NPU
or model accuracy evidence. Attention itself is deliberately held fixed to
isolate W_O output and TP output rounding from the separate query regression.
"""
import copy

import pytest
import torch

from oscar_ascend import optr_calibration as calibration
from oscar_ascend.optr_artifacts import nonworse
from oscar_ascend.optr_policy import optr_policy
from oscar_ascend.optr_traces import aggregate_layer_error


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def projection_case(kind):
    heads, dim, group = 4, 256, 6
    samples, outputs = [], {choice: [] for choice in ("baseline", "selected")}
    for head in range(heads):
        teacher = torch.zeros(1, group, dim, dtype=torch.bfloat16)
        baseline, selected = teacher.clone(), teacher.clone()
        wo = torch.zeros(group, 1, dim)
        if kind == "rank" and head == 0:
            teacher[0, 0, :2] = torch.tensor([1., 1.])
            baseline[0, 0, :2] = torch.tensor([1., .5])
            selected[0, 0, :2] = torch.tensor([1., 1.125])
            wo[0, 0, :2] = torch.tensor([1., 1 / 256])
        elif kind == "tp" and head < 2:
            teacher[0, 0, 0] = [1., 1 / 256][head]
            baseline[0, 0, 0] = [1., 1 / 512][head]
            selected[0, 0, 0] = [1., 9 / 2048][head]
            wo[0, 0, 0] = 1.
        sample = {"request_id": "validation", "q": torch.zeros_like(teacher),
            "k": torch.zeros(1, dim, dtype=torch.bfloat16),
            "v": torch.zeros(1, dim, dtype=torch.bfloat16),
            "teacher": teacher, "gate": torch.ones_like(teacher), "wo": wo,
            "positions": torch.tensor([0]), "contexts": torch.tensor([0]),
            "current_ends": torch.tensor([1]), "frame_ids": torch.tensor([0]),
            "token_ids": torch.tensor([7]), "sample_sha256": f"{head:064x}"}
        samples.append([sample])
        outputs["baseline"].append(baseline)
        outputs["selected"].append(selected)
        assert torch.equal(wo, wo.bfloat16().float())
    base = torch.eye(dim).repeat(heads, 1, 1)
    return {"Rk": base, "Rv": base, "k_mean": torch.zeros(heads, dim)}, samples, outputs


@pytest.mark.parametrize("kind", ["rank", "tp"])
def test_fp32_improvement_cannot_hide_bf16_projection_regression(monkeypatch, kind):
    constants, samples, outputs = projection_case(kind)
    replay_calls, projection_calls = [], []
    actual_projection = calibration.project_output
    active = "baseline"

    def replay(rk, rv, mean, sample, policy):
        head = int(sample["sample_sha256"], 16)
        replay_calls.append((active, head))
        return outputs[active][head].float()

    def project(attention, sample):
        projection_calls.append(1)
        return actual_projection(attention, sample)

    monkeypatch.setattr(calibration, "replay_sample", replay)
    monkeypatch.setattr(calibration, "project_output", project)
    before = aggregate_layer_error(constants, samples, optr_policy({}))
    active = "selected"
    after = aggregate_layer_error(constants, samples, optr_policy({}))
    # Exactly one replay and two W_O products (actual/teacher) per rank:
    # collecting the additional views never replays attention or W_O.
    assert len(replay_calls) == 8 and len(projection_calls) == 16
    assert before["squared_error"] == 2 ** -18
    assert after["squared_error"] == 2 ** -22
    fp32_before, fp32_after = copy.deepcopy(before), copy.deepcopy(after)
    fp32_before.pop("projection_checks"); fp32_after.pop("projection_checks")
    assert nonworse(fp32_before, fp32_after, require_queries=True)
    for name in ("bf16_output", "bf16_rank_output"):
        assert before["projection_checks"][name]["squared_error"] == 0.
        assert after["projection_checks"][name]["squared_error"] == 2 ** -14
    assert not nonworse(before, after, require_queries=True, require_projection=True)
    assert nonworse(before, copy.deepcopy(before), require_queries=True, require_projection=True)


def test_projection_only_test_regression_reports_view_and_query(tmp_path):
    from oscar_ascend.optr_checkpoint import FitCheckpoints
    from tools.prepare_optr_rotations import _report_independent_test_failure
    from test_optr_integration import metric
    before, after = metric("test", 1.), metric("test", .5)
    # Synthetic certificate data isolates reporting, not arithmetic.
    for view in after["projection_checks"].values():
        view["squared_error"] = 2.
        view["normalized_error"] = .5
        request = view["requests"][0]
        frame = request["frames"][0]
        for row in (request, frame, frame["terminal_query"], frame["queries"][0]):
            row.update(squared_error=2., normalized_error=.5)
    events = []
    report = _report_independent_test_failure("layer", 3, {"test": {"baseline": before, "selected": after}},
        optr_policy({}), FitCheckpoints(tmp_path, {"test_only": True}), lambda **row: events.append(row))
    assert len(report["regressed_projection_views"]) == 2
    assert all(row["regressed_queries"][0]["position"] == 0 for row in report["regressed_projection_views"])
    assert sum(event["phase"] == "independent_test_projection_regressed" for event in events) == 2
    assert report["artifact_publish_allowed"] and not report["test_used_for_selection"]
