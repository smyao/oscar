"""Archive #148/#170/#173: joint/frame-aware subset gains without extra replay."""
import hashlib

import pytest
import torch

from oscar_ascend import optr_selection as selection


def case(monkeypatch, base_values, trial_values, *, second_partition=None, unchanged=()):
    heads = len(base_values)
    baseline = {"layer_id": 3, "Rk": torch.eye(2).repeat(heads, 1, 1),
                "Rv": torch.eye(2).repeat(heads, 1, 1), "k_mean": torch.zeros(heads, 2)}
    trial = {**baseline, "Rk": -baseline["Rk"], "Rv": baseline["Rv"].clone()}
    for head in unchanged:
        trial["Rk"][head] = baseline["Rk"][head]
    samples = [[{"request_id": "validation", "head": head,
                 "teacher": torch.zeros(2, 1), "sample_sha256": hashlib.sha256(str(head).encode()).hexdigest(),
                 "q": torch.zeros(2, 1, 2, dtype=torch.bfloat16), "wo": torch.zeros(1, 1, 2),
                 "positions": torch.tensor([7, 8]), "contexts": torch.tensor([0, 8]),
                 "current_ends": torch.tensor([8, 9]), "frame_ids": torch.tensor([0, 1]),
                 "token_ids": torch.arange(9)}] for head in range(heads)]
    calls, projections = [], []
    def replay(rk, rv, mean, sample, policy):
        partition = policy["replay_partition"]
        changed = bool(rk[0, 0] < 0)
        values = trial_values if changed else base_values
        if second_partition is not None and partition == "captured_runtime_frames":
            values = second_partition[1 if changed else 0]
        calls.append((partition, sample["head"], changed))
        return torch.tensor(values[sample["head"]], dtype=torch.float32).reshape(2, 1)
    def projected(value, sample):
        projections.append(sample["head"])
        return value
    monkeypatch.setattr(selection, "replay_sample", replay)
    monkeypatch.setattr(selection, "project_output", projected)
    policy = {"validation_partitions": ["single_split", "captured_runtime_frames"],
              "selection": "joint_all_heads_each_request_frame_and_terminal_nonworse"}
    return baseline, trial, samples, policy, calls, projections


def test_subset_recovers_joint_gain_when_all_proposals_regress_without_more_replay(monkeypatch):
    baseline, trial, samples, policy, calls, projections = case(
        monkeypatch, [[1, 1], [1, 1]], [[0, 0], [3, 3]])
    selected, before, after, record = selection.select_head_subset(baseline, trial, samples, policy)
    assert record["selected_heads"] == [0]
    assert record["combinations"] == 4 and record["test_used_for_selection"] is False
    assert before["squared_error"] == 16 and after["squared_error"] == 4
    assert record["candidate_errors"][3]["squared_error"] == 36
    assert torch.equal(selected["Rk"][0], trial["Rk"][0])
    assert torch.equal(selected["Rk"][1], baseline["Rk"][1])
    assert len(calls) == 8 and len(set(calls)) == 8  # 2 variants x 2 heads x 2 partitions
    assert len(projections) == 10  # 2 invariant teachers plus 8 replay outputs
    assert torch.equal(baseline["Rk"], torch.eye(2).repeat(2, 1, 1))


def test_subset_accounts_for_cross_head_cancellation(monkeypatch):
    args = case(monkeypatch, [[2, 2], [-1, -1]], [[3, 3], [-3, -3]])
    _, before, after, record = selection.select_head_subset(*args[:4])
    assert before["squared_error"] == 4 and after["squared_error"] == 0
    assert record["selected_heads"] == [0, 1]
    assert record["admissible_combinations"] == 1


def test_lower_total_cannot_hide_a_worse_decode_frame(monkeypatch):
    args = case(monkeypatch, [[2, .1]], [[1, .2]])
    selected, before, after, record = selection.select_head_subset(*args[:4])
    assert selected is args[0] and before == after
    assert record["candidate_errors"][1]["squared_error"] < before["squared_error"]
    assert record["selected_mask"] == 0 and not record["candidate_errors"][1]["accepted"]


def test_every_partition_must_accept_the_same_head_subset(monkeypatch):
    args = case(monkeypatch, [[2, 2]], [[0, 0]], second_partition=([[1, 1]], [[2, 2]]))
    selected, before, after, record = selection.select_head_subset(*args[:4])
    assert selected is args[0] and before == after and record["selected_heads"] == []
    assert record["candidate_errors"][1]["squared_error"] < before["squared_error"]


def test_tie_prefers_fewer_changed_heads_and_then_stable_mask(monkeypatch):
    args = case(monkeypatch, [[1, 1], [0, 0]], [[0, 0], [0, 0]])
    _, _, after, record = selection.select_head_subset(*args[:4])
    assert after["squared_error"] == 0 and record["selected_mask"] == 1
    args = case(monkeypatch, [[1, 1], [1, 1]], [[-1, -1], [-1, -1]])
    _, _, after, record = selection.select_head_subset(*args[:4])
    assert after["squared_error"] == 0 and record["selected_mask"] == 1


def test_unchanged_heads_are_not_replayed_twice_or_counted_as_proposals(monkeypatch):
    args = case(monkeypatch, [[1, 1], [1, 1]], [[0, 0], [1, 1]], unchanged=(1,))
    _, _, _, record = selection.select_head_subset(*args[:4])
    assert record["proposed_heads"] == [0] and record["combinations"] == 2
    assert len(args[4]) == 6  # two replays for changed head, one for unchanged, two partitions
    args = case(monkeypatch, [[1, 1]], [[1, 1]], unchanged=(0,))
    selected, before, after, record = selection.select_head_subset(*args[:4])
    assert selected is args[0] and before == after and record["combinations"] == 1
    assert len(args[4]) == 2


def test_four_head_search_stays_bounded_at_sixteen_combinations(monkeypatch):
    args = case(monkeypatch, [[1, 1]] * 4, [[0, 0]] * 4)
    _, _, after, record = selection.select_head_subset(*args[:4])
    assert record["combinations"] == 16 and record["selected_heads"] == [0, 1, 2, 3]
    assert after["squared_error"] == 0 and len(args[4]) == 16


@pytest.mark.parametrize("fault", ["mean", "rotation", "heads", "partition", "nonfinite"])
def test_invalid_subset_inputs_fail_closed(monkeypatch, fault):
    baseline, trial, samples, policy, *_ = case(monkeypatch, [[1, 1]], [[0, 0]])
    if fault == "mean": trial["k_mean"] = torch.ones_like(trial["k_mean"])
    if fault == "rotation": trial["Rk"] = trial["Rk"].double()
    if fault == "heads": samples = []
    if fault == "partition": policy["validation_partitions"] *= 2
    if fault == "nonfinite":
        monkeypatch.setattr(selection, "replay_sample", lambda *args: torch.full((2, 1), float("nan")))
    with pytest.raises(ValueError):
        selection.select_head_subset(baseline, trial, samples, policy)


def test_real_cpu_fitter_and_subset_replay_complete_validation_and_untouched_test(tmp_path):
    from test_optr_traces import build_bundle, load
    from oscar_ascend.optr_policy import optr_policy
    from oscar_ascend.optr_traces import load_layer_samples
    from oscar_ascend.optr_artifacts import nonworse
    from tools.prepare_optr_rotations import refine_layer
    path, config, _ = build_bundle(tmp_path, tp=2, dim=4, steps=1)
    samples = load_layer_samples(load(path, config), 3)
    baseline = {"layer_id": 3, "Rk": torch.eye(4).repeat(2, 1, 1),
                "Rv": torch.eye(4).repeat(2, 1, 1), "k_mean": torch.zeros(2, 4)}
    policy = {**optr_policy(config), "sink_tokens": 0, "recent_tokens": 1}
    events = []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        selected, record = refine_layer("model.layers.3.self_attn.attn", baseline, samples,
            policy, fit_device="cpu", progress=lambda **event: events.append(event))
    finally:
        torch.set_num_threads(previous)
    assert nonworse(record["validation"]["baseline"], record["validation"]["selected"], require_frames=True)
    assert record["head_subset_selection"]["test_used_for_selection"] is False
    assert all(row["frames"] for split in ("validation", "test")
               for row in record[split]["selected"]["requests"])
    validation_ids = {row["request_id"] for row in record["validation"]["selected"]["requests"]}
    test_ids = {row["request_id"] for row in record["test"]["selected"]["requests"]}
    assert validation_ids.isdisjoint(test_ids)
    chosen_event = next(i for i, event in enumerate(events) if event.get("event") == "subset_selected")
    test_event = next(i for i, event in enumerate(events) if event.get("split") == "test")
    assert chosen_event < test_event
    assert selected["Rk"].shape == (2, 4, 4) and selected["Rv"].dtype == torch.float32
