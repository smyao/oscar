"""Archive #170/#173/#175/#176: don't lose a checked OptR V checkpoint.

This deliberately SYNTHETIC CPU fixture has zero prefill queries and pairs of
opposite V rows between sampled positions. This isolates the exact prefill
from the nonzero decode quantization error; it is not a model trace or model
quality evidence. Teachers use unquantized dense causal attention, and all
splits use independent random tensors with the same per-head W_O coefficients.

The actual shared fitter runs the unchanged official 80-step K then V search.
Only _score is wrapped to copy matrices it already evaluates. No loss,
gradient, optimizer step, replay output or acceptance result is injected.
"""
from contextlib import contextmanager

import pytest
import torch

from oscar_ascend import optr_shared
from oscar_ascend.optr_artifacts import nonworse
from tools.prepare_optr_rotations import _metrics
from tools import prepare_optr_rotations as preparation
from test_optr_calibration import trace
from test_optr_upstream_parity import options


@contextmanager
def actual_candidate_case():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy = options()
        assert (policy["steps"], policy["eval_every"], policy["lr"]) == (80, 20, .02)
        baseline = {"layer_id": 3, "Rk": torch.eye(8).repeat(2, 1, 1),
                    "Rv": torch.eye(8).repeat(2, 1, 1), "k_mean": torch.zeros(2, 8)}
        seed = 2
        positions = torch.tensor([i * 47 // 7 for i in range(8)] + [48])
        weights = [trace(seed * 50 + head)["wo"].bfloat16().float() for head in range(2)]
        samples = {}
        for split_index, split in enumerate(("train", "validation", "test")):
            heads = []
            for head in range(2):
                sample = trace(seed * 50 + split_index * 7 + head, split,
                               length=49, dim=8, queries=9, group=2)
                sample["q"][:8].zero_()
                values = sample["v"].clone()
                sample["v"].zero_()
                for index in (1, 15, 28, 42):
                    sample["v"][index] = values[index]
                    sample["v"][index + 1] = -values[index]
                logits = torch.einsum("mgd,ld->mgl", sample["q"].float(), sample["k"].float()) / 8 ** .5
                logits.masked_fill_(torch.arange(49)[None, None] > positions[:, None, None], -torch.inf)
                sample.update(
                    teacher=torch.einsum("mgl,ld->mgd", logits.softmax(-1), sample["v"].float()).bfloat16(),
                    positions=positions, contexts=torch.tensor([0] * 8 + [48]),
                    current_ends=torch.tensor([48] * 8 + [49]), frame_ids=torch.tensor([0] * 8 + [1]),
                    history_splits=torch.ones(9, dtype=torch.int64), token_ids=torch.arange(49),
                    wo=weights[head], sample_sha256=f"{seed * 10 + split_index * 2 + head:064x}")
                heads.append([sample])
            samples[split] = heads
        evaluated = {"k": [], "v": []}
        original_score = optr_shared._score

        def record(rotation, rows, phase, configuration, emit):
            score = original_score(rotation, rows, phase, configuration, emit)
            evaluated[phase].append({"rotation": rotation.detach().clone(), "proxy_loss": score})
            return score

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(optr_shared, "_score", record)
            rk, rv, decision = optr_shared.fit_shared(
                baseline["Rk"], baseline["Rv"], baseline["k_mean"],
                samples["train"], samples["validation"], policy=policy)
        steps = [0, 1, 21, 41, 61, 80]
        for phase in ("k", "v"):
            assert len(evaluated[phase]) == len(steps)
            for checkpoint, step in zip(evaluated[phase], steps, strict=True):
                checkpoint["step"] = step
        proposal = {**baseline, "Rk": rk, "Rv": rv}
        before = _metrics(baseline, samples["validation"], policy)
        final = _metrics(proposal, samples["validation"], policy)
        candidates = []
        for checkpoint in evaluated["v"]:
            # Keep the official selected K fixed throughout, just as V fitting
            # did. These are existing evaluations, never new matrix mixtures.
            constants = {**proposal, "Rv": checkpoint["rotation"].repeat(2, 1, 1)}
            candidates.append({**checkpoint, "constants": constants,
                "validation": _metrics(constants, samples["validation"], policy)})
        yield {"baseline": baseline, "proposal": proposal, "samples": samples, "policy": policy,
               "decision": decision, "evaluated": evaluated, "candidates": candidates,
               "before": before, "final": final}
    finally:
        torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def fitted_case():
    with actual_candidate_case() as case:
        yield case


def passes_v8(before, candidate):
    return nonworse(before, candidate, require_frames=True, require_terminal=True,
                    require_queries=True, require_projection=True) and candidate["squared_error"] < before["squared_error"]


def test_actual_80_step_proxy_winner_is_rejected_but_existing_v_checkpoint_passes_all_v8_gates(fitted_case):
    case = fitted_case
    assert not passes_v8(case["before"], case["final"])
    alternative = next(row for row in case["candidates"] if row["step"] == 61)
    assert passes_v8(case["before"], alternative["validation"])
    assert alternative["proxy_loss"] > case["decision"]["phases"][1]["selected_proxy_loss"]
    assert alternative["validation"]["squared_error"] < case["before"]["squared_error"] * .5
    assert case["final"]["squared_error"] > case["before"]["squared_error"]
    assert set(alternative["validation"]["projection_checks"]) == {"bf16_output", "bf16_rank_output"}
    for view in [alternative["validation"], *alternative["validation"]["projection_checks"].values()]:
        assert {row["partition"] for row in view["requests"]} == {"single_split", "captured_runtime_frames"}
        assert all("queries" in frame for row in view["requests"] for frame in row["frames"])


def test_checkpoint_evidence_preserves_official_selected_k_and_excludes_independent_test(fitted_case):
    case = fitted_case
    selected_k = next(row["rotation"] for row in case["evaluated"]["k"] if row["step"] == 21)
    selected_v = next(row["rotation"] for row in case["evaluated"]["v"] if row["step"] == 1)
    assert torch.equal(case["proposal"]["Rk"], selected_k.repeat(2, 1, 1))
    assert torch.equal(case["proposal"]["Rv"], selected_v.repeat(2, 1, 1))
    for candidate in case["candidates"]:
        assert torch.equal(candidate["constants"]["Rk"], case["proposal"]["Rk"])
        assert torch.equal(candidate["constants"]["Rv"], candidate["constants"]["Rv"][:1].expand(2, 8, 8))
        assert all(request["request_id"].startswith("validation:")
                   for request in candidate["validation"]["requests"])


def test_real_refine_recovers_existing_checkpoint_and_tests_only_final_selection(fitted_case, monkeypatch):
    """Run the actual fitter and collector; observe, never replace, metrics."""
    case = fitted_case
    calls, events = [], []
    original_metrics = preparation._metrics
    from oscar_ascend.optr_artifacts import matrix_digest
    name = "model.layers.3.self_attn.attn"

    def observe(constants, samples, policy, **kwargs):
        calls.append((samples[0][0]["request_id"], matrix_digest({name: constants})))
        return original_metrics(constants, samples, policy, **kwargs)

    monkeypatch.setattr(preparation, "_metrics", observe)
    selected, record = preparation.refine_layer(name, case["baseline"], case["samples"], case["policy"],
                                               progress=lambda **row: events.append(row))
    expected = next(row for row in case["candidates"] if row["step"] == 61)
    assert record["accepted"]
    assert record["shared_layer_selection"]["selected_candidate"] == "v_step_61"
    assert [row["candidate"] for row in record["shared_layer_selection"]["evaluated_candidates"]] == [
        "primary", "v_step_0", "v_step_61"]
    assert torch.equal(selected["Rk"], case["proposal"]["Rk"])
    assert torch.equal(selected["Rv"], expected["constants"]["Rv"])
    assert passes_v8(record["validation"]["baseline"], record["validation"]["selected"])
    assert record["shared_decision"]["phases"] == case["decision"]["phases"]
    for phase in ("k", "v"):
        assert len([row for row in events if row.get("event") == "step_done" and row.get("phase") == phase]) == 80
    test_calls = [digest for split, digest in calls if split == "test"]
    assert test_calls == [matrix_digest({name: case["baseline"]}), matrix_digest({name: selected})]
    assert [split for split, _ in calls][-2:] == ["test", "test"]
    assert record["shared_layer_selection"]["test_used_for_selection"] is False
