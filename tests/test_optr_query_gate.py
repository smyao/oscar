"""Archive #170/#173/#175/#176: actual OptR proposals need every query checked.

Synthetic CPU captures below have complete K/V and two contiguous forward
frames. The real shared official-objective optimizer produces the proposal;
no replay output, metric, gradient or acceptance result is mocked. This is a
counterexample to aggregate acceptance, not model/NPU quality evidence.
"""
import copy

import pytest
import torch

from oscar_ascend.optr_artifacts import nonworse
from oscar_ascend.optr_calibration import project_output, replay_sample
from oscar_ascend.optr_shared import fit_shared
from tools.prepare_optr_rotations import _metrics, refine_layer
from test_optr_calibration import trace
from test_optr_upstream_parity import options


@pytest.fixture(scope="module")
def fitted_case():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy = options()  # Production upstream schedule: 80 steps / every 20.
        rng = torch.Generator().manual_seed(123)
        rk = torch.linalg.qr(torch.randn(8, 8, generator=rng)).Q
        rv = torch.linalg.qr(torch.randn(8, 8, generator=rng)).Q
        baseline = {"layer_id": 3, "Rk": rk.repeat(2, 1, 1),
                    "Rv": rv.repeat(2, 1, 1), "k_mean": torch.zeros(2, 8)}
        seed = 9
        weights = [trace(seed * 50 + head)["wo"].bfloat16().float() for head in range(2)]
        positions = torch.tensor([i * 76 // 7 for i in range(8)] + [77, 78, 79, 80])
        samples = {}
        for split_index, split in enumerate(("train", "validation", "test")):
            heads = []
            for head in range(2):
                sample = trace(seed * 50 + split_index * 7 + head, split,
                               length=81, dim=8, queries=12, group=2)
                logits = torch.einsum("mgd,ld->mgl", sample["q"].float(), sample["k"].float()) / 8 ** .5
                logits.masked_fill_(torch.arange(81)[None, None] > positions[:, None, None], -torch.inf)
                sample.update(
                    teacher=torch.einsum("mgl,ld->mgd", logits.softmax(-1), sample["v"].float()).bfloat16(),
                    positions=positions, contexts=torch.tensor([0] * 8 + [77] * 4),
                    current_ends=torch.tensor([77] * 8 + [81] * 4),
                    frame_ids=torch.tensor([0] * 8 + [1] * 4),
                    history_splits=torch.ones(12, dtype=torch.int64),
                    token_ids=torch.arange(81), wo=weights[head],
                    sample_sha256=f"{seed * 10 + split_index * 2 + head:064x}")
                heads.append([sample])
            samples[split] = heads
        rk, rv, decision = fit_shared(baseline["Rk"], baseline["Rv"], baseline["k_mean"],
            samples["train"], samples["validation"], policy=policy)
        proposal = {**baseline, "Rk": rk, "Rv": rv}
        assert decision["accepted"]
        assert torch.equal(rk, rk[:1].expand_as(rk)) and torch.equal(rv, rv[:1].expand_as(rv))
        yield baseline, proposal, samples, policy, decision
    finally:
        torch.set_num_threads(threads)


def _query_errors(constants, heads, policy):
    """Direct real replay; retain every query instead of reducing a frame."""
    actual = teacher = 0
    with torch.no_grad():
        for head, rows in enumerate(heads):
            sample = rows[0]
            output = replay_sample(constants["Rk"][head], constants["Rv"][head],
                                   constants["k_mean"][head], sample, policy)
            actual = actual + project_output(output, sample).double()
            teacher = teacher + project_output(sample["teacher"], sample).double()
    return (actual - teacher).square().sum(-1)


def _fp32_evidence(metric, *, keep_queries):
    """Select the original v7 metric's fields, independently of new domains."""
    scalars = ("squared_error", "teacher_energy", "normalized_error")
    result = {key: metric[key] for key in scalars}
    result["requests"] = []
    for request in metric["requests"]:
        row = {key: request[key] for key in (*scalars, "request_id", "sample_sha256")}
        row["frames"] = []
        for frame in request["frames"]:
            keys = (*scalars, "frame_id", "context", "current_end", "query_positions", "terminal_query")
            copied = {key: copy.deepcopy(frame[key]) for key in keys}
            if keep_queries:
                copied["queries"] = copy.deepcopy(frame["queries"])
            row["frames"].append(copied)
        result["requests"].append(row)
    return result


def test_real_shared_optr_proposal_passes_v7_gate_but_worsens_sampled_nonterminal_queries(fitted_case):
    baseline, proposal, samples, policy, _ = fitted_case
    before = _metrics(baseline, samples["validation"], policy)
    after = _metrics(proposal, samples["validation"], policy)
    assert nonworse(_fp32_evidence(before, keep_queries=False),
                    _fp32_evidence(after, keep_queries=False), require_frames=True, require_terminal=True)
    assert after["squared_error"] < before["squared_error"]
    previous = _query_errors(baseline, samples["validation"], policy)
    selected = _query_errors(proposal, samples["validation"], policy)
    # The full 80-step optimizer improves both frame totals and both terminal
    # queries, while query 77 worsens by ~26% and query 43 by ~290%.
    assert selected[8] > previous[8] * 1.2
    assert selected[4] > previous[4] * 3.5
    assert selected[:8].sum() < previous[:8].sum()
    assert selected[8:].sum() < previous[8:].sum()
    assert selected[7] < previous[7] and selected[-1] < previous[-1]


def test_v8_query_gate_rejects_actual_proposal_without_changing_optimizer(fitted_case):
    baseline, proposal, samples, policy, decision = fitted_case
    before = _metrics(baseline, samples["validation"], policy)
    after = _metrics(proposal, samples["validation"], policy)
    # Isolate the query safeguard: this comparison uses the same FP32
    # projection as v7, irrespective of the separate BF16 projection checks.
    assert not nonworse(_fp32_evidence(before, keep_queries=True),
                        _fp32_evidence(after, keep_queries=True), require_frames=True,
                        require_terminal=True, require_queries=True)

    def completed_fit(*args, **kwargs):
        return proposal["Rk"], proposal["Rv"], decision

    selected, record = refine_layer("model.layers.3.self_attn.attn", baseline, samples, policy,
                                   fitter=completed_fit, progress=lambda **kwargs: None)
    assert not record["accepted"]
    assert torch.equal(selected["Rk"], baseline["Rk"])
    assert torch.equal(selected["Rv"], baseline["Rv"])
