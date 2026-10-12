"""Archive #137/#152/#155: synthetic admission work is conserved exactly."""

import ast
from pathlib import Path
from types import SimpleNamespace

from tools.experiments.admission_replay import (
    WHOLE_LIMIT, align_split, distribute_rounds, scenarios, simulate,decode_routes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_mixed_selector_exposes_short_decodes_to_cluster_path():
    requests=[{'qlen':4,'context':20000,'decode':True} for _ in range(31)]
    old,new=decode_routes(requests+[{'qlen':4096,'context':15904,'decode':False}])
    assert old=={'attention_cv_striped_cluster4_out':31}
    assert new=={'attention_cv_bundle_decode_out':31}
    old,new=decode_routes(requests+[{'qlen':20000,'context':0,'decode':False}])
    assert old=={'attention_cv_striped_cluster16_out':31}
    assert new=={'attention_cv_bundle_decode_out':31}


def native_align_split(prompt: int, already: int, proposed: int, block: int) -> int:
    tree = ast.parse((ROOT / "references/vllm/vllm/v1/core/sched/scheduler.py").read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "_mamba_block_aligned_split")
    namespace = {"Request": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<pinned native align>", "exec"), namespace)
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=block), use_eagle=True)
    request = SimpleNamespace(num_computed_tokens=already, num_prompt_tokens=prompt, num_tokens=prompt)
    return namespace[function.name](scheduler, request, proposed)


def test_mtp_eagle_align_leaves_extra_block_and_tail():
    assert WHOLE_LIMIT == 32256
    assert align_split(20000, 0, 20000, None) == 20000
    assert align_split(20000, 0, 20000, 512) == 19456
    assert align_split(20000, 19456, 544, 512) == 544
    assert align_split(20000, 0, 20000, 1024) == 18432
    for block in (512, 1024):
        for prompt, already, proposed in ((20000, 0, 20000), (20000, 0, 16384),
                                          (20000, 0, 12000), (20000, 19456, 544),
                                          (30000, 0, 30000), (65000, 16384, 16384)):
            assert align_split(prompt, already, proposed, block) == native_align_split(
                prompt, already, proposed, block)


def test_whole_admission_removes_only_prior_chunk_history_and_preserves_tokens():
    lengths = [20000]
    rounds = [16]
    old = simulate(lengths, rounds, policy="old_greedy_16k", align_block=None, client_concurrency=1)
    whole = simulate(lengths, rounds, policy="whole_32k", align_block=None, client_concurrency=1)
    aligned = simulate(lengths, rounds, policy="whole_32k", align_block=512, client_concurrency=1)
    assert old["historical_query_prior_key_pairs"] == 16384 * 3616
    assert whole["historical_query_prior_key_pairs"] == 0
    assert aligned["historical_query_prior_key_pairs"] == 19456 * 544
    assert aligned["prefill_chunk_events"] == 2
    total_causal_pairs = 20000 * 20001 // 2
    for arm in (old, whole, aligned):
        assert arm["total_scheduled_prompt_tokens"] == 20000
        assert arm["total_scheduled_decode_q4_tokens"] == 4 * 16
        assert arm["late_two_draft_input_token_work"] == 2 * (20000 + 4 * 16)
        assert (arm["historical_query_prior_key_pairs"] + arm["prefill_current_causal_pairs"]
                == total_causal_pairs)
    # Larger first chunks trade historical pairs for current pairs. Omitting
    # first MTP's existing FP32 current path would overstate the saving.
    assert whole["prefill_current_causal_pairs"] > old["prefill_current_causal_pairs"]
    assert old["remaining_first_draft_current_fp32_pairs_if_fresh_fia"] == 3616 * 3617 // 2
    assert whole["remaining_first_draft_current_fp32_pairs_if_fresh_fia"] == 0
    assert aligned["remaining_first_draft_current_fp32_pairs_if_fresh_fia"] == 544 * 545 // 2


def test_long_request_retains_16k_chunks_and_round_distribution_is_exact():
    rounds = distribute_rounds(127, 33005, "staggered")
    assert sum(rounds) == 33005
    assert min(rounds) > 0
    result = simulate([65000], [16], policy="whole_32k", align_block=None, client_concurrency=1)
    assert result["prefill_chunk_events"] == 4
    assert result["peak_scheduled_tokens_N"] <= 32768
    same_mean = scenarios()["same_mean_20k_with_32x40k_synthetic"]
    assert len(same_mean) == 127 and sum(same_mean) == 127 * 20000
    assert sum(length > WHOLE_LIMIT for length in same_mean) == 32
