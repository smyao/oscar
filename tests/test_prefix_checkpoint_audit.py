"""Archive #26/#37-49/#127/#128/#157: real native sparse-checkpoint regression.

These cases execute native splitter/manager/pool code, unlike a seed that
artificially creates every 768-token checkpoint. Metadata evidence only.
"""

import pytest

from tools.experiments.prefix_checkpoint_audit import run_case, source_provenance


def test_no_mtp_alignment_mismatch_alone_can_erase_repeat_prompt_hit():
    kwargs = dict(prompt_tokens=15000, mtp=False)
    native = run_case(**kwargs, arm="native")
    oscar = run_case(**kwargs, arm="oscar_current")
    aligned = run_case(**kwargs, arm="oscar_joint_alignment_hypothesis")
    assert native["repeat_prompt_hit_tokens"] == 14592
    assert oscar["repeat_prompt_hit_tokens"] == 0
    assert oscar["cached_mamba_checkpoint_tokens"] == [14592]
    assert oscar["joint_aligned_mamba_checkpoint_tokens"] == []
    assert oscar["full_block_tokens"] == 2304
    assert oscar["mamba_speculative_blocks"] == 0
    assert aligned["repeat_prompt_hit_tokens"] == 13824
    assert aligned["hypothetical"] and not oscar["hypothetical"]


def test_real_final_chunk_cut_uses_native_768_but_lookup_requires_2304():
    row = run_case(prompt_tokens=32768, mtp=True, arm="oscar_current")
    assert [chunk["end"] for chunk in row["chunks"]] == [16128, 31488, 32768]
    assert row["native_final_cache_position"] == 31488
    assert row["cached_mamba_checkpoint_tokens"] == [16128, 31488]
    assert row["joint_aligned_mamba_checkpoint_tokens"] == [16128]
    assert row["repeat_prompt_hit_tokens"] == 16128
    fixed = run_case(prompt_tokens=32768, mtp=True, arm="oscar_joint_alignment_hypothesis")
    assert [chunk["end"] for chunk in fixed["chunks"]] == [16128, 29952, 32768]
    assert fixed["repeat_prompt_hit_tokens"] == 29952


def test_remaining_budget_is_distinct_from_frozen_global_budget():
    row = run_case(prompt_tokens=32768, mtp=False, arm="oscar_current",
                   remaining_budgets=(15360,))
    assert row["global_token_budget"] == 16384
    assert [chunk["end"] for chunk in row["chunks"]] == [15360, 30720, 32256, 32768]
    assert row["cached_mamba_checkpoint_tokens"] == [15360, 30720, 32256]
    assert row["joint_aligned_mamba_checkpoint_tokens"] == [32256]
    assert all(chunk["scheduled"] <= 15360 for chunk in row["chunks"])


def test_hypothetical_alignment_does_not_hide_small_budget_stall():
    row = run_case(prompt_tokens=32768, mtp=False,
                   arm="oscar_joint_alignment_hypothesis", remaining_budgets=(2048,))
    assert row["status"] == "blocked_by_repeated_insufficient_chunk_budget"
    assert row["completed_prompt_tokens"] == row["repeat_prompt_hit_tokens"] == 0
    assert row["chunks"][0]["scheduled"] == 0


@pytest.mark.parametrize("prompt", [4608, 6912, 9216])
@pytest.mark.parametrize("mtp", [False, True])
def test_naive_joint_alignment_is_not_a_complete_terminal_boundary_fix(prompt, mtp):
    row = run_case(prompt_tokens=prompt, mtp=mtp, arm="oscar_joint_alignment_hypothesis")
    assert row["status"] == "completed"
    assert row["repeat_prompt_hit_tokens"] == 0
    assert row["max_lookup_tokens"] == prompt - 1
    expected = [prompt - 2304, prompt] if mtp else [prompt]
    assert [chunk["end"] for chunk in row["chunks"]] == expected
    # This native terminal-boundary issue also exists without compression;
    # do not attribute every zero in the matrix uniquely to OSCAR.
    native = run_case(prompt_tokens=prompt, mtp=mtp, arm="native")
    assert native["repeat_prompt_hit_tokens"] == 0


def test_provenance_pins_executed_source_and_research_harness():
    provenance = source_provenance()
    assert len(provenance["git_head"]) == 40
    assert all(len(value) == 64 for value in provenance["sha256"].values())
    assert "references/vllm/vllm/v1/core/sched/scheduler.py" in provenance["sha256"]
    assert "tests/test_runtime_native.py" in provenance["sha256"]
    assert "oscar_ascend/integration/prefix_checkpoint.py" in provenance["sha256"]


@pytest.mark.parametrize("prompt", [4608, 6912, 9216, 15000, 16000, 32768, 65536])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("budgets", [(16384,), (15360,), (16384, 15360), (2048,),
                                    (768,), (2048, 768, 15360)])
def test_production_policy_reaches_actual_native_full_lookup_ceiling(prompt, mtp, budgets):
    row = run_case(prompt_tokens=prompt, mtp=mtp, arm="oscar_fixed",
                   remaining_budgets=budgets)
    assert row["production_policy_executed"] and not row["hypothetical"]
    assert row["status"] == "completed"
    assert row["completed_prompt_tokens"] == prompt
    assert row["symbolic_state_identity_violations"] == []
    assert sum(chunk["scheduled"] for chunk in row["chunks"]) == prompt
    assert row["repeat_prompt_hit_tokens"] == row["full_only_native_lookup_ceiling_tokens"]
    terminal = row["production_terminal_checkpoint"]
    if terminal:
        assert terminal in row["cached_mamba_checkpoint_tokens"]
        assert terminal in row["joint_aligned_mamba_checkpoint_tokens"]
    for chunk in row["chunks"]:
        assert 0 < chunk["scheduled"] <= chunk["proposed"] <= chunk["remaining_budget"]
        if chunk["before"] < terminal:
            assert chunk["end"] <= terminal
            assert chunk["end"] % row["mamba_block_tokens"] == 0
    assert row["global_token_budget"] == 16384
    assert row["native_cache_config_block_size"] == row["splitter_view_block_size"] == 768


def test_fixed_small_budget_advances_without_skipping_next_joint_boundary():
    row = run_case(prompt_tokens=15000, mtp=False, arm="oscar_fixed",
                   remaining_budgets=(2048,))
    assert [chunk["end"] for chunk in row["chunks"][:4]] == [1536, 2304, 3840, 4608]
    assert row["repeat_prompt_hit_tokens"] == 13824


def test_fixed_insufficient_state_budget_can_resume_on_next_step():
    blocked = run_case(prompt_tokens=15000, mtp=False, arm="oscar_fixed",
                       remaining_budgets=(512,))
    assert blocked["status"] == "blocked_by_repeated_insufficient_chunk_budget"
    assert blocked["completed_prompt_tokens"] == 0
    resumed = run_case(prompt_tokens=15000, mtp=False, arm="oscar_fixed",
                       remaining_budgets=(512, 2048))
    assert resumed["status"] == "completed"
    assert resumed["chunks"][0]["scheduled"] == 0
    assert resumed["repeat_prompt_hit_tokens"] == 13824


def test_fixed_post_terminal_tail_preserves_native_state_hash_identity():
    row = run_case(prompt_tokens=32768, mtp=True, arm="oscar_fixed",
                   remaining_budgets=(16384, 16384, 2048, 768))
    assert [chunk["end"] for chunk in row["chunks"]] == [16128, 29952, 31488, 32256, 32768]
    assert [chunk["splitter_route"] for chunk in row["chunks"]] == [
        "production_policy", "production_policy", "native_tail", "native_tail", "native_tail"]
    assert row["symbolic_state_identity_violations"] == []
    assert row["repeat_prompt_hit_tokens"] == row["full_only_native_lookup_ceiling_tokens"] == 29952


def test_symbolic_tags_detect_the_reviewed_unsafe_tail_counterexample(monkeypatch):
    from oscar_ascend.integration import prefix_checkpoint

    repaired = prefix_checkpoint.split_prefix_checkpoint

    def unsafe_tail_passthrough(scheduler, request, proposed, *args):
        planned = repaired(scheduler, request, proposed, *args)
        return proposed if planned is None else planned

    monkeypatch.setattr(prefix_checkpoint, "split_prefix_checkpoint", unsafe_tail_passthrough)
    row = run_case(prompt_tokens=32768, mtp=True, arm="oscar_fixed",
                   remaining_budgets=(16384, 16384, 2048, 768))
    assert [chunk["end"] for chunk in row["chunks"]] == [16128, 29952, 32000, 32768]
    assert row["symbolic_state_identity_violations"]
    assert all((item["represented_state_end"], item["hashed_checkpoint_end"]) == (32000, 32256)
               for item in row["symbolic_state_identity_violations"])
