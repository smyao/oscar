"""Local prefix-checkpoint audit; archive #26/#37-49/#127/#128/#157.

Execute pinned native scheduler splitting, managers, block pool and hybrid
lookup through the existing AST fixture harness. Archive #127/#128 requires
separate native Mamba, physical FULL and metadata128 sizes; #26/#37-49/#157
forbids treating metadata reuse as proof of correct device/window execution.

Research only: no production imports this module. No NPU, model, service,
timing, decode acceptance, tensor checkpoint contents or real traffic is
simulated. Each scheduled prefill chunk is assumed to finish successfully.
Synthetic identity hashes are supplied at native hash granularity. The
hypothetical arm changes only the isolated splitter's config view to the
joint block size; it is neither a deployed fix nor a GDN-state-size change.
The fixed arm calls the real production boundary policy, then executes the
same native manager/pool lifecycle. It does not stand in for an engine run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("native", "oscar_current", "oscar_joint_alignment_hypothesis", "oscar_fixed")
SOURCE_PATHS = (
    "references/vllm/vllm/v1/core/sched/scheduler.py",
    "references/vllm/vllm/v1/core/single_type_kv_cache_manager.py",
    "references/vllm/vllm/v1/core/block_pool.py",
    "references/vllm/vllm/v1/core/kv_cache_utils.py",
    "references/vllm/vllm/v1/kv_cache_interface.py",
    "references/vllm/vllm/v1/kv_cache_spec_registry.py",
    "references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_manager.py",
    "references/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py",
    "references/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py",
    "oscar_ascend/integration/specs.py",
    "oscar_ascend/integration/prefix_checkpoint.py",
    "oscar_ascend/plugin.py",
    "oscar_ascend/layout.py",
    "tests/test_runtime_native.py",
    "tests/test_prefix_caching.py",
    "tests/test_prefix_checkpoint_audit.py",
    "tools/experiments/prefix_checkpoint_audit.py",
)


def source_provenance() -> dict:
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in SOURCE_PATHS
        },
        "source_state": "working tree; file hashes bind uncommitted source as well",
    }


def _imports():
    # Deliberately reuse the research harness, never install fake vLLM into a
    # service. MonkeyPatch restores every substituted module after each case.
    tests_path = str(ROOT / "tests")
    inserted = tests_path not in sys.path
    if inserted:
        sys.path.insert(0, tests_path)
    try:
        import pytest
        from test_runtime_native import definitions, native_groups, native_specs
        from test_prefix_caching import _coordinator_with_hashing
    finally:
        if inserted:
            sys.path.remove(tests_path)
    return pytest, definitions, native_groups, native_specs, _coordinator_with_hashing


def run_case(*, prompt_tokens: int, mtp: bool, arm: str,
             remaining_budgets: tuple[int, ...] = (16384,)) -> dict:
    """Repeat the supplied remaining-budget cycle after other requests run.

    The configured global token budget remains 16384 in every arm. Other
    requests' computations are not modeled; their already-consumed budget
    is explicit. A zero split is recorded as blocked rather than hidden.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if prompt_tokens <= 0 or not remaining_budgets:
        raise ValueError("positive prompt and nonempty remaining-budget cycle required")
    if any(type(value) is not int or not 0 < value <= 16384 for value in remaining_budgets):
        raise ValueError("remaining budgets must be integers within the frozen 16384 budget")
    pytest, definitions, native_groups, native_specs, coordinator = _imports()
    with pytest.MonkeyPatch.context() as mp:
        native = native_specs.__wrapped__(mp)
        interface, ours, registry = native
        managers_module = sys.modules["vllm.v1.core.single_type_kv_cache_manager"]
        registry.KVCacheSpecRegistry.register(
            interface.FullAttentionSpec, managers_module.FullAttentionManager,
            interface.FullAttentionSpec)
        convert = ours.transform_native_specs
        speculative = 3 if mtp else 0
        page_bytes = 817152 if mtp else 801792

        def configured_specs(source):
            configured = {}
            for name, spec in source.items():
                if not mtp and name == "full.16":
                    continue  # no MTP draft FULL layer in the paired control
                if isinstance(spec, interface.MambaSpec):
                    spec = replace(spec, shapes=((6 if mtp else 3, 2560), (12, 128, 128)),
                                   num_speculative_blocks=speculative,
                                   page_size_padded=page_bytes)
                else:
                    spec = replace(spec, page_size_padded=page_bytes)
                configured[name] = spec
            if arm == "native":
                return configured
            return convert(configured, metadata_bytes=4)

        mp.setattr(ours, "transform_native_specs", configured_specs)
        grouped = native_groups.__wrapped__(native, mp, SimpleNamespace(param=True))
        original = coordinator(grouped, mp)
        # The shared fixture creates MTP=True. Instantiate the identical native
        # class again with the actual arm flag, including its group annotation.
        cache = type(original)(
            original.kv_cache_config, max_model_len=262144, use_eagle=mtp,
            enable_caching=True, enable_kv_cache_events=False,
            dcp_world_size=1, pcp_world_size=1,
            hash_block_size=original.hash_block_size,
            max_num_batched_tokens=16384,
            scheduler_block_size=original.scheduler_block_size)
        splitter = definitions(
            SOURCE_PATHS[0], "prefix_audit_native_split", set(), {}, mp,
            methods=[("Scheduler", "_mamba_block_aligned_split")]
        )._mamba_block_aligned_split
        joint = cache.lcm_block_size
        split_block = joint if arm.endswith("hypothesis") else 768
        scheduler = SimpleNamespace(
            block_size=joint, cache_config=SimpleNamespace(
                block_size=split_block, enable_prefix_caching=True, mamba_cache_mode="align"),
            use_eagle=mtp, kv_cache_manager=SimpleNamespace(coordinator=cache),
            kv_cache_config=cache.kv_cache_config, max_num_scheduled_tokens=16384,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16384))
        fixed_policy = None
        if arm == "oscar_fixed":
            from oscar_ascend.integration.prefix_checkpoint import (
                policy_for_scheduler, split_prefix_checkpoint)
            fixed_policy = policy_for_scheduler(scheduler)
            assert fixed_policy is not None, "fixed arm did not bind OSCAR production policy"
            scheduler._oscar_prefix_checkpoint_policy = fixed_policy
        request = SimpleNamespace(
            request_id="synthetic-seed", num_prompt_tokens=prompt_tokens,
            num_tokens=prompt_tokens, num_computed_tokens=0,
            block_hashes=[i.to_bytes(8, "big")
                          for i in range(prompt_tokens // cache.hash_block_size)])
        chunks = []
        consecutive_zero = 0
        status = "completed"
        state_end_by_block = {}
        state_identity_violations = []
        while request.num_computed_tokens < prompt_tokens:
            remaining = remaining_budgets[len(chunks) % len(remaining_budgets)]
            before = request.num_computed_tokens
            proposed = min(remaining, prompt_tokens - before)
            planned = (split_prefix_checkpoint(scheduler, request, proposed)
                       if fixed_policy is not None else None)
            # Mirror the actual plugin wrapper: None retains the unchanged
            # native splitter, especially for post-terminal partial tails.
            scheduled = (splitter(scheduler, request, proposed)
                         if planned is None else planned)
            route = ("production_policy" if planned is not None else
                     "native_tail" if fixed_policy is not None else "pinned_native")
            end = before + scheduled
            chunks.append({"before": before, "remaining_budget": remaining,
                           "proposed": proposed, "scheduled": scheduled, "end": end,
                           "joint_aligned": end % joint == 0, "splitter_route": route})
            if scheduled == 0:
                consecutive_zero += 1
                if consecutive_zero == len(remaining_budgets):
                    status = "blocked_by_repeated_insufficient_chunk_budget"
                    break
                continue
            consecutive_zero = 0
            for manager in cache.single_type_managers:
                manager.new_step_starts()
                manager.remove_skipped_blocks(request.request_id, before)
                manager.allocate_new_blocks(request.request_id, end + speculative, end)
                is_mamba = isinstance(manager.kv_cache_spec, interface.MambaSpec)
                if is_mamba:
                    # A symbolic position tag, not a simulated GDN tensor:
                    # native forward stores the running state at this slot.
                    index = (end - 1) // manager.block_size
                    block = manager.req_to_blocks[request.request_id][index]
                    assert not block.is_null
                    state_end_by_block[manager.kv_cache_group_id, block.block_id] = end
                manager.cache_blocks(request, end)
                if is_mamba:
                    for index, block in enumerate(manager.req_to_blocks[request.request_id]):
                        if block.is_null or block.block_hash is None:
                            continue
                        represented = state_end_by_block[manager.kv_cache_group_id, block.block_id]
                        hashed_end = (index + 1) * manager.block_size
                        if represented != hashed_end:
                            state_identity_violations.append({
                                "step": len(chunks) - 1, "group": manager.kv_cache_group_id,
                                "physical_block": block.block_id,
                                "hashed_checkpoint_end": hashed_end,
                                "represented_state_end": represented})
            request.num_computed_tokens = end

        mamba_group_ids = [manager.kv_cache_group_id for manager in cache.single_type_managers
                           if isinstance(manager.kv_cache_spec, interface.MambaSpec)]
        # Read actual native pool hashes, not inferred chunk endpoints: null
        # intermediate states were skipped by native cache_full_blocks().
        checkpoints = [
            (index + 1) * cache.hash_block_size
            for index, block_hash in enumerate(request.block_hashes)
            if cache.block_pool.get_cached_block(block_hash, mamba_group_ids)
        ]
        for manager in cache.single_type_managers:
            manager.free(request.request_id)
        # Independent lookup ceiling: execute the unchanged native FULL
        # finder directly, without the Mamba minimum. This includes prompt-1
        # and the actual FULL-group EAGLE drop, not the new policy formula.
        full = cache.single_type_managers[0]
        full_hashes = request.block_hashes
        if full.block_size != cache.hash_block_size:
            full_hashes = grouped[3].BlockHashListWithBlockSize(
                full_hashes, cache.hash_block_size, full.block_size)
        full_hit = type(full).find_longest_cache_hit(
            block_hashes=full_hashes, max_length=prompt_tokens - 1,
            kv_cache_group_ids=[full.kv_cache_group_id], block_pool=cache.block_pool,
            kv_cache_spec=full.kv_cache_spec, drop_eagle_block=full.use_eagle,
            alignment_tokens=joint, dcp_world_size=1, pcp_world_size=1)
        full_lookup_ceiling = len(full_hit[0]) * full.block_size
        _, hit = cache.find_longest_cache_hit(request.block_hashes, prompt_tokens - 1)
        return {
            "arm": arm, "hypothetical": arm.endswith("hypothesis"),
            "production_policy_executed": arm == "oscar_fixed",
            "prompt_tokens": prompt_tokens, "mtp": mtp,
            "global_token_budget": 16384, "remaining_budget_cycle": list(remaining_budgets),
            "native_cache_config_block_size": 768, "splitter_view_block_size": split_block,
            "full_block_tokens": cache.single_type_managers[0].block_size,
            "mamba_block_tokens": 768, "mamba_speculative_blocks": speculative,
            "hash_block_tokens": cache.hash_block_size, "joint_lookup_alignment": joint,
            "metadata_bytes": None if arm == "native" else 4,
            "native_final_cache_position": max(prompt_tokens // split_block * split_block
                                               - (split_block if mtp else 0), 0),
            "production_terminal_checkpoint": (fixed_policy.terminal_position(prompt_tokens)
                                               if fixed_policy is not None else None),
            "status": status, "chunks": chunks,
            "cached_mamba_checkpoint_tokens": checkpoints,
            "joint_aligned_mamba_checkpoint_tokens": [p for p in checkpoints if p % joint == 0],
            "max_lookup_tokens": prompt_tokens - 1,
            "full_only_native_lookup_ceiling_tokens": full_lookup_ceiling,
            "repeat_prompt_hit_tokens": hit,
            "repeat_prompt_hit_fraction": hit / prompt_tokens,
            "completed_prompt_tokens": request.num_computed_tokens,
            "symbolic_state_identity_violations": state_identity_violations,
        }


def audit() -> dict:
    rows = [run_case(prompt_tokens=length, mtp=mtp, arm=arm, remaining_budgets=budgets)
            for length in (4608, 6912, 9216, 15000, 16000, 32768, 65536)
            for mtp in (False, True)
            for budgets in ((16384,), (15360,), (16384, 15360), (2048,), (768,),
                            (2048, 768, 15360))
            for arm in ARMS]
    return {
        "schema_version": 2, "evidence": "local production splitter plus pinned native cache contract execution",
        "provenance": source_provenance(),
        "arm_descriptions": {
            "native": "Native FULL768/Mamba768 with the pinned native splitter.",
            "oscar_current": "Pre-fix OSCAR2304/Mamba768 with the pinned native splitter; retained regression control.",
            "oscar_joint_alignment_hypothesis": "Incomplete naive joint-alignment experiment, not the production repair.",
            "oscar_fixed": "Actual production prefix-checkpoint policy followed by unchanged native cache contracts.",
        },
        "limits": [
            "Synthetic repeated prompts and explicit remaining budgets, not lost user data.",
            "Executes the native splitter and cache contracts, not complete Scheduler.schedule or actual concurrent admission.",
            "No NPU, GDN arithmetic, attention/window correctness, graph, quality or performance result.",
            "Checkpoint payloads assumed valid after each successful prefill; decode is not executed.",
            "Symbolic state-position tags check native hash identity only; no GDN tensor arithmetic is executed.",
            "No eviction pressure; pool is deliberately large enough for each isolated prompt.",
            "The hypothetical arm changes only an isolated splitter view; the fixed arm calls the production policy. No launch argv is modified.",
            "Naive joint alignment is incomplete: exact-block prompts can still miss because lookup excludes the final token, then MTP drops a FULL block.",
            "Tiny repeated budgets can block the hypothetical larger alignment and are reported.",
        ],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(audit(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
