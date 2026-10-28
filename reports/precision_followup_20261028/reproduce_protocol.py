"""Archive #37-49/#126/#145/#157/#170-173: reproduce protocol counterexamples.

Runs production/pinned-native CPU ownership contracts, actual test-only INT2
codec and attention mathematics, the current greedy adapter, and the pinned
Ascend pure-torch GDN golden. No native source is edited/imported wholesale;
no NPU, model weights, benchmark, timer or production forward is executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
GDN_GOLDEN = "references/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_recurrent_gated_delta_rule.py"


def native_cohort_replay():
    import pytest
    from oscar_ascend import plugin
    from oscar_ascend.integration.prefix_checkpoint import split_prefix_checkpoint
    from test_prefix_checkpoint_retention import _setup
    from test_runtime_native import definitions

    with pytest.MonkeyPatch.context() as mp:
        cache, scheduler, native_split, interface, _ = _setup(mp, 160, "lease")
        adopt = definitions("references/vllm/vllm/v1/core/kv_cache_coordinator.py",
            "cohort_native_adopt", set(), {}, mp,
            methods=[("KVCacheCoordinator", "allocate_new_computed_blocks")]).allocate_new_computed_blocks
        mp.setattr(type(cache), "allocate_new_computed_blocks", adopt, raising=False)
        owners, trace = {}, []

        def request(name, prompt, shared_hashes=0):
            # Synthetic chain identities at the native hash granularity;
            # native group composition/hash lookup are unchanged. No model
            # tokenizer or hash collision is needed for duplicate prefixes.
            hashes = [(i if i < shared_hashes else (100000 if name == "X" else 0) + i).to_bytes(8, "big")
                      for i in range(prompt // cache.hash_block_size)]
            return NS(request_id=name, num_prompt_tokens=prompt, num_tokens=prompt,
                      num_computed_tokens=0, has_encoder_inputs=False, block_hashes=hashes)

        def advance(req, count):
            before, end = req.num_computed_tokens, req.num_computed_tokens + count
            cache.remove_skipped_blocks(req.request_id, before)
            cache.allocate_new_blocks(req.request_id, end + 3, end, 0)
            for manager in cache.single_type_managers:
                blocks = manager.req_to_blocks[req.request_id]
                if isinstance(manager.kv_cache_spec, interface.MambaSpec):
                    block = blocks[(end - 1) // manager.block_size]
                    owners[manager.kv_cache_group_id, block.block_id] = (req.request_id, end)
                else:
                    for position in range(before, end):
                        block = blocks[position // manager.block_size]
                        owners[manager.kv_cache_group_id, block.block_id] = (req.request_id, position + 1)
            cache.cache_blocks(req, end)
            req.num_computed_tokens = end
            trace.append({"request": req.request_id, "before": before, "scheduled": count, "end": end})

        def finish(req, remaining):
            while req.num_computed_tokens < req.num_prompt_tokens:
                cache.new_step_starts()
                proposed = min(remaining, req.num_prompt_tokens - req.num_computed_tokens)
                count = split_prefix_checkpoint(scheduler, req, proposed)
                if count is None:
                    count = native_split(scheduler, req, proposed)
                assert count > 0
                advance(req, count)

        def lookup(req):
            blocks, length = cache.find_longest_cache_hit(req.block_hashes, req.num_tokens - 1)
            result = {"hit_tokens": length, "groups": []}
            for group, row in enumerate(blocks):
                result["groups"].append({
                    "group": group,
                    "live_blocks": [{"block_id": block.block_id,
                                     "symbolic_writer": owners.get((group, block.block_id)),
                                     "ref_count": block.ref_cnt}
                                    for block in row if not block.is_null],
                })
            return blocks, result

        try:
            a = request("A", 5000)
            finish(a, 16384)
            _, a_lookup = lookup(a)
            assert a_lookup["hit_tokens"] == 2304
            full_a = cache.single_type_managers[0].req_to_blocks["A"][0]

            # X adopts the real joint A checkpoint while it still exists.
            # Its longer prompt has a different terminal4608, so its lease
            # later holds that state, not the inherited A2304 state.
            x = request("X", 8000, shared_hashes=5000 // cache.hash_block_size)
            computed, x_lookup = lookup(x)
            assert x_lookup["hit_tokens"] == 2304
            cache.new_step_starts()
            cache.allocate_new_computed_blocks("X", computed, 2304, 0)
            x.num_computed_tokens = 2304
            cache.free("A")
            finish(x, 16384)
            assert cache.single_type_managers[0].req_to_blocks["X"][0] is full_a
            assert full_a.ref_cnt > 0
            lease_before = {key: [b.block_id for b in blocks] for key, blocks in
                            cache._oscar_prefix_checkpoint_leases.blocks_by_request.items()}

            # Allocate only the real free queue. Referenced X/FULL_A blocks
            # cannot enter this pressure allocation. This is a block-pool
            # contract exercise, not a replay of a complete serving schedule.
            pressure = cache.block_pool.get_new_blocks(cache.block_pool.get_num_free_blocks())
            assert full_a not in pressure
            pressure_ids = [b.block_id for b in pressure]
            cache.block_pool.free_blocks(pressure)
            b = request("B", 5000)
            assert a.block_hashes == b.block_hashes
            _, before_b = lookup(b)
            assert before_b["hit_tokens"] == 0
            finish(b, 768)
            full_b = cache.single_type_managers[0].req_to_blocks["B"][0]
            _, after_b = lookup(b)
            assert after_b["hit_tokens"] == 2304 and full_a is not full_b
            assert after_b["groups"][0]["live_blocks"][0]["symbolic_writer"][0] == "A"
            assert all(group["live_blocks"][-1]["symbolic_writer"][0] == "B"
                       for group in after_b["groups"][1:])
            leases = cache._oscar_prefix_checkpoint_leases.blocks_by_request
            assert all(len(blocks) <= 3 for blocks in leases.values())
            return {
                "backend": "actual production leases/splitter and pinned native manager/coordinator/pool/hash contracts",
                "global_budget": 16384, "pool_blocks": 160, "hash_block_tokens": cache.hash_block_size,
                "A_initial_joint_lookup": a_lookup, "X_initial_joint_lookup": x_lookup,
                "X_keeps_FULL_A": full_a.block_id, "X_leases_after_advance": lease_before,
                "pressure_allocated_only_free_blocks": pressure_ids,
                "before_B_joint_lookup": before_b, "after_B_joint_lookup": after_b,
                "B_owned_FULL_first_block": full_b.block_id,
                "max_extra_GDN_lease_pages_per_request": max(map(len, leases.values())),
                "trace": trace,
                "block_owners_are_symbolic_execution_tags": True,
                "mixed_execution_origins_proven": True,
            }
        finally:
            plugin.unregister()


def numerical_evidence():
    import torch
    from tools.official_reference import pack_kv, unpack_kv, attention_tiles, merge_and_inverse
    from oscar_ascend.integration.greedy_sampling import rejection_greedy_sample_with_triton

    torch.set_num_threads(1)
    dim = 256
    rotation = torch.eye(dim, dtype=torch.bfloat16)

    def inputs(last, selected):
        key = torch.zeros((last + 1, 1, dim), dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        query = torch.zeros((1, 6, dim), dtype=torch.bfloat16)
        key[selected, 0, :13] = 4
        query[:, :, :13] = 4
        value[selected, 0, :128] = .2
        value[selected, 0, 128:192] = -1
        value[selected, 0, 192:] = 1
        return query, key, value

    def evaluate(query, key, value, context, positions, quantized=True):
        cut = max(min(64, context), context + 1 - 256)
        indices = torch.cat((torch.arange(min(64, context)), torch.arange(cut, context)))
        history_key, history_value = unpack_kv(pack_kv(key[:context], value[:context], .96, .92), dim, 0)
        if not quantized:
            history_key, history_value = key, value
        queries = query.expand(len(positions), -1, -1)
        last = int(positions[-1])
        causal = torch.arange(context, last + 1)[None] <= positions[:, None]
        return merge_and_inverse([
            attention_tiles(queries, history_key[min(64, context):cut], history_value[min(64, context):cut]),
            attention_tiles(queries, key[indices], value[indices]),
            attention_tiles(queries, key[context:last + 1], value[context:last + 1], visible=causal),
        ], rotation).output

    # Cohort payload distinguishability, at a row that remains in the native
    # FULL2304 page's last259-row exact snapshot.
    q, k, v = inputs(2200, 64)
    a = merge_and_inverse([attention_tiles(q, k, v)], rotation).output
    b = evaluate(q, k, v, 1536, torch.tensor([2200]))
    ablation = evaluate(q, k, v, 1536, torch.tensor([2200]), False)
    assert torch.equal(a, ablation)

    def projection(hidden):
        raw = hidden[:, :1].float()
        result = raw.clone()
        result[:, :, :128] += .4 * raw[:, :, -1:]
        return result.bfloat16()

    pa, pb = projection(a), projection(b)
    packed_a, packed_b = pack_kv(pa, pa, .96, .92), pack_kv(pb, pb, .96, .92)
    assert int((packed_a != packed_b).sum()) == 64

    # Execute the actual pure-torch golden function from the allowed native
    # Ascend test without importing the test's torch_npu top-level effects.
    tree = ast.parse((ROOT / GDN_GOLDEN).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "golden_recurrent_gated_delta_rule")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), GDN_GOLDEN, "exec"), namespace)
    golden = namespace[node.name]
    length, head_dim = 104, 128
    gq = torch.zeros(length, 1, head_dim, dtype=torch.bfloat16)
    gk = gq.clone()
    gq[:, :, 0] = 1
    gk[0, :, 0] = 1
    gk[1:, :, 1] = 1
    initial = torch.zeros(1, 1, head_dim, head_dim, dtype=torch.bfloat16)
    beta = torch.full((length, 1), .5, dtype=torch.bfloat16)
    gate = torch.full((length, 1), -.001, dtype=torch.bfloat16)
    state_indices = torch.zeros(length, dtype=torch.int64)
    states = []
    for hidden in (a, b):
        gv = torch.zeros(length, 1, head_dim, dtype=torch.bfloat16)
        gv[0] = hidden[0, 0, :head_dim]
        _, state = golden(gq, gk, gv, initial, beta, 1., [length], state_indices, gate, None)
        states.append(state)
    assert int((states[0] != states[1]).sum()) == 128

    # MTP3 verification: same logical query and incoming tensors. The third
    # accepted draft is checked by target row C+2, not the bonus row C+3.
    context = 768
    mq, mk, mv = inputs(context + 3, 514)
    positions = torch.arange(context, context + 4)
    verify = evaluate(mq, mk, mv, context, positions)
    serial = torch.cat([evaluate(mq, mk, mv, int(p), p[None]) for p in positions])
    verify_exact = evaluate(mq, mk, mv, context, positions, False)
    serial_exact = torch.cat([evaluate(mq, mk, mv, int(p), p[None], False) for p in positions])

    def argmax(hidden):
        x = hidden[:, 0, 0].float()
        return torch.stack((.25 - x, x - .25), dim=-1).argmax(-1)

    verify_ids, serial_ids = argmax(verify), argmax(serial)
    output = torch.full((1, 4), -1, dtype=torch.int64)
    rejection_greedy_sample_with_triton(output, [3], torch.tensor([3]), torch.tensor([0, 0, 0]),
        verify_ids[:3], verify_ids[3:], None, 3, None, None)
    assert verify_ids[:3].tolist() == [0, 0, 0]
    assert serial_ids[:3].tolist() == [0, 0, 1]
    assert argmax(verify_exact).tolist() == argmax(serial_exact).tolist() == [0, 0, 0, 0]
    return {
        "synthetic_inputs": "D256/Hq6/Hkv1; identity rotations, zero center; K/Q first13 dims4; selected V first128 dimsBF16(.2), next64=-1,last64=1; all others0",
        "codec": "actual test-only144B pack/unpack, Kclip.96,Vclip.92,mode0",
        "cohort_payload": {
            "query_position": 2200, "checkpoint": 2304,
            "position_in_last259_snapshot_rows": True,
            "A_hidden_element": float(a[0, 0, 0]), "B_hidden_element": float(b[0, 0, 0]),
            "all_BF16_split_ablation_bitwise_equal": True,
            "same_linear_projection": "y[:128]=h[:128]+.4*h[-1]; other dimensions unchanged",
            "A_projected_element": float(pa[0, 0, 0]), "B_projected_element": float(pb[0, 0, 0]),
            "FULL_INT2_slot_different_bytes": int((packed_a != packed_b).sum()),
            "FULL_snapshot_different_elements": int((pa != pb).sum()),
            "GDN_reference": GDN_GOLDEN,
            "GDN_reference_execution": "same BF16 initial state,104-token suffix2200..2303, first V from hidden; remaining V0; beta=.5,g=-.001; K switches e0 to e1",
            "GDN_A_final_element": float(states[0][0, 0, 0, 0]),
            "GDN_B_final_element": float(states[1][0, 0, 0, 0]),
            "GDN_final_different_elements": int((states[0] != states[1]).sum()),
        },
        "MTP": {
            "context": context, "selected_cached_token": 514, "third_verify_query_position": 770,
            "verify_fixed_cut": 513, "serial_third_query_cut": 515,
            "fixed_two_class_head": "logits=[.25-x,x-.25],x=output[head0,dim0]",
            "verify_target_argmax": verify_ids.tolist(), "serial_q1_target_argmax": serial_ids.tolist(),
            "third_verify_output": float(verify[2, 0, 0]),
            "third_serial_output": float(serial[2, 0, 0]),
            "actual_greedy_adapter_output": output.tolist(),
            "all_BF16_verify_argmax": argmax(verify_exact).tolist(),
            "all_BF16_serial_argmax": argmax(serial_exact).tolist(),
            "third_draft_accepted_despite_serial_target_disagreement": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("protocol_evidence.json"))
    args = parser.parse_args()
    numerical = numerical_evidence()
    native = native_cohort_replay()
    sources = (
        "oscar_ascend/integration/prefix_checkpoint.py", "oscar_ascend/plugin.py",
        "oscar_ascend/integration/greedy_sampling.py", "tools/official_reference.py",
        "csrc/kernels/attention_cv_official_decode.cpp", "csrc/kernels/attention_official_tasks.cpp",
        "references/vllm/vllm/v1/core/block_pool.py", "references/vllm/vllm/v1/core/kv_cache_coordinator.py",
        "references/vllm/vllm/v1/core/single_type_kv_cache_manager.py",
        "references/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py",
        "references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_manager.py",
        "tests/test_prefix_checkpoint_retention.py", "tests/test_runtime_native.py", GDN_GOLDEN,
        "reports/precision_followup_20261028/reproduce_protocol.py",
    )
    report = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_state": "working-tree hashes bind actual inputs independently of HEAD",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "native_cohort_replay": native, "numerical_evidence": numerical,
        "limits": [
            "Synthetic hash identities, model inputs and a fixed synthetic projection/head; no real LongBench examples.",
            "Native ownership/cache methods execute unchanged; block-pool pressure is not a full scheduler/service trace.",
            "Execution-owner tags are symbolic; separate numerical examples demonstrate distinguishable valid FULL/GDN payloads.",
            "GDN uses the pinned Ascend pure-torch reference, not actual device GDN or complete Qwen weights.",
            "The numerical examples are compositional counterexamples, not a measured share of accuracy loss or runtime variance.",
            "No uncommitted-token hash publication or physical-page write race is claimed by these examples.",
        ],
        "NPU": "not_run", "graph": "not_run", "model_quality": "not_run", "performance": "not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output),
        "joint_lookup": native["after_B_joint_lookup"], "MTP": numerical["MTP"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
