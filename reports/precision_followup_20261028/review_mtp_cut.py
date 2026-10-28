"""Archive #37-49/#126/#145/#157/#170-173: reject unsafe MTP cut changes.

Independent CPU mathematics and ownership arithmetic only. Production remains
unchanged. The codec is the current independent v4 reference; no NPU/model run.
"""
from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def numerical_review():
    import torch
    from tools.official_reference import pack_kv, unpack_kv, attention_tiles, merge_and_inverse

    torch.set_num_threads(1)
    context, position, selected = 768, 770, 514
    q = torch.zeros((1, 6, 256), dtype=torch.bfloat16)
    k = torch.zeros((position + 1, 1, 256), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    q[:, :, :13] = 4
    k[selected, :, :13] = 4
    v[selected, :, :128] = .2
    v[selected, :, 128:192] = -1
    v[selected, :, 192:] = 1
    qk, qv = unpack_kv(pack_kv(k, v, .96, .92), 256, 0)
    rotation = torch.eye(256, dtype=torch.bfloat16)

    def evaluate(cut):
        exact_ids = torch.cat((torch.arange(64), torch.arange(cut, context)))
        return merge_and_inverse([
            attention_tiles(q, qk[64:cut], qv[64:cut]),
            attention_tiles(q, k[exact_ids], v[exact_ids]),
            attention_tiles(q, k[context:], v[context:]),
        ], rotation).output

    old = evaluate(context + 1 - 256)
    query_relative = evaluate(position + 1 - 256)
    native = merge_and_inverse([attention_tiles(q, k, v)], rotation).output

    def label(result):
        x = result[0, 0, 0].float()
        return int(torch.stack((.25 - x, x - .25)).argmax())

    assert label(old) == label(native) == 0
    assert label(query_relative) == 1
    assert torch.equal(old, native)
    return {
        "context": context, "logical_query": position, "selected_old_token": selected,
        "old_q4_cut": context + 1 - 256,
        "query_relative_cut": position + 1 - 256,
        "old_q4": {"output_dim0": float(old[0, 0, 0]), "argmax": label(old)},
        "query_relative": {"output_dim0": float(query_relative[0, 0, 0]), "argmax": label(query_relative)},
        "all_bf16": {"output_dim0": float(native[0, 0, 0]), "argmax": label(native)},
        "old_q4_equals_all_bf16": True,
        "uniform_mask_fix_can_worsen_attention_and_argmax": True,
    }


def rollback_review():
    # Scalar model of the actual physical-page writer: sink plus inPage%ring.
    # A q4 forward writes [768,771], then only the known token is committed.
    block, sink, recent, spec, context = 2304, 64, 256, 3, 768
    ring = recent + spec
    tags = {}
    for logical in range(context + spec + 1):
        page, in_page = divmod(logical, block)
        tags[page, sink + in_page % ring] = in_page
        if logical < sink:
            tags[page, logical] = in_page
    next_context = context + 1

    def invalid(cut):
        bad = []
        for logical in range(cut, next_context):
            page, in_page = divmod(logical, block)
            stored = tags[page, sink + in_page % ring]
            if stored != in_page:
                bad.append({"logical": logical, "observed_tag": stored})
        return bad

    old_cut = next_context + 1 - recent
    expanded_cut = next_context + 1 - (recent + spec)
    old_invalid, expanded_invalid = invalid(old_cut), invalid(expanded_cut)
    assert old_invalid == []
    assert [entry["logical"] for entry in expanded_invalid] == [511, 512]
    return {
        "writer_ring_rows": ring, "written_positions": [context, context + spec],
        "accepted_drafts": 0, "next_context": next_context,
        "old_cut": old_cut, "old_invalid_tags": old_invalid,
        "expanded_cut": expanded_cut, "expanded_invalid_tags": expanded_invalid,
        "extra_spec_rows_are_rollback_reserve": True,
    }


def route_review():
    import torch
    from oscar_ascend.integration.mixed_decode_split import plan_mixed_decode
    from oscar_ascend.ops.cv_dispatch import select_cv_op, OFFICIAL_CV_MAP
    from tools.serving_variants import variant_config

    config = variant_config(json.loads((ROOT / "configs/target.json").read_text()), "candidate")
    names = ("fast_unpack", "mixed_cv", "striped_cache", "decode_bundle", "c4_member_pipeline",
             "mdl_decode", "c16_member_pipeline", "hybrid_l1_decode", "c16_l1_reuse", "bf16_attention")
    flags = {name: config.get("experimental_" + name, False) for name in names}
    rows = []
    for lengths in ([4, 768], [768, 4], [4, 8192], [8192, 4], [4] * 33 + [8192]):
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        metadata = NS(mixed_decode_split=True, capture_origin=False, dummy_origin=False,
                      is_draft=False, query_start_loc_cpu=torch.tensor(starts),
                      num_reqs=len(lengths), num_actual_tokens=starts[-1], cv_shape_tokens=None)
        plan = plan_mixed_decode(metadata, starts[-1])
        parent_op = OFFICIAL_CV_MAP[select_cv_op(16, 6, 1, starts[-1], max(lengths), **flags)]
        q4_op = (OFFICIAL_CV_MAP[select_cv_op(16, 6, 1, plan.prefix_tokens, 4, **flags)]
                 if plan else parent_op)
        rows.append({"lengths": lengths, "partitioned": plan is not None,
                     "parent_operator": parent_op, "q4_operator": q4_op})
    assert rows[1]["q4_operator"] == "attention_cv_official_cluster4_out"
    assert rows[3]["q4_operator"] == rows[4]["q4_operator"] == "attention_cv_official_cluster16_out"
    return rows


def native_commit_review():
    # Execute the pinned allocate_slots method unchanged. Dependencies only
    # record allocation/cache endpoints; they cannot alter its cap arithmetic.
    path = ROOT / "references/vllm/vllm/v1/core/kv_cache_manager.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KVCacheManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "allocate_slots")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])),
                 str(path), "exec"), namespace)
    endpoints = []
    rows = []
    coordinator = NS(
        remove_skipped_blocks=lambda *a: None,
        get_num_blocks_to_allocate=lambda **kw: 1,
        allocate_new_blocks=lambda *a: (),
        cache_blocks=lambda req, end: endpoints.append(end),
    )
    manager = NS(empty_kv_cache_blocks=NS(blocks=()), max_model_len=262144, coordinator=coordinator,
                 block_pool=NS(get_num_free_blocks=lambda: 100), enable_caching=True,
                 create_kv_cache_blocks=lambda blocks: blocks)
    for context in (767, 768, 2301, 2302, 2303, 2304, 4607):
        req = NS(request_id="x", num_computed_tokens=context, num_tokens=context + 1)
        namespace["allocate_slots"](manager, req, 4, num_lookahead_tokens=3)
        assert endpoints[-1] == context + 1
        rows.append({"context": context, "scheduled_end": context + 4,
                     "finalized_end": context + 1, "actual_native_cache_end": endpoints[-1],
                     "FULL2304_hashed_pages": endpoints[-1] // 2304})
    return {"uncommitted_drafts_excluded_in_all_cases": True, "cases": rows}


def main():
    paths = (
        "csrc/kernels/attention_cv_official_decode.cpp",
        "csrc/kernels/attention_cv_official_cluster4.cpp",
        "csrc/kernels/attention_cv_official_cluster16.cpp",
        "csrc/kernels/attention_official_tasks.cpp",
        "csrc/kernels/store_aligned_int2.cpp",
        "oscar_ascend/lifecycle.py",
        "oscar_ascend/integration/mixed_decode_split.py",
        "oscar_ascend/ops/cv_dispatch.py",
        "tools/official_reference.py",
        "references/vllm/vllm/v1/core/kv_cache_manager.py",
        "reports/precision_followup_20261028/review_mtp_cut.py",
    )
    result = {"scope": "synthetic CPU/reference and current candidate route audit; no production edit",
              "numerical_review": numerical_review(), "rollback_review": rollback_review(),
              "route_review": route_review(), "native_commit_review": native_commit_review(),
              "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}}
    destination = Path(__file__).with_name("mtp_cut_review.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "rejected_as_default_quality_fix", "evidence": str(destination)}))


if __name__ == "__main__":
    main()
