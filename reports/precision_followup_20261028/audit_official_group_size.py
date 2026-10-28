"""Verify saved official provenance and execute host launch-selection ASTs.

No imported official serving module, Triton/NPU launch, production modification,
or performance experiment. Kernel invocations are recorded, not executed.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "reports/official_semantics_audit_20261021/upstream"


def functions(path, names, namespace, methods=()):
    tree = ast.parse(path.read_text())
    chosen = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in methods:
        found = [n for cls in tree.body if isinstance(cls, ast.ClassDef)
                 for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method]
        assert len(found) == 1
        chosen.extend(found)
    assert len(chosen) == len(names) + len(methods)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *chosen], type_ignores=[])),
                 str(path), "exec"), namespace)


def main():
    import torch
    manifest = json.loads((UPSTREAM.parent / "upstream_manifest.json").read_text())
    selected = (
        "rotation/run/qwen3.8-27b.sh", "rotation/run/_common.sh", "rotation/eval_oscar_gpqa.sh",
        "python/sglang/srt/mem_cache/unified_kv_pool.py",
        "python/sglang/srt/mem_cache/kv_quant_kernels.py",
        "python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py",
        "python/sglang/QuantKernel/gpu_flush_int2.py",
        "python/sglang/srt/layers/attention/triton_ops/decode_attention.py",
    )
    sources = {}
    for relative in selected:
        record = manifest["files"][relative]
        actual = hashlib.sha256((UPSTREAM / relative).read_bytes()).hexdigest()
        assert actual == record["sha256"]
        sources[relative] = {"url": record["url"], "sha256": actual, "manifest_match": True}
    old = ROOT / "reports/official_alignment_20261018/official_qwen38_recipe.sh"
    original_manifest = json.loads((old.parent / "sources.json").read_text())
    expected = original_manifest["sources"]["qwen38-pinned.sh"]
    actual = hashlib.sha256(old.read_bytes()).hexdigest()
    assert actual == expected["sha256"] == sources[selected[0]]["sha256"]

    calls = []
    class RecordKernel:
        def __init__(self, name):
            self.name = name
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                calls.append({"name": self.name, "grid": list(grid), "constexprs": kwargs})
            return call

    namespace = {"torch": torch, "triton": NS(cdiv=lambda x, y: (x + y - 1) // y,
        next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
        "_pretransformed_int2_set_kv_clip_single_kernel": RecordKernel("single"),
        "_pretransformed_int2_set_kv_clip_grouped_kernel": RecordKernel("grouped")}
    functions(UPSTREAM / "python/sglang/srt/mem_cache/kv_quant_kernels.py",
              {"_get_num_scale_groups", "_is_power_of_two"}, namespace)
    functions(UPSTREAM / "python/sglang/srt/mem_cache/unified_kv_pool.py", set(), namespace,
              ("_resolve_quant_grouping",))
    functions(UPSTREAM / "python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py", {
        "_can_use_grouped_clip_kernel", "_clip_idx_arg", "_clip_index", "_vectorized_elems_per_thread",
        "_pick_block_tok_and_num_warps", "_launch_single_clip_int2", "_launch_grouped_clip_int2",
        "quantized_set_kv_int2_pretransformed_clip_triton"}, namespace)
    functions(UPSTREAM / "python/sglang/QuantKernel/gpu_flush_int2.py",
              {"_resolve_kv_quant_config"}, namespace)
    cases = []
    for requested in (256, 128):
        resolved, groups = namespace["_resolve_quant_grouping"](NS(kv_cache_quant_group_size=requested), 256, "K")
        values = torch.zeros((4, 1, 256), dtype=torch.bfloat16)
        packed = torch.zeros((4, 1, 64), dtype=torch.uint8)
        sz = torch.zeros((4, 1, groups * 2), dtype=torch.float32)
        calls.clear()
        namespace["quantized_set_kv_int2_pretransformed_clip_triton"](
            values, values, torch.arange(4), packed, packed, sz, sz, .96, .92, lloyd_max=False)
        assert len(calls) == 2
        if requested == 128:
            assert all(c["name"] == "grouped" and c["constexprs"]["NUM_GROUPS"] == 2
                       and c["constexprs"]["GROUP_SIZE"] == 128 for c in calls)
        else:
            assert all(c["name"] == "single" and c["constexprs"]["LLOYD_MAX"] is False for c in calls)
        cases.append({"head_dim": 256, "requested_group_size": requested, "resolved": [resolved, groups],
                      "K_and_V_launches": list(calls),
                      "decode_flush_config": list(namespace["_resolve_kv_quant_config"](256, groups))})
    result = {
        "scope": "saved official source SHA verification and unchanged host AST dispatch; GPU kernels not executed",
        "official_commit": manifest["resolved_commit"], "sources": sources,
        "earlier_qwen38_recipe": {"commit": original_manifest["official_commit"], "url": expected["url"],
                                  "sha256": actual, "matches_later_recipe": True},
        "target_recipe": {"model": "Qwen3.8-27B", "GROUP_SIZE_default": 256, "LLOYD_MAX_default": 0},
        "source_correction": "Old generic/Qwen3-8B/Qwen3-32B scripts default128; they do not override target Qwen3.8 recipe256.",
        "dispatch_cases": cases,
    }
    destination = Path(__file__).with_name("official_group_size_audit.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verified_target_recipe": "GROUP_SIZE256 LLOYD_MAX0", "group128_supported_as_explicit_alternative": True,
                      "evidence": str(destination)}))


if __name__ == "__main__":
    main()
