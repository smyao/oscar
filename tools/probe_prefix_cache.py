"""Archive #26/#37-49/#126/#127/#128/#142/#148/#157: real APC storage gate.

D.4: exercises the real bounded snapshot writer and fused INT2 reader, never a
serving full-history restore. This is a synthetic physical-page contract, not
a service/cache-hit-rate or performance benchmark. Native allocator admission
is covered separately by test_prefix_caching; no allocator semantics change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from oscar_ascend.layout import HybridPageLayout, SlotLayout
from oscar_ascend.lifecycle import SnapshotLayout
from oscar_ascend.ops.cv_dispatch import select_cv_op
from .phase import atomic_json
from . import probe_history_reuse as reuse

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "synthetic_page_owned_prefix_writer_reader_graph"
BLOCKS, HEADS, DIM, QUERIES = 7, 6, 256, 8
STATES = ("shared_prefix_forks", "partial_hit_changed_graph", "released_tail_reuse")
SOURCE_PATHS = (
    "oscar_ascend/lifecycle.py", "oscar_ascend/layout.py",
    "oscar_ascend/ops/cv_dispatch.py", "csrc/kernels/rotate_clip_store_striped.cpp",
    "csrc/kernels/rotate_vector_init.cpp", "csrc/kernels/attention_tasks.cpp",
    "csrc/kernels/attention_cv_mdl_decode.cpp", "csrc/kernels/merge_lse.cpp",
)


def source_binding(root=ROOT):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def fixture_layout():
    # Declared D256/TP4 align ABI fixture, NOT a new service allocation or a
    # claim that a model's runtime geometry was observed on this host.
    return SnapshotLayout(HybridPageLayout(SlotLayout(DIM), 30720, 393216,
                          768, native_page_size_bytes=817152), 64, 256, 3)


def selected_reader(config):
    if config.get("enable_prefix_caching") is not True:
        raise ValueError("prefix gate requires explicit prefix caching enabled")
    if config.get("experimental_request_windows", False):
        raise ValueError("prefix gate requires physical-page snapshots")
    if not config.get("experimental_striped_cache", False):
        raise ValueError("prefix gate requires the candidate striped writer/reader")
    return select_cv_op(16, HEADS, 1, QUERIES, 4,
        fast_unpack=config.get("experimental_fast_unpack", False),
        mixed_cv=config.get("experimental_mixed_cv", False), striped_cache=True,
        decode_bundle=config.get("experimental_decode_bundle", False),
        c4_member_pipeline=config.get("experimental_c4_member_pipeline", False),
        mdl_decode=config.get("experimental_mdl_decode", False))


def make_inputs(torch, logical, pages, seed):
    """Independent logical K/V and fixed-size native virtual128 metadata."""
    generator = torch.Generator().manual_seed(seed)
    contexts = tuple(k.shape[0] for k, _ in logical)
    if len(logical) != 2 or any(c < 320 or c + 4 > 3 * 2304 for c in contexts):
        raise ValueError("APC fixture needs two bounded positive-context forks")
    q = torch.randn((QUERIES, HEADS, DIM), generator=generator).to(torch.bfloat16)
    key = torch.randn((QUERIES, 1, DIM), generator=generator).to(torch.bfloat16)
    value = torch.randn((QUERIES, 1, DIM), generator=generator).to(torch.bfloat16)
    table = torch.full((2, 54), -1, dtype=torch.int32)
    slots = torch.empty(QUERIES, dtype=torch.int64)
    for r, (context, mapping) in enumerate(zip(contexts, pages)):
        if len(mapping) * 2304 < context + 4 or len(set(mapping)) != len(mapping):
            raise ValueError("invalid physical page chain")
        for p, physical in enumerate(mapping):
            table[r, p * 18:(p + 1) * 18] = torch.arange(physical * 18, (physical + 1) * 18)
        for i in range(4):
            position = context + i
            slots[r * 4 + i] = mapping[position // 2304] * 2304 + position % 2304
    return {"q": q, "key": key, "value": value, "table": table, "slots": slots,
            "starts": torch.tensor([0, 4, 8], dtype=torch.int32),
            "lens": torch.tensor([c + 4 for c in contexts], dtype=torch.int32)}


def oracle(torch, logical, inputs):
    from oscar_ascend.ops.reference import attention, encode_kv, decode_kv
    outputs, lses = [], []
    for r, (key, value) in enumerate(logical):
        context = key.shape[0]
        # Identity rotation isolates physical caching from rotation accuracy;
        # the MDL gate separately exercises the production Hadamard rotation.
        restored_k, restored_v = decode_kv(encode_kv(key.float(), value.float()), DIM)
        for i in range(4):
            cut = min(context, max(64, context + i + 1 - 256))
            k, v = key.float().clone(), value.float().clone()
            k[64:cut], v[64:cut] = restored_k[64:cut], restored_v[64:cut]
            index = r * 4 + i
            k = torch.cat((k, inputs["key"][r * 4:index + 1].float()))
            v = torch.cat((v, inputs["value"][r * 4:index + 1].float()))
            result = attention(inputs["q"][index:index + 1], k, v,
                               scale=DIM ** -0.5, causal=False)
            outputs.append(result.output[0]); lses.append(result.lse[0])
    return torch.stack(outputs), torch.stack(lses)


def page_bytes(torch, raw, layout, page):
    intervals = (layout.page.packed_interval(BLOCKS, page), layout.interval(BLOCKS, page))
    return torch.cat([raw[a:b] for a, b in intervals]).clone()


def _store(torch, ops, state, key, value, slots, positions):
    status = torch.empty(key.shape[0], dtype=torch.int32, device=key.device)
    layout = state["layout"]
    ops.rotate_clip_store_striped_out(key, value, state["rotation"], state["rotation"],
        slots, positions, state["raw"], state["wk"], state["wv"], state["tags"], status,
        layout.page.block_size, BLOCKS, BLOCKS * layout.page.conv_bytes,
        layout.page.ssm_bytes, layout.sink_tokens, layout.ring_tokens, 0., 0., False)
    torch.npu.synchronize()
    if bool(status.cpu().ne(0).any()):
        raise RuntimeError("APC real prefix writer reported nonzero status")


def seed_page_chain(torch, ops, state, logical, pages):
    key, value = logical
    positions = torch.arange(key.shape[0], dtype=torch.int64)
    mapping = torch.tensor(pages, dtype=torch.int64)
    slots = mapping[positions // 2304] * 2304 + positions % 2304
    _store(torch, ops, state, key.to(state["raw"].device), value.to(state["raw"].device),
           slots.to(state["raw"].device), positions.to(state["raw"].device))


def allocate(torch, device, cores):
    layout = fixture_layout()
    raw = torch.full((BLOCKS * layout.page.page_size_bytes,), 0x55, dtype=torch.uint8, device=device)
    wk, wv, tags = layout.views(raw, BLOCKS); tags.fill_(-1)
    n, h, d = QUERIES, HEADS, DIM
    return {"layout": layout, "raw": raw, "wk": wk, "wv": wv, "tags": tags,
        "rotation": torch.eye(d, dtype=torch.float32, device=device),
        "tasks": torch.empty((n * 3, 16), dtype=torch.int64, device=device),
        "positions": torch.empty(n, dtype=torch.int64, device=device),
        "qr": torch.empty((n, h, d), dtype=torch.float32, device=device),
        "rotate_status": torch.empty((n, h), dtype=torch.int32, device=device),
        "partial": torch.empty((n, h, 3, d), dtype=torch.float32, device=device),
        "lse": torch.empty((n, h, 3), dtype=torch.float32, device=device),
        "status": torch.empty((n * 3, 2), dtype=torch.int32, device=device),
        "workspace": torch.empty(cores * 1146880, dtype=torch.uint8, device=device),
        "output": torch.empty((n * h, d), dtype=torch.float32, device=device),
        "output_lse": torch.empty(n * h, dtype=torch.float32, device=device),
        "merge_status": torch.empty(n * h, dtype=torch.int32, device=device),
        "bf16": torch.empty((n, h, d), dtype=torch.bfloat16, device=device),
        "store_status": torch.empty(n, dtype=torch.int32, device=device)}


def execute(ops, state, inputs, reader, rotate, cores):
    layout = state["layout"]
    ops.prepare_attention_tasks_out(inputs["starts"], inputs["lens"], inputs["slots"],
        state["tasks"], state["positions"], HEADS, 1, 64, 256, 1)
    getattr(ops, rotate)(inputs["q"], state["rotation"], state["qr"], state["rotate_status"],
                         False, inputs["slots"])
    getattr(ops, reader)(inputs["q"], state["qr"], inputs["key"], inputs["value"],
        state["rotation"], state["raw"], inputs["table"], state["wk"], state["wv"], state["tags"],
        state["tasks"], state["partial"], state["lse"], state["status"], state["workspace"],
        layout.page.block_size, BLOCKS, BLOCKS * layout.page.conv_bytes, layout.page.ssm_bytes,
        64, 256, 3, 1, DIM ** -0.5, cores)
    ops.merge_lse_out(state["partial"].view(QUERIES * HEADS, 3, DIM),
        state["lse"].view(QUERIES * HEADS, 3), state["output"], state["output_lse"], state["merge_status"])
    state["bf16"].copy_(state["output"].view(QUERIES, HEADS, DIM))
    # Same read-before-write sequence as integration.impl.forward.
    ops.rotate_clip_store_striped_out(inputs["key"], inputs["value"], state["rotation"], state["rotation"],
        inputs["slots"], state["positions"], state["raw"], state["wk"], state["wv"], state["tags"],
        state["store_status"], 2304, BLOCKS, BLOCKS * layout.page.conv_bytes,
        layout.page.ssm_bytes, 64, 259, 0., 0., False)


def check_state(torch, state, inputs, logical, immutable, tolerance):
    torch.npu.synchronize()
    for name in ("status", "rotate_status", "merge_status", "store_status"):
        if bool(state[name].cpu().ne(0).any()):
            raise RuntimeError("APC forward status failed: " + name)
    expected, expected_lse = oracle(torch, logical, inputs)
    torch.testing.assert_close(state["output"].view(QUERIES, HEADS, DIM).cpu(), expected, **tolerance)
    torch.testing.assert_close(state["bf16"].float().cpu(), expected, **tolerance)
    torch.testing.assert_close(state["output_lse"].view(QUERIES, HEADS).cpu(), expected_lse, **tolerance)
    for page, before in immutable.items():
        if not torch.equal(page_bytes(torch, state["raw"], state["layout"], page), before):
            raise RuntimeError("APC suffix modified shared immutable page " + str(page))
    tasks = state["tasks"].cpu()
    counts = {str(i): int(((tasks[:, 7] == i) & (tasks[:, 1] > 0) & (tasks[:, 4] > tasks[:, 3])).sum()) for i in range(3)}
    if any(v <= 0 for v in counts.values()):
        raise RuntimeError("APC gate omitted a live history/window/current source")
    return {"status": "passed", "oracle_queries": QUERIES, "source_leaders": counts,
            "shared_pages_unchanged": True,
            "input_sha256": reuse._fingerprint(inputs), "tasks_sha256": reuse._fingerprint({"tasks": tasks})}


def run_device(torch, ops, config, cores, tolerance):
    device = torch.device("npu:0")
    reader = selected_reader(config)
    rotate = "rotate_vector_init_out" if config.get("experimental_quantization_bundle", False) else "rotate_out"
    state = allocate(torch, device, cores)
    generator = torch.Generator().manual_seed(137081)
    shared = tuple(torch.randn((4608, 1, DIM), generator=generator).to(torch.bfloat16) for _ in range(2))
    seed_page_chain(torch, ops, state, shared, (3, 1))
    immutable = {p: page_bytes(torch, state["raw"], state["layout"], p) for p in (3, 1)}
    logical = (shared, shared)
    cpu = make_inputs(torch, logical, ((3, 1, 5), (3, 1, 2)), 81923)
    inputs = {k: v.to(device) for k, v in cpu.items()}
    execute(ops, state, inputs, reader, rotate, cores)
    first = check_state(torch, state, cpu, logical, immutable, tolerance)
    # Exact suffix bytes independently prove both real stores hit private rows.
    for request, page in enumerate((5, 2)):
        for i in range(4):
            row = 64 + i
            for name, source in (("wk", "key"), ("wv", "value")):
                if not torch.equal(state[name][page, row].cpu().view(torch.int16), cpu[source][request * 4 + i].view(torch.int16)):
                    raise RuntimeError("APC fork suffix bytes differ from original BF16")
    protected_tail = page_bytes(torch, state["raw"], state["layout"], 2)
    first.update(case=STATES[0], contexts=[4608, 4608], suffix_isolation=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        execute(ops, state, inputs, reader, rotate, cores)
    addresses = {k: v.data_ptr() for k, v in inputs.items()}
    logical = (shared, tuple(v[:2304] for v in shared))
    cpu = make_inputs(torch, logical, ((3, 1, 5), (3, 4)), 81924)
    for k, v in cpu.items(): inputs[k].copy_(v)
    graph.replay()
    second = check_state(torch, state, cpu, logical, immutable, tolerance)
    captured = state["bf16"].clone()
    execute(ops, state, inputs, reader, rotate, cores)
    check_state(torch, state, cpu, logical, immutable, tolerance)
    if not torch.equal(captured.view(torch.int16), state["bf16"].view(torch.int16)):
        raise RuntimeError("APC changed-metadata graph differs from eager")
    if not torch.equal(protected_tail, page_bytes(torch, state["raw"], state["layout"], 2)):
        raise RuntimeError("APC changed fork wrote the detached private suffix")
    second.update(case=STATES[1], contexts=[4608, 2304], suffix_isolation=True)
    # Simulate native free/reallocation of private page5 after previous work
    # completed. The pinned native allocator itself is exercised in host tests.
    replacement = tuple(torch.randn((512, 1, DIM), generator=generator).to(torch.bfloat16) for _ in range(2))
    seed_page_chain(torch, ops, state, replacement, (5,))
    logical = (replacement, tuple(v[:2304] for v in shared))
    cpu = make_inputs(torch, logical, ((5,), (3, 4)), 81925)
    for k, v in cpu.items(): inputs[k].copy_(v)
    graph.replay()
    third = check_state(torch, state, cpu, logical, immutable, tolerance)
    third.update(case=STATES[2], contexts=[512, 2304], suffix_isolation=True)
    if addresses != {k: v.data_ptr() for k, v in inputs.items()}:
        raise RuntimeError("APC graph changed input addresses")
    return {"reader": reader, "writer": "rotate_clip_store_striped_out", "cases": [first, second, third],
            "graph_capture": "passed", "graph_replay": "passed", "graph_same_addresses": True,
            "graph_eager_bitwise": "passed", "device_completion": "passed", "precision": "passed"}


def validate_evidence(report):
    wanted = {"status": "passed", "scope": SCOPE, "prefix_caching": True,
              "device_completion": "passed", "precision": "passed", "graph_capture": "passed",
              "graph_replay": "passed", "graph_same_addresses": True, "graph_eager_bitwise": "passed",
              "writer": "rotate_clip_store_striped_out", "rotation_scope": "identity_storage_contract"}
    bad = [k for k, v in wanted.items() if not isinstance(report, dict) or report.get(k) != v]
    cases = report.get("cases", []) if isinstance(report, dict) else []
    if not isinstance(cases, list) or len(cases) != len(STATES): return bad + ["cases"]
    for name, row, contexts in zip(STATES, cases, ([4608, 4608], [4608, 2304], [512, 2304])):
        if not isinstance(row, dict) or any(row.get(k) != v for k, v in {
            "case": name, "contexts": contexts, "status": "passed", "oracle_queries": QUERIES,
            "shared_pages_unchanged": True, "suffix_isolation": True}.items()):
            bad.append(name + ".contract"); continue
        counts = row.get("source_leaders", {})
        if set(counts) != {"0", "1", "2"} or any(type(v) is not int or v <= 0 for v in counts.values()):
            bad.append(name + ".sources")
    if len({c.get("input_sha256") for c in cases if isinstance(c, dict)}) != len(STATES):
        bad.append("changed_inputs")
    if len({c.get("tasks_sha256") for c in cases if isinstance(c, dict)}) != len(STATES):
        bad.append("changed_metadata")
    return bad


def probe(config_path, acceptance_path):
    config = json.loads(config_path.read_text()); selected_reader(config)
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("frozen_before_measurement") is not True or acceptance["fused_attention"] != {"atol": .005, "rtol": .005}:
        raise RuntimeError("APC frozen precision contract changed")
    reuse._active_device(config)
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available(): raise RuntimeError("prefix gate requires real NPU; no CPU substitution")
    torch.npu.set_device(0); torch.set_num_threads(min(4, torch.get_num_threads()))
    from .probe_native_current_fia import _target_geometry
    from .build_ops import normalize_soc
    if _target_geometry(config) != (HEADS, 1, DIM): raise RuntimeError("APC fixture requires Hq6/Hkv1/D256")
    if normalize_soc(torch.npu.get_device_name(0)) != config["soc_version"]: raise RuntimeError("APC target SoC differs")
    from oscar_ascend.ops.loader import validate_build_artifacts, require_capabilities
    manifest = validate_build_artifacts(ROOT / "build/ascendc/build_manifest.json")
    rotate = "rotate_vector_init_out" if config.get("experimental_quantization_bundle", False) else "rotate_out"
    require_capabilities({selected_reader(config), rotate, "prepare_attention_tasks_out",
                          "merge_lse_out", "rotate_clip_store_striped_out"})
    report = {"status": "passed", "scope": SCOPE, "prefix_caching": True,
        "rotation_scope": "identity_storage_contract", "native_allocator": "host_contract_only",
        "model_quality": "not_established", "service_performance": "not_established",
        "fixture_geometry": {"full_block": 2304, "mamba_block": 768, "virtual_block": 128},
        "source_binding": source_binding(), "artifact_signature": manifest["signature"],
        "artifact_sha256": manifest["sha256"],
        **run_device(torch, torch.ops.oscar_ascend_ops, config, reuse._core_count(torch, config), acceptance["fused_attention"])}
    bad = validate_evidence(report)
    if bad: report.update(status="failed", evidence_mismatches=bad)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/acceptance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try: report = probe(args.config, args.acceptance)
    except Exception as error:
        traceback.print_exc(); report = {"status": "failed", "first_error": str(error)}
    atomic_json(args.output, report)
    print("[oscar] PERF_PREFIX_CACHE " + json.dumps({"status": report["status"],
        "device_completion": report.get("device_completion", "not_established"),
        "graph_replay": report.get("graph_replay", "not_established"), "report": str(args.output)}), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__": raise SystemExit(main())
