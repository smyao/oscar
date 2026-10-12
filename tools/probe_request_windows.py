"""Real-NPU byte/graph gate for request-stable exact windows.

Archive G26-G34/#13-22/#37-49/#69/#126-128/#145/#151/#153-155: independent
PR oracle, BF16 poison bits, request identity, fixed-address changed-input
replay, signed artifacts and separate accuracy/graph evidence. D.4: this is
phase1_stores; never restores history or infers speed from a byte comparison.
No CPU fallback; the overall gate requires both writer and reader evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import traceback

from .phase import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OLD_OP = "rotate_clip_store_striped_out"
NEW_OP = "rotate_clip_store_request_window_out"
INPUTS = ("key", "value", "rk", "rv", "slots", "positions", "qstarts", "mapping", "baseline_slots")
OUTPUTS = ("packed", "raw_key", "raw_value", "tags", "status", "window_status")


class RequestWindowGateError(RuntimeError):
    pass


def case_plan():
    common = {"dim": 256, "dtype_name": "bf16", "segment": 2816}
    return tuple(common | spec for spec in (
        {"name": "segment_boundary", "lengths": (9,), "in_page": 2813, "hadamard": True},
        {"name": "physical_page_boundary", "lengths": (9,), "in_page": 5628},
        {"name": "reordered_hkv2", "lengths": (4, 7, 3), "heads": 2, "hadamard": True},
        {"name": "negative_holes", "lengths": (333,), "mode": "holes"},
        {"name": "rollback_0_to_3", "lengths": (700,), "mode": "rollback", "in_page": 0},
        {"name": "numeric_and_address_faults", "lengths": (17,), "heads": 2, "mode": "faults"},
        {"name": "bad_epoch", "mode": "bad_epoch"},
        {"name": "positive_qstarts_origin", "mode": "positive_qstarts_origin"},
        {"name": "active_tail", "mode": "active_tail", "token_capacity": 12},
        {"name": "clipped_hadamard", "lengths": (9,), "mode": "clip", "hadamard": True},
    ))


def _fixture_module():
    return importlib.import_module("csrc.experiments.request_windows_store.fixture")


def _poison(torch, count, pattern):
    out = torch.tensor(pattern, dtype=torch.uint8).repeat((count + len(pattern) - 1) // len(pattern))[:count].clone()
    out[-64:] = 0xAD
    return out


def _raw_view(torch, storage, owners, rows, heads, dim, stride):
    return storage.view(torch.bfloat16).as_strided((owners, rows, heads, dim), (stride, heads * dim, dim, 1))


def _tag_view(torch, storage, owners, rows, stride):
    return storage.view(torch.int64).as_strided((owners, rows), (stride, 1))


def make_fixture(torch, spec):
    """Rebuild legal production views; never loosen binding's32B stride gate."""
    module = _fixture_module()
    values, _, meta = module.build_case(**spec)
    shape = dict(meta["shape_fields"])
    if shape["dim"] != 256 or shape["dtype"] != 2 or shape["segment_tokens"] % 128:
        raise RequestWindowGateError("NPU gate requires legal D256/BF16/128-aligned segment geometry")
    h, d, cap, rows = shape["heads"], 256, shape["capacity"], shape["sink"] + shape["recent_with_spec"]
    shape.update(offset=64, page_stride=shape["segment_tokens"] * h * 136 + 64,
                 key_stride=rows * h * d + 16, value_stride=rows * h * d + 32,
                 tag_stride=((rows + 3) // 4) * 4)
    cpu = {name: values[name].contiguous().clone() for name in INPUTS}
    if tuple(cpu["mapping"].shape) != (shape["requests"], 2):
        raise RequestWindowGateError("Fixture must exercise sliced request_map[requests,2]")
    initial = {
        "packed": torch.full((shape["offset"] + 2 * shape["blocks"] * shape["page_stride"] + 64,), 0xAD, dtype=torch.uint8),
        "raw_key": _poison(torch, cap * shape["key_stride"] * 2 + 64, (0xC5, 0x7F)),
        "raw_value": _poison(torch, cap * shape["value_stride"] * 2 + 64, (0xC7, 0x7F)),
        "tags": _poison(torch, cap * shape["tag_stride"] * 8 + 64, (0xAD,)),
        "status": torch.full((shape["tokens"], h), 0x55555555, dtype=torch.int32),
        "window_status": torch.full((cap,), 0x55555555, dtype=torch.int32),
    }
    fixture = {"name": spec["name"], "shape": shape, "cpu": cpu, "initial": initial}
    fixture["expected"] = oracle(torch, fixture, initial)
    return fixture


def oracle(torch, fixture, initial):
    """Independent PR quantizer and logical ring oracle, never NPU-derived."""
    from oscar_ascend.ops import reference
    from tools.experiments.active_identity_store_fixture import _side, _striped
    module = _fixture_module()
    s, cpu = fixture["shape"], fixture["cpu"]
    out = {name: tensor.clone() for name, tensor in initial.items()}
    out["status"].zero_(); out["window_status"].zero_()
    n, h, d, segment, blocks = (s[k] for k in ("tokens", "heads", "dim", "segment_tokens", "blocks"))
    baseline_slots = cpu["slots"].clone()
    for token in range(n):
        slot, pos = int(cpu["slots"][token]), int(cpu["positions"][token])
        if slot < 0:
            continue
        if slot >= blocks * 2 * segment:
            out["status"][token] = 1; continue
        page, inside = divmod(slot, 2 * segment)
        region, within = divmod(inside, segment)
        baseline_slots[token] = (region * blocks + page) * segment + within
        if pos < 0:
            out["status"][token] = 4; continue
        for head in range(h):
            ke, key = _side(cpu["key"][token, head], cpu["rk"].T, s["key_clip"])
            ve, value = _side(cpu["value"][token, head], cpu["rv"].T, s["value_clip"])
            out["status"][token, head] = ke or ve
            if ke or ve:
                continue
            encoded = _striped(reference.encode_kv(key, value), d)
            address = module.packed_address(slot, head, segment=segment, blocks=blocks,
                        offset=s["offset"], stride=s["page_stride"], heads=h, dim=d)
            out["packed"][address:address + 136] = encoded
    cpu["baseline_slots"] = baseline_slots
    winners, errors, _ = module.logical_winners(cpu["qstarts"].tolist(), cpu["mapping"].tolist(),
        cpu["slots"].tolist(), cpu["positions"].tolist(), out["status"].tolist(), s["capacity"],
        s["sink"], s["recent_with_spec"], s["position_bits"])
    out["window_status"][:len(errors)] = torch.tensor(errors, dtype=torch.int32)
    for req, chosen in winners.items():
        lease, epoch = cpu["mapping"][req].tolist()
        for ring, token in chosen.items():
            for source, target, stride in (("key", "raw_key", "key_stride"), ("value", "raw_value", "value_stride")):
                offset = 2 * (lease * s[stride] + ring * h * d)
                out[target][offset:offset + 2 * h * d] = cpu[source][token].to(torch.bfloat16).contiguous().view(torch.uint8).flatten()
            offset = 8 * (lease * s["tag_stride"] + ring)
            out["tags"][offset:offset + 8] = torch.tensor([(epoch << s["position_bits"]) | int(cpu["positions"][token])], dtype=torch.int64).view(torch.uint8)
    return out


def _old_window_oracle(torch, fixture):
    s, cpu, status = fixture["shape"], fixture["cpu"], fixture["expected"]["status"]
    owners, rows, h, d = 2 * s["blocks"], s["sink"] + s["recent_with_spec"], s["heads"], s["dim"]
    out = {"raw_key": _poison(torch, owners * rows * h * d * 2 + 64, (0xC5, 0x7F)),
           "raw_value": _poison(torch, owners * rows * h * d * 2 + 64, (0xC7, 0x7F)),
           "tags": _poison(torch, owners * rows * 8 + 64, (0xAD,))}
    initial = {k: v.clone() for k, v in out.items()}
    for token in range(s["tokens"]):
        slot, pos = int(cpu["baseline_slots"][token]), int(cpu["positions"][token])
        if slot < 0 or slot >= owners * s["segment_tokens"] or pos < 0:
            continue
        page, inside = divmod(slot, s["segment_tokens"])
        recent = s["recent_with_spec"]
        keep = pos >= s["sink"]
        if keep and token + recent < s["tokens"]:
            later = int(cpu["baseline_slots"][token + recent])
            keep = not (later >= 0 and later // s["segment_tokens"] == page and later == slot + recent)
        if pos >= s["sink"] and not keep:
            continue
        ring = pos if pos < s["sink"] else s["sink"] + inside % recent
        for head in range(h):
            if int(status[token, head]): continue
            for source, target in (("key", "raw_key"), ("value", "raw_value")):
                offset = ((page * rows + ring) * h + head) * d * 2
                out[target][offset:offset + d * 2] = cpu[source][token, head].to(torch.bfloat16).view(torch.uint8)
            if head == 0:
                offset = (page * rows + ring) * 8
                out["tags"][offset:offset + 8] = torch.tensor([inside], dtype=torch.int64).view(torch.uint8)
    return initial, out


def _fingerprint(torch, values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        digest.update(name.encode() + b"\0")
        digest.update(bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def _equal(torch, actual, expected, label):
    got = actual.detach().cpu().contiguous().view(torch.uint8).flatten()
    want = expected.contiguous().view(torch.uint8).flatten()
    if got.shape != want.shape or not bool(torch.equal(got, want)):
        detail = []
        if got.shape == want.shape:
            for index in (got != want).nonzero().flatten()[:8].tolist():
                detail.append({"byte": index, "got": int(got[index]), "expected": int(want[index])})
        raise RequestWindowGateError(f"{label} differs: {detail}")


def _buffers(torch, fixture, device, initial=None):
    s = fixture["shape"]
    values = initial or fixture["initial"]
    out = {name: tensor.to(device).clone() for name, tensor in values.items()}
    out["wk"] = _raw_view(torch, out["raw_key"], s["capacity"], s["sink"] + s["recent_with_spec"], s["heads"], s["dim"], s["key_stride"])
    out["wv"] = _raw_view(torch, out["raw_value"], s["capacity"], s["sink"] + s["recent_with_spec"], s["heads"], s["dim"], s["value_stride"])
    out["window_tags"] = _tag_view(torch, out["tags"], s["capacity"], s["sink"] + s["recent_with_spec"], s["tag_stride"])
    return out


def store_arguments(tensors, buffers, shape):
    return tuple(tensors[k] for k in ("key", "value", "rk", "rv", "slots", "positions", "qstarts", "mapping")) + tuple(
        buffers[k] for k in ("packed", "wk", "wv", "window_tags", "status", "window_status")) + (
        2 * shape["segment_tokens"], shape["blocks"], shape["offset"], shape["page_stride"],
        shape["sink"], shape["recent_with_spec"], shape["key_clip"], shape["value_clip"],
        bool(shape["hadamard"]), shape["capacity"], shape["position_bits"])


def _launch_new(ops, tensors, buffers, fixture):
    getattr(ops, NEW_OP)(*store_arguments(tensors, buffers, fixture["shape"]))


def _check_new(torch, buffers, expected, label):
    # Both complete status tensors are checked, including unused capacity tail.
    for name in ("status", "window_status", "packed", "raw_key", "raw_value", "tags"):
        _equal(torch, buffers[name], expected[name], f"{label}/{name}")


def run_store_case(torch, ops, fixture, device):
    if device.type != "npu": raise RequestWindowGateError("real NPU device required")
    s = fixture["shape"]
    tensors = {k: v.to(device) for k, v in fixture["cpu"].items()}
    buffers = _buffers(torch, fixture, device)
    old_initial, old_expected = _old_window_oracle(torch, fixture)
    old = {k: v.to(device) for k, v in old_initial.items()}
    old["packed"] = fixture["initial"]["packed"].to(device).clone()
    old["status"] = fixture["initial"]["status"].to(device).clone()
    owners, rows = 2 * s["blocks"], s["sink"] + s["recent_with_spec"]
    h, d = s["heads"], s["dim"]
    old_wk = _raw_view(torch, old["raw_key"], owners, rows, h, d, rows * h * d)
    old_wv = _raw_view(torch, old["raw_value"], owners, rows, h, d, rows * h * d)
    old_tags = _tag_view(torch, old["tags"], owners, rows, rows)
    getattr(ops, OLD_OP)(tensors["key"], tensors["value"], tensors["rk"], tensors["rv"],
        tensors["baseline_slots"], tensors["positions"], old["packed"], old_wk, old_wv, old_tags,
        old["status"], s["segment_tokens"], owners, s["offset"], s["page_stride"],
        s["sink"], s["recent_with_spec"], s["key_clip"], s["value_clip"], bool(s["hadamard"]))
    _launch_new(ops, tensors, buffers, fixture)
    torch.npu.synchronize()
    for name in ("packed", "status"):
        _equal(torch, old[name], fixture["expected"][name], f"old/{fixture['name']}/{name}")
        _equal(torch, buffers[name], old[name].cpu(), f"old/new/{fixture['name']}/{name}")
    for name in old_expected: _equal(torch, old[name], old_expected[name], f"old/window/{name}")
    _check_new(torch, buffers, fixture["expected"], fixture["name"])
    for name in INPUTS: _equal(torch, tensors[name], fixture["cpu"][name], f"input/{name}")
    return {"case": fixture["name"], "status": "passed", "device_completion": "passed",
            "independent_pr_bytes": "passed", "old_new_packed_status": "bitwise_passed",
            "sidecar_and_poison_bytes": "passed", "old_snapshot_and_guard_bytes": "passed",
            "packed_status_checked": True, "window_status_checked": True, "input_immutable": True,
            "request_map_shape": list(fixture["cpu"]["mapping"].shape),
            "expected_packed_status_nonzero": int(fixture["expected"]["status"].count_nonzero()),
            "expected_window_status": fixture["expected"]["window_status"].tolist(),
            "input_sha256": _fingerprint(torch, fixture["cpu"]), "shape": s}


def run_store_graph(torch, ops, device, *, evidence=None, checkpoint=None):
    if device.type != "npu": raise RequestWindowGateError("real NPU graph required")
    evidence = {} if evidence is None else evidence
    evidence.update(status="running", capture="not_run", replay="not_run", real_replay_stages=[])
    spec = {"name": "graph_q1_32_n128", "lengths": (1,) * 32, "heads": 1,
            "dim": 256, "dtype_name": "bf16", "segment": 2816, "in_page": 2813,
            "token_capacity": 128, "window_capacity": 64, "hadamard": True}
    fixture = make_fixture(torch, spec)
    tensors = {k: v.to(device) for k, v in fixture["cpu"].items()}
    buffers = _buffers(torch, fixture, device)
    pointers = {k: v.data_ptr() for k, v in tensors.items()} | {"out/" + k: buffers[k].data_ptr() for k in OUTPUTS}
    tensors["slots"].fill_(-1); tensors["positions"].fill_(-1)
    tensors["mapping"][:, 0].fill_(-1); tensors["mapping"][:, 1].zero_()
    noop = {k: v.clone() for k, v in fixture["initial"].items()}
    noop["status"].zero_(); noop["window_status"].zero_()
    # Warm only the exact all-padding capture path, without a published lease.
    _launch_new(ops, tensors, buffers, fixture); torch.npu.synchronize()
    _check_new(torch, buffers, noop, "capture_padding_warmup")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        _launch_new(ops, tensors, buffers, fixture)
    graph.replay(); torch.npu.synchronize()
    _check_new(torch, buffers, noop, "capture_padding_replay")
    evidence.update(capture="passed",capture_all_padding_unpublished_map=True)
    if checkpoint: checkpoint()
    stages = evidence["real_replay_stages"]
    previous = {k: v.clone() for k, v in fixture["initial"].items()}
    previous_input = None
    for stage in range(2):
        if stage:
            live = fixture["cpu"]["slots"] >= 0
            fixture["cpu"]["key"][live] *= -1
            fixture["cpu"]["value"][live] = fixture["cpu"]["value"][live].flip(-1)
            fixture["cpu"]["positions"][live] += 521
            fixture["cpu"]["slots"][live] += 1
            fixture["cpu"]["mapping"][:, 0] = (fixture["cpu"]["mapping"][:, 0] + 1) % 32
            fixture["cpu"]["mapping"][:, 1] += 101
        expected = oracle(torch, fixture, previous)
        for name in INPUTS: tensors[name].copy_(fixture["cpu"][name])
        for name in ("status", "window_status"): buffers[name].fill_(-99)
        graph.replay(); torch.npu.synchronize()
        _check_new(torch, buffers, expected, f"graph_real_stage{stage}")
        eager = _buffers(torch, fixture, device, previous)
        _launch_new(ops, tensors, eager, fixture); torch.npu.synchronize()
        _check_new(torch, eager, expected, f"eager_real_stage{stage}")
        for name in OUTPUTS: _equal(torch, buffers[name], eager[name].cpu(), f"graph/eager/{name}")
        actual_pointers = {k: v.data_ptr() for k, v in tensors.items()} | {"out/" + k: buffers[k].data_ptr() for k in OUTPUTS}
        if actual_pointers != pointers: raise RequestWindowGateError("graph tensors changed address")
        fingerprint = _fingerprint(torch, fixture["cpu"])
        if fingerprint == previous_input: raise RequestWindowGateError("graph changed-input stage did not change input")
        stages.append({"stage": stage, "status": "passed", "input_sha256": fingerprint,
                       "all_bytes_and_status": "passed", "same_addresses": True,
                       "lease_epoch_position_oracle": "passed", "eager_parity": "bitwise_passed"})
        if checkpoint: checkpoint()
        previous_input = fingerprint
        previous = expected
    evidence.update({"status": "passed", "capture": "passed", "replay": "passed",
            "capture_all_padding_unpublished_map": True, "real_replay_stages": stages,
            "changed_lease_epoch_position_and_kv": True, "preserved_old_tags_checked_as_bytes": True,
            "reader_stale_tag_rejection": "pending_reader_gate"})
    return evidence


def empty_report():
    return {"schema_version": 1, "status": "pending", "evidence_kind": "real_npu_bytes_and_graph_only",
            "device_completion": "not_run", "accuracy": {"store": {"status": "not_run", "cases": []},
            "reader": {"status": "pending", "reason": "request-window reader gate not implemented here"}},
            "graph": {"store": {"status": "not_run", "capture": "not_run", "replay": "not_run"},
                      "reader": {"status": "pending"}}, "performance": {"status": "not_run"},
            "full_service_acceptance": "not_run", "full_model_quality": "not_run"}


def final_status(report):
    if report.get("first_error"): return "failed"
    if report.get("device_completion") != "passed": return "failed"
    if not report.get("artifact_signature") or not report.get("artifact_sha256") or report.get("device", {}).get("logical") != "npu:0": return "failed"
    accuracy, graph = report["accuracy"], report["graph"]
    store = accuracy["store"]
    rows = store.get("cases", [])
    expected = {spec["name"] for spec in case_plan()}
    if store.get("status") != "passed" or {r.get("case") for r in rows} != expected or len(rows) != len(expected): return "failed"
    required = ("packed_status_checked", "window_status_checked", "input_immutable")
    if any(r.get("status") != "passed" or any(r.get(k) is not True for k in required)
           or r.get("independent_pr_bytes") != "passed" or r.get("sidecar_and_poison_bytes") != "passed" for r in rows): return "failed"
    gs = graph["store"]
    if any(gs.get(k) != "passed" for k in ("status", "capture", "replay")) or gs.get("capture_all_padding_unpublished_map") is not True:
        return "failed"
    stages=gs.get("real_replay_stages", [])
    if len(stages) < 2 or gs.get("changed_lease_epoch_position_and_kv") is not True: return "failed"
    if any(row.get("status")!="passed" or row.get("same_addresses") is not True or row.get("all_bytes_and_status")!="passed" for row in stages):return "failed"
    if len({row.get("input_sha256") for row in stages})!=len(stages):return "failed"
    reader, reader_graph = accuracy["reader"], graph["reader"]
    if reader.get("status") == "failed" or reader_graph.get("status") == "failed":return "failed"
    if reader.get("status") != "passed" or reader_graph.get("status") != "passed":return "pending"
    if reader.get("device_completion") != "passed" or reader.get("independent_oracle") != "passed":return "failed"
    if any(reader_graph.get(k)!="passed" for k in ("capture","replay","stale_epoch_rejection")):return "failed"
    if reader_graph.get("same_address_changed_input") is not True:return "failed"
    from .request_window_reader_gate import ReaderGateError, validate_evidence
    try:
        validate_evidence({"accuracy":reader,"graph":reader_graph},
                          c4_member_pipeline=report.get("c4_member_pipeline",False))
    except ReaderGateError:
        return "failed"
    return "passed"


def _require_npu(target):
    from .probe_native_current_fia import _select_target_npu
    _select_target_npu(target)
    import torch
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RequestWindowGateError("real target NPU required; torch_npu unavailable") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RequestWindowGateError("real target NPU required; CPU cannot pass this gate")
    from .build_ops import normalize_soc
    torch.npu.set_device(0)
    if normalize_soc(torch.npu.get_device_name(0)) != target["soc_version"]:
        raise RequestWindowGateError("selected NPU SoC differs from target config")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch, torch.device("npu:0")


def probe(config_path, acceptance_path, manifest_path, *, report=None, checkpoint=None):
    report = empty_report() if report is None else report
    target = json.loads(config_path.read_text())
    if type(target.get("experimental_c4_member_pipeline",False)) is not bool:
        raise RequestWindowGateError("C4 member pipeline selection must be an explicit boolean")
    report["c4_member_pipeline"] = target.get("experimental_c4_member_pipeline",False)
    acceptance_bytes = acceptance_path.read_bytes(); acceptance = json.loads(acceptance_bytes)
    if acceptance.get("frozen_before_measurement") is not True or acceptance.get("pack_unpack", {}).get("exact") is not True:
        raise RequestWindowGateError("frozen exact pack policy is required")
    report["acceptance_sha256"] = hashlib.sha256(acceptance_bytes).hexdigest()
    torch, device = _require_npu(target)
    from oscar_ascend.ops.loader import require_capabilities, validate_build_artifacts
    manifest = validate_build_artifacts(manifest_path)
    require_capabilities({OLD_OP, NEW_OP}, manifest_path)
    report["artifact_signature"] = manifest["signature"]; report["artifact_sha256"] = manifest["sha256"]
    report["device"] = {"logical": str(device), "physical_selection": target["devices"], "name": torch.npu.get_device_name(0)}
    ops = torch.ops.oscar_ascend_ops
    for spec in case_plan():
        report["active_phase"]="store_accuracy";report["active_case"]=spec["name"]
        row = run_store_case(torch, ops, make_fixture(torch, spec), device)
        report["accuracy"]["store"]["cases"].append(row)
        if checkpoint: checkpoint(report)
    report["accuracy"]["store"]["status"] = "passed"
    report["device_completion"] = "passed"
    if checkpoint: checkpoint(report)
    report["active_phase"]="store_graph";report.pop("active_case",None)
    report["graph"]["store"] = run_store_graph(torch, ops, device,evidence=report["graph"]["store"],
        checkpoint=(lambda:checkpoint(report)) if checkpoint else None)
    from .probe_history_reuse import _core_count
    from .request_window_reader_gate import run_reader_gate
    report["active_phase"]="reader"
    def reader_checkpoint(current):
        report["accuracy"]["reader"] = current["accuracy"]
        report["graph"]["reader"] = current["graph"]
        report["reader_detail"] = current
        if checkpoint: checkpoint(report)
    reader_cores=_core_count(torch,target)
    report["device"]["cube_cores"]=reader_cores
    reader = run_reader_gate(torch, ops, device, cores=reader_cores,
        acceptance=acceptance, manifest_path=manifest_path, checkpoint=reader_checkpoint,
        c4_member_pipeline=target.get("experimental_c4_member_pipeline",False))
    reader_checkpoint(reader)
    if reader.get("status") != "passed":
        raise RequestWindowGateError(reader.get("first_error", "reader gate did not complete"))
    report["active_phase"]="complete"
    report["status"] = final_status(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/acceptance.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "build/ascendc/build_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = empty_report()
    try:
        report = probe(args.config, args.acceptance, args.manifest, report=report,
                       checkpoint=lambda current: atomic_json(args.output, current))
    except Exception as exc:
        traceback.print_exc()
        report["status"] = "failed"; report["first_error"] = str(exc)
        report["error_type"] = type(exc).__name__
        if report.get("active_phase")=="store_accuracy":report["accuracy"]["store"]["status"]="failed"
        if report.get("active_phase")=="store_graph":report["graph"]["store"]["status"]="failed"
    atomic_json(args.output, report)
    print("REQUEST_WINDOW_GATE " + json.dumps({"status": report["status"],
          "store_accuracy": report["accuracy"]["store"]["status"],
          "store_graph": report["graph"]["store"]["status"],
          "reader": report["accuracy"]["reader"]["status"], "performance": "not_run",
          "report": str(args.output)}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__": raise SystemExit(main())
