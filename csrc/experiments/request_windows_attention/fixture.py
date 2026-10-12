"""Archive #17-20/#126-129/#145/#154/#155 and D.4: independent logical fixtures.

The old page snapshots and new request windows are separately scattered from
logical BF16 K/V. Both compressed layouts are independently encoded by the PR
oracle. This is an offline fixture, never a production history restoration.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import probe_history_reuse as reuse
from tools.striped_fixture import canonical_slots_to_striped, raw_to_striped
from csrc.experiments.window_batch_decode.fixture import _bytes


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    contexts: tuple[int, ...]
    block: int = 512
    padded: int = 8
    order: tuple[int, ...] | None = None
    fault: str | None = None
    qlen: int | None = None
    recent: int = 256
    suppress_current: bool = False
    second_segment: bool = False
    seed_offset: int = 0
    splits: int = 1


def case_plan():
    return (
        Case("q4_cross_segments_pages", "q4", (1301, 511), padded=12),
        Case("q4_reordered", "q4", (1301, 511), padded=12, order=(1, 0)),
        Case("q1_cross_segments_pages", "q1", (1299, 1025, 511)),
        Case("q1_reordered", "q1", (1299, 1025, 511), order=(2, 0, 1)),
        Case("q1_native2816_to5632", "q1", (5901,), block=2816, padded=2),
        Case("q4_stale_epoch", "q4", (511,), fault="stale_epoch"),
        Case("q1_poison_stale", "q1", (511,), fault="poison_stale"),
        Case("q1_bad_tag", "q1", (511,), fault="bad_tag"),
        Case("q1_invalid_slot", "q1", (511,), fault="invalid_slot"),
        Case("q1_invalid_epoch", "q1", (511,), fault="invalid_epoch"),
        Case("q1_position_overflow", "q1", (511,), fault="position_overflow"),
    )


def cluster_case_plan():
    return tuple(Case("c16_history_window_"+("segment1" if shifted else "segment0"),
        "c16", (65,), padded=336, qlen=336, recent=1, suppress_current=True,
        second_segment=shifted) for shifted in (False, True))


def make_case(torch, case):
    from oscar_ascend.ops.reference import encode_kv
    qlen = case.qlen if case.qlen is not None else 4 if case.mode == "q4" else 1
    recent = case.recent
    count, hk, hq, dim = len(case.contexts), 1, 6, 256
    active = count*qlen
    if case.padded < active or case.block % 128: raise ValueError("invalid fixture geometry")
    logical = {}
    def observe(request, begin, q, old_k, old_v, ck, cv):
        logical[request] = (old_k.clone(), old_v.clone())
    spec = reuse.Shape(case.name, (qlen,)*count, case.contexts, dim, hk, case.splits, False,
                       padded_tokens=case.padded, slot_context=case.mode == "q1", recent_tokens=recent)
    # The saved PR helper's page-size knob changes allocation only. Restore it
    # before returning; no native/project production configuration is changed.
    saved_block, saved_seed = reuse.BLOCK_TOKENS, reuse.SEED
    try:
        reuse.BLOCK_TOKENS = case.block
        reuse.SEED += case.seed_offset
        old = reuse.make_fixture(torch, spec, on_request=observe)
        old_raw = raw_to_striped(old)
    finally:
        reuse.BLOCK_TOKENS = saved_block
        reuse.SEED = saved_seed
    baseline = old["cpu"]
    inputs = {key: baseline[key].clone() for key in ("q", "qr", "ck", "cv", "rv", "starts", "lens")}
    for key in ("table", "wk", "wv", "tags", "slots"):
        inputs["old_"+key] = baseline[key].clone()
    stride = 393216 if case.block == 2816 else case.block*hk*136
    if stride != old["stride"]:
        padded_raw = torch.full((reuse.PREFIX+old["blocks"]*stride,), 0xA5, dtype=torch.uint8)
        for page in range(old["blocks"]):
            src = reuse.PREFIX+page*old["stride"]
            dst = reuse.PREFIX+page*stride
            padded_raw[dst:dst+old["stride"]] = old_raw[src:src+old["stride"]]
        old_raw = padded_raw
    inputs["old_raw"] = old_raw
    # A different permutation and a different physical block width ensure
    # candidate addresses cannot accidentally reuse the baseline arrangement.
    new_b = 2*case.block
    shift = case.block if case.second_segment else 0
    budgets = [math.ceil((context+qlen+shift)/new_b) for context in case.contexts]
    blocks = sum(budgets)
    pages = list(reversed(range(blocks)))
    assignments, cursor = [], 0
    for budget in budgets:
        assignments.append(pages[cursor:cursor+budget]); cursor += budget
    raw = torch.full((reuse.PREFIX+2*blocks*stride,), 0x6D, dtype=torch.uint8)
    capacity = count+3
    window_rows = reuse.SINK+recent+reuse.SPECULATIVE
    wk = torch.full((capacity, window_rows, hk, dim), float("nan"), dtype=torch.bfloat16)
    wv = torch.full_like(wk, float("nan"))
    tags = torch.full((capacity, window_rows), -1, dtype=torch.int64)
    mapping = torch.full((capacity, 2), -1, dtype=torch.int64)
    table = torch.full_like(baseline["table"], -1)
    slots = torch.full_like(baseline["slots"], -1)
    bits = 8 if case.fault == "position_overflow" else 19
    rk = reuse._hadamard(torch, dim)
    for request, context in enumerate(case.contexts):
        old_k, old_v = logical[request]
        striped = canonical_slots_to_striped(encode_kv(old_k.float()@rk, old_v.float()@inputs["rv"]))
        slot, epoch = capacity-1-request, 11+request*3
        mapping[request] = torch.tensor((slot, epoch))
        for position in range(context):
            physical_position = position+shift
            page = assignments[request][physical_position//new_b]
            inpage = physical_position % new_b
            segment, row = divmod(inpage, case.block)
            offset = reuse.PREFIX+page*stride+segment*blocks*stride+row*hk*136
            raw[offset:offset+hk*136] = striped[position].flatten()
        for column in range(math.ceil((context+qlen)/128)):
            physical_position = column*128+shift
            page = assignments[request][physical_position//new_b]
            table[request, column] = page*(new_b//128)+(physical_position%new_b)//128
        for local in range(qlen):
            position = context+local+shift
            page = assignments[request][position//new_b]
            slots[request*qlen+local] = page*new_b+position%new_b
        precise = set(range(min(reuse.SINK, context))) | set(range(max(0, context-recent), context))
        for position in sorted(precise):
            row = position if position < reuse.SINK else reuse.SINK+position%(recent+reuse.SPECULATIVE)
            if position < 1 << bits:
                wk[slot, row], wv[slot, row] = old_k[position], old_v[position]
                tags[slot, row] = (epoch << bits) | position
    inputs.update(new_raw=raw, new_table=table, new_slots=slots, new_wk=wk,
                  new_wv=wv, new_tags=tags, request_map=mapping)
    mirrored_fault = case.fault in {"stale_epoch", "poison_stale", "bad_tag"}
    if case.fault:
        if count != 1 or case.contexts != (511,): raise ValueError("fault domain must be explicit")
        slot, epoch = map(int, mapping[0])
        if case.fault == "invalid_slot": mapping[0, 0] = capacity
        elif case.fault == "invalid_epoch": mapping[0, 1] = 0
        elif case.fault != "position_overflow":
            position = 501
            row = reuse.SINK+position%(recent+reuse.SPECULATIVE)
            tags[slot, row] = ((epoch-1) << bits) | position if case.fault == "stale_epoch" else -1
            physical = old["page_assignments"][0][position//case.block]
            old_row = reuse.SINK+(position%case.block)%(recent+reuse.SPECULATIVE)
            inputs["old_tags"][physical, old_row] = -1
            if case.fault == "poison_stale":
                wv[slot, row, 0, 0] = float("nan")
                inputs["old_wv"][physical, old_row, 0, 0] = float("nan")
    output = torch.zeros((case.padded, hq, dim), dtype=torch.float32)
    lse = torch.full((case.padded, hq), -float("inf"), dtype=torch.float32)
    positions = torch.full((case.padded,), -1, dtype=torch.int64)
    for token, (values, logs) in old["expected"].items(): output[token], lse[token] = values, logs
    if qlen > 128 or case.suppress_current:
        # Large-cluster gates check every query, not the original helper's
        # sparse diagnostic sample. The reduced CPU case omits only source2.
        from oscar_ascend.ops.reference import attention, decode_kv
        for request, context in enumerate(case.contexts):
            old_k, old_v = logical[request]
            decoded_k, decoded_v = decode_kv(encode_kv(old_k.float()@rk, old_v.float()@inputs["rv"]), dim)
            decoded_k, decoded_v = decoded_k@rk.T, decoded_v@inputs["rv"].T
            for local in range(qlen):
                cut = min(context, max(reuse.SINK, context+local+1-recent))
                keys, values = old_k.float().clone(), old_v.float().clone()
                keys[reuse.SINK:cut], values[reuse.SINK:cut] = decoded_k[reuse.SINK:cut], decoded_v[reuse.SINK:cut]
                token = request*qlen+local
                if not case.suppress_current:
                    keys = torch.cat((keys,inputs["ck"][request*qlen:token+1].float()))
                    values = torch.cat((values,inputs["cv"][request*qlen:token+1].float()))
                result = attention(inputs["q"][token:token+1],keys,values,scale=dim**-.5,causal=False)
                output[token], lse[token] = result.output[0], result.lse[0]
    for request, context in enumerate(case.contexts):
        positions[request*qlen:(request+1)*qlen] = torch.arange(context, context+qlen)
    inputs.update(expected_output=output, expected_lse=lse, expected_positions=positions)
    if case.order is not None:
        if sorted(case.order) != list(range(count)): raise ValueError("invalid request permutation")
        rows = torch.tensor([request*qlen+local for request in case.order for local in range(qlen)])
        for key in ("q", "qr", "ck", "cv", "old_slots", "new_slots", "expected_output", "expected_lse", "expected_positions"):
            inputs[key][:active] = inputs[key][rows].clone()
        for key in ("old_table", "new_table", "lens"):
            inputs[key] = inputs[key][list(case.order)].clone()
        inputs["request_map"][:count] = mapping[list(case.order)].clone()
    return {"case": case, "inputs": inputs, "shape": (
        case.padded, hq, hk, dim, count, table.shape[1], case.block, old["blocks"],
        new_b, blocks, reuse.PREFIX, stride, capacity, bits, reuse.SINK, recent,
        reuse.SPECULATIVE, case.splits, 2), "flags": (int(bool(case.fault)), int(mirrored_fault),
            1 if case.fault in {"invalid_slot", "invalid_epoch"} else 4 if case.fault else 0),
        "logical": logical, "old_assignments": old["page_assignments"], "new_assignments": assignments,
        "oracle_scope": "source0_and_source1_only" if case.suppress_current else "complete_three_sources"}


def export(directory, selected=None, *, cluster_only=False):
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    policy = json.loads((ROOT/"configs/acceptance.json").read_text())
    if policy.get("frozen_before_measurement") is not True: raise ValueError("unfrozen oracle")
    rows = []
    for case in cluster_case_plan() if cluster_only else case_plan():
        if selected and case.name not in selected: continue
        fixture = make_case(torch, case)
        path = directory/case.name; path.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name, tensor in fixture["inputs"].items():
            content = _bytes(tensor); (path/(name+".bin")).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        (path/"shape.txt").write_text(" ".join(map(str, fixture["shape"]))+"\n")
        (path/"flags.txt").write_text(" ".join(map(str, fixture["flags"]))+"\n")
        (path/"suppress_current.txt").write_text(f"{int(case.suppress_current)}\n")
        tolerance = policy["fused_attention"]
        (path/"tolerance.txt").write_text(f"{tolerance['atol']} {tolerance['rtol']}\n")
        rows.append({"name": case.name, "mode": case.mode, "shape": fixture["shape"],
            "flags": fixture["flags"], "sha256": hashes, "path": str(path.resolve()),
            "measurement_scope": "bounded_address_and_numerical_contract_only",
            "oracle_scope": fixture["oracle_scope"]})
        print(f"REQUEST_WINDOW_FIXTURE {case.name}", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "host_generated_only", "cases": rows,
                "baseline": "current_attention_cv_decode_bundle", "cpu_debug": "not_run", "npu": "not_run"}
    (directory/"cases.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--cluster-only", action="store_true")
    args = parser.parse_args()
    export(args.output, args.case, cluster_only=args.cluster_only)
