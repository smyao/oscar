"""Archive G26-G34/#13-22/#37-49/#69/#126-128/#145/#154/#155; D.4.

Independent PR quantizer goldens and logical request-ring dictionary oracle.
No device work. Byte equality includes BF16 poison/NaN payload and signed zero.
D.4 phase1_stores208.6-216.0ms/16K sets a future device target, not CPU timing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from oscar_ascend.ops import reference
from tools.experiments.active_identity_store_fixture import _payload, _side, _striped

SHAPE_FIELDS = ("tokens", "heads", "dim", "dtype", "segment_tokens", "blocks", "offset",
                "page_stride", "requests", "capacity", "key_stride", "value_stride",
                "tag_stride", "sink", "recent_with_spec", "position_bits", "key_clip", "value_clip", "hadamard")


def packed_address(slot: int, head: int, *, segment: int, blocks: int, offset: int,
                   stride: int, heads: int, dim: int) -> int:
    page, in_page = divmod(slot, 2 * segment)
    region, within = divmod(in_page, segment)
    return offset + (region * blocks + page) * stride + (within * heads + head) * (dim // 2 + 8)


def logical_winners(qstarts, mapping, slots, positions, status, capacity, sink=64,
                    recent=259, bits=18):
    """Independent dictionary of last successful logical positions per owner.

    One linear pass per request; counted visits make quadratic scans visible.
    A request with invalid metadata publishes nothing, never truncated epochs.
    """
    result, errors, visits = {}, [], 0
    reqs, n = len(qstarts) - 1, len(slots)
    if qstarts[0]!=0:return {},[6]*reqs,0
    active_tail=0<=qstarts[-1]<=n and any(slot>=0 for slot in slots[qstarts[-1]:])
    for req, (begin, end) in enumerate(zip(qstarts, qstarts[1:])):
        error, chosen = 0, {}
        if not 0 <= begin <= end <= n:
            errors.append(1); continue
        if req==0 and active_tail:
            errors.append(7);continue
        if begin == end:
            errors.append(0); continue
        previous = -1
        for token in range(begin, end):
            visits += 1
            if slots[token] < 0 or any(status[token]):
                continue
            pos = positions[token]
            if not 0 <= pos < 1 << bits or pos <= previous:
                error = 4; break
            previous = pos
            ring = pos if pos < sink else sink + pos % recent
            chosen[ring] = token
        if not error and chosen:
            lease, epoch = mapping[req]
            if not 0 <= lease < capacity or not 0 < epoch < 1 << (63 - bits):error=2
            elif any(other!=req and qstarts[other]<qstarts[other+1] and mapping[other][0]==lease for other in range(reqs)):error=3
        if not error:
            result[req] = chosen
        errors.append(error)
    return result, errors, visits


def build_case(name: str, *, lengths=(7,), dim=64, heads=1, dtype_name="bf16",
               segment=128, in_page=125, mode="normal", hadamard=False,
               token_capacity=None, window_capacity=None, padding_requests_after=None):
    live, requests = sum(lengths), len(lengths)
    n, capacity = token_capacity or live, window_capacity or max(4, len(lengths))
    if n<live or capacity<requests:raise ValueError("fixture capacity below active shape")
    dtype_code, dtype = {"fp32": (0, torch.float32), "fp16": (1, torch.float16), "bf16": (2, torch.bfloat16)}[dtype_name]
    rng = torch.Generator().manual_seed(1009 + n * 11 + dim + heads)
    key = torch.randn(n, heads, dim, generator=rng).to(dtype)
    value = torch.randn(n, heads, dim, generator=rng).to(dtype)
    key[..., 0] = -0.0; value[..., 0] = 0.0
    qstarts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    slots, positions = torch.full((n,),-1,dtype=torch.int64),torch.full((n,),-1,dtype=torch.int64)
    key[live:]=torch.nan;value[live:]=torch.inf
    mapping = torch.full((capacity, 2), -1, dtype=torch.int64)
    for req, (begin, end) in enumerate(zip(qstarts.tolist(), qstarts.tolist()[1:])):
        page = requests - req + 1
        slots[begin:end] = page * 2 * segment + in_page + torch.arange(end - begin)
        positions[begin:end] = (0 if mode in ("rollback", "epoch_reuse") else 511 * req + in_page) + torch.arange(end - begin)
        mapping[req] = torch.tensor([requests - 1 - req, 7 + req])
    blocks = max(4, int(slots.max()) // (2 * segment) + 1)
    if padding_requests_after is not None:
        first_padding=int(qstarts[padding_requests_after])
        slots[first_padding:]=-1;positions[first_padding:]=-1;key[first_padding:]=torch.nan;value[first_padding:]=torch.inf
        mapping[padding_requests_after:,0]=-1;mapping[padding_requests_after:,1]=0
    if mode == "holes":
        slots[1::5] = -1; positions[1::5] = -1; key[1::5] = torch.nan; value[1::5] = torch.inf
    elif mode in ("all_padding","capture_all_padding"):
        slots.fill_(-1); positions.fill_(-1); key.fill_(torch.nan); value.fill_(torch.inf)
        if mode=="capture_all_padding":mapping[:,0]=-1;mapping[:,1]=0
    elif mode == "faults":
        key[0, 0, 1] = torch.nan
        value[1, -1, 2] = torch.inf
        key[2] = 0.0
        slots[3] = 2 * segment * blocks
        positions[4] = -1
        slots[5] = -1; positions[5] = -1; key[5] = torch.nan
    elif mode == "bad_epoch":
        mapping[0, 1] = 1 << 45
    elif mode == "duplicate_lease":
        mapping[1, 0] = mapping[0, 0]
    elif mode == "bad_position":
        positions[-1] = 1 << 18
    elif mode == "bad_qstarts":
        qstarts[-1] = n + 1
    elif mode == "positive_qstarts_origin":
        qstarts[0] = 1
    elif mode == "active_tail":
        if n<=live:raise ValueError("active tail fixture needs padding")
        slots[live]=0;positions[live]=0
    elif mode not in ("normal", "clip", "rollback", "epoch_reuse"):
        raise ValueError(mode)
    rotation = torch.eye(dim, dtype=torch.float32)
    if hadamard:
        rotation = torch.ones(1, 1)
        while len(rotation) < dim:
            rotation = torch.cat((torch.cat((rotation, rotation), 1), torch.cat((rotation, -rotation), 1)), 0)
        rotation *= dim ** -0.5
    kc, vc = (0.75, 0.95) if mode == "clip" else (0.0, 0.0)
    sink, recent, bits, offset = 64, 259, 18, 13
    slot_bytes = dim // 2 + 8
    stride = segment * heads * slot_bytes + 17
    ks, vs, ts = 323 * heads * dim + 19, 323 * heads * dim + 35, 326
    packed = torch.full((offset + 2 * blocks * stride,), 0xAD, dtype=torch.uint8)
    wk = torch.tensor([0xC5, 0x7F], dtype=torch.uint8).repeat(capacity * ks)
    wv = torch.tensor([0xC7, 0x7F], dtype=torch.uint8).repeat(capacity * vs)
    tags = torch.full((capacity * ts * 8,), 0xAD, dtype=torch.uint8)
    if mode == "epoch_reuse":
        # A former owner has valid-looking low positions, but the old epoch.
        for ring in range(323):
            old = (6 << bits) | ring
            tags[ring * 8:ring * 8 + 8] = torch.tensor([old], dtype=torch.int64).view(torch.uint8)
    initial = {"packed": packed.clone(), "raw_key": wk.clone(), "raw_value": wv.clone(), "tags": tags.clone(),
               "status": torch.full((n, heads), 0x55555555, dtype=torch.int32),
               "window_status": torch.full((capacity,), 0x55555555, dtype=torch.int32)}
    status = torch.zeros((n, heads), dtype=torch.int32)
    baseline_slots = slots.clone()
    written_addresses = []
    for token in range(n):
        slot, pos = int(slots[token]), int(positions[token])
        if slot < 0: continue
        if slot >= blocks * 2 * segment:
            status[token] = 1; continue
        page, ip = divmod(slot, 2 * segment)
        region, within = divmod(ip, segment)
        baseline_slots[token] = (region * blocks + page) * segment + within
        if pos < 0:
            status[token] = 4; continue
        for head in range(heads):
            ke, k = _side(key[token, head], rotation, kc)
            ve, v = _side(value[token, head], rotation, vc)
            status[token, head] = ke or ve
            if ke or ve: continue
            encoded = _striped(reference.encode_kv(k, v), dim)
            address = packed_address(slot, head, segment=segment, blocks=blocks, offset=offset, stride=stride, heads=heads, dim=dim)
            packed[address:address + slot_bytes] = encoded
            written_addresses.append([address, address + slot_bytes])
    winners, errors, visits = logical_winners(qstarts.tolist(), mapping.tolist(), slots.tolist(), positions.tolist(), status.tolist(), capacity)
    for req, chosen in winners.items():
        lease, epoch = mapping[req].tolist()
        for ring, token in chosen.items():
            for source, target, ws in ((key, wk, ks), (value, wv, vs)):
                start = (lease * ws + ring * heads * dim) * 2
                target[start:start + heads * dim * 2] = source[token].to(torch.bfloat16).contiguous().view(torch.uint8).flatten()
            start = (lease * ts + ring) * 8
            tags[start:start + 8] = torch.tensor([(epoch << bits) | int(positions[token])], dtype=torch.int64).view(torch.uint8)
    shape = (n, heads, dim, dtype_code, segment, blocks, offset, stride, requests, capacity, ks, vs, ts, sink, recent, bits, kc, vc, int(hadamard))
    values = {"key": key, "value": value, "rk": rotation.T, "rv": rotation.T,
              "slots": slots, "baseline_slots": baseline_slots, "positions": positions,
              "qstarts": qstarts, "mapping": mapping[:requests].clone(), "expected_packed": packed,
              "expected_status": status, "expected_window_status": torch.tensor(errors+[0]*(capacity-requests), dtype=torch.int32),
              "expected_raw_key": wk, "expected_raw_value": wv, "expected_tags": tags,
              **{"initial_" + k: v for k, v in initial.items()}}
    metadata = {"name": name, "shape_fields": dict(zip(SHAPE_FIELDS, shape)), "mode": mode,
                "status_histogram": dict(Counter(status.flatten().tolist())), "window_status": errors,
                "scan_visits": visits, "winners": winners, "packed_written_intervals": written_addresses,
                "oracle": "Independent PR FP32 rotate/clip/FP16-metadata quantizer + striped permutation; logical-position/epoch dictionary for sidecar",
                "packing_baseline": "Original striped kernel with slots remapped to region*blocks+page, two times page count and half physical blockTokens",
                "CANN": "not_run", "CPU_debug": "not_run", "NPU": "not_run", "graph": "not_run"}
    return values, shape, metadata


def specs():
    return [
        dict(name="small_bf16", lengths=(7,)),
        dict(name="nonaligned_segment_heads2", lengths=(19,), segment=129, heads=2, in_page=126),
        dict(name="physical_page_cross_fp16", lengths=(19,), dtype_name="fp16", in_page=249),
        dict(name="reordered_requests", lengths=(4, 7, 3), heads=2),
        dict(name="negative_holes", lengths=(333,), mode="holes"),
        dict(name="rollback_0_to_3", lengths=(700,), mode="rollback", in_page=0),
        dict(name="epoch_reuse", lengths=(7,), mode="epoch_reuse", in_page=0),
        dict(name="all_padding_poison", lengths=(9,), mode="all_padding"),
        dict(name="numeric_and_address_faults", lengths=(17,), heads=2, mode="faults"),
        dict(name="clipped_hadamard_d256", lengths=(9,), dim=256, mode="clip", hadamard=True),
        dict(name="q1_n128_32live_hkv2",lengths=(1,)*32,heads=2,token_capacity=128,window_capacity=128),
        dict(name="bad_epoch", mode="bad_epoch"),
        dict(name="duplicate_lease", lengths=(3, 4), mode="duplicate_lease"),
        dict(name="bad_position_tag_domain", mode="bad_position"),
        dict(name="bad_qstarts", mode="bad_qstarts"),
        dict(name="positive_qstarts_origin",mode="positive_qstarts_origin"),
        dict(name="active_tail",mode="active_tail",token_capacity=12),
        dict(name="capture_all_padding_unpublished_map",lengths=(9,),mode="capture_all_padding"),
        dict(name="q1_n128_128rows_32live_unpublished_tail",lengths=(1,)*128,heads=2,padding_requests_after=32,window_capacity=128),
    ]


def generate(directory: Path):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for spec in specs():
        values, shape, metadata = build_case(**spec)
        path = directory / metadata["name"]; path.mkdir(exist_ok=True)
        files = {}
        for key, value in values.items():
            payload = _payload(value); (path / (key + ".bin")).write_bytes(payload)
            files[key + ".bin"] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        (path / "shape.txt").write_text(" ".join(map(str, shape)) + "\n")
        metadata.update(path=str(path), files=files)
        (path / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases.append(metadata)
    manifest = {"status": "host_fixtures_generated_not_device_verified", "cases": cases,
                "shape_fields": SHAPE_FIELDS, "case_count": len(cases),
                "reference_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest()}
    (directory / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.output)
    print(json.dumps({"status": result["status"], "cases": result["case_count"], "manifest": str(args.output / "cases.json")}))
