"""CPU-only screen of exact INT8 planes with integer affine correction.

Archive #13-22/#126/#148/#154; startup D.4 four questions:
1. This screens the bounded history dequant/QK/PV phase, not a production kernel.
2. The failed 6.5s full-history restoration is not an admissible implementation.
3. The proposed route would keep one KV256 tile and publish INT8 levels; here
   exact dyadic arithmetic tests its semantics before any AscendC implementation.
4. More INT8 planes, INT32 outputs and affine combination are charged explicitly;
   no cycles, NPU accuracy, graph, or whole-model speedup are claimed.

The existing reference and acceptance file are read unchanged. Python int64 /
unbounded integers audit INT32 bounds and exact reconstruction; they are not a
claim that all metadata correction fits one hardware INT32 instruction.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import torch

from oscar_ascend.ops.reference import attention, compressed_attention, dequantize_int2, encode_kv, quantize_int2, unpack_int2

ROOT = Path(__file__).resolve().parents[2]


def dyadic(value: float) -> tuple[int, int]:
    numerator, denominator = float(value).as_integer_ratio()
    if numerator == 0:
        return 0, 0
    trailing = (abs(numerator) & -abs(numerator)).bit_length() - 1
    return numerator >> trailing, trailing - (denominator.bit_length() - 1)


def split_row(values: list[float]) -> tuple[torch.Tensor, int, list[int]]:
    parts = [dyadic(v) for v in values]
    exponent = min((e for n, e in parts if n), default=0)
    integers = [n << (e - exponent) if n else 0 for n, e in parts]
    count = max(1, (max(map(abs, integers)).bit_length() + 6) // 7)
    planes = torch.tensor([
        [(-1 if n < 0 else 1) * ((abs(n) >> (7 * p)) & 127) for n in integers]
        for p in range(count)], dtype=torch.int32)
    for i, n in enumerate(integers):
        assert sum(int(planes[p, i]) << (7 * p) for p in range(count)) == n
        assert math.ldexp(float(n), exponent) == values[i]
    assert int(planes.abs().max()) <= 127
    return planes, exponent, integers


def exact_qk(query: torch.Tensor, encoded) -> tuple[torch.Tensor, dict]:
    codes = unpack_int2(encoded.packed, query.shape[-1]).to(torch.int64)
    meta = []
    for s, z in zip(encoded.scale.flatten().tolist(), encoded.zero.flatten().tolist()):
        sn, se = dyadic(s); zn, ze = dyadic(z)
        exponent = min(se, ze) if zn else se
        meta.append((sn << (se - exponent), zn << (ze - exponent) if zn else 0, exponent))
    outputs, counts, peak = [], [], 0
    for row in query.tolist():
        planes, qe, _ = split_row(row)
        counts.append(len(planes))
        # Every integer MAC is emulated exactly, then checked for INT32 range.
        dots = planes.to(torch.int64) @ codes.T
        sums = planes.to(torch.int64).sum(-1)
        peak = max(peak, int(dots.abs().max()), int(sums.abs().max()))
        result = []
        for j, (s, z, me) in enumerate(meta):
            fixed = []
            for p in range(len(planes)):
                ds, sz = int(dots[p, j]) * s, int(sums[p]) * z
                fixed.append(ds + sz)
                peak = max(peak, abs(ds), abs(sz), abs(ds + sz))
            exact = sum(n << (7 * p) for p, n in enumerate(fixed))
            result.append(math.ldexp(float(exact), qe + me))
        outputs.append(result)
    assert peak <= 2**31 - 1, "this case cannot use the proposed INT32 correction"
    return torch.tensor(outputs, dtype=torch.float32), {
        "signed_base128_planes_per_query": counts,
        "peak_integer_intermediate_abs": peak, "int32_safe_in_this_case": True,
    }


def exact_pv(weights: torch.Tensor, encoded) -> tuple[torch.Tensor, dict]:
    codes = unpack_int2(encoded.packed, encoded.head_dim).to(torch.int64)
    scales, zeros = encoded.scale.flatten().tolist(), encoded.zero.flatten().tolist()
    outputs, counts, peak = [], [], 0
    for row in weights.tolist():
        # FP32 probability x FP16 metadata is exactly representable in FP64.
        planes, se, _ = split_row([p * s for p, s in zip(row, scales)])
        _, ze, zi = split_row([p * z for p, z in zip(row, zeros)])
        dots = planes.to(torch.int64) @ codes
        counts.append(len(planes)); peak = max(peak, int(dots.abs().max()))
        exponent = min(se, ze)
        zero_sum = sum(zi) << (ze - exponent)
        outputs.append([math.ldexp(float(
            sum(int(dots[p, j]) << (7 * p + se - exponent) for p in range(len(planes)))
            + zero_sum), exponent) for j in range(encoded.head_dim)])
    assert peak <= 2**31 - 1
    return torch.tensor(outputs, dtype=torch.float32), {
        "exact_probability_times_scale_planes_per_row": counts,
        "peak_raw_int32_dot_abs": peak,
        "zero_and_plane_combination": "exact Python integer upper-bound semantics; not a proved INT32 implementation",
    }


def hadamard(value: torch.Tensor) -> torch.Tensor:
    x = value.float().clone(); index = torch.arange(256)
    for stage in range(8):
        sign = torch.where((index & (1 << stage)) == 0, 1.0, -1.0)
        x = x * sign + x[:, index ^ (1 << stage)]
    return x * (1.0 / 16.0)


def compare(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, *, qlen: int = 1) -> dict:
    ek, ev = quantize_int2(keys), quantize_int2(values)
    assert torch.equal(dequantize_int2(ek), keys)
    assert torch.equal(dequantize_int2(ev), values)
    rows = len(query)
    shaped_q = query.reshape(qlen, rows // qlen, 256)
    gold = attention(shaped_q, keys[:, None], values[:, None],
                     causal=False).output.reshape(rows, 256)
    identity_gold = compressed_attention(shaped_q, encode_kv(keys[:, None], values[:, None]),
                                        torch.eye(256), torch.eye(256), causal=False).output.reshape(rows, 256)
    assert torch.equal(gold, identity_gold)
    logits, qledger = exact_qk(query, ek)
    logits *= 1.0 / 16.0
    weights = torch.exp(logits - logits.amax(-1, keepdim=True))
    numerator, pledger = exact_pv(weights, ev)
    candidate = numerator / weights.sum(-1, keepdim=True)
    tolerance = json.loads((ROOT / "configs/acceptance.json").read_text())["fused_attention"]
    delta = (candidate - gold).abs()
    allowed = tolerance["atol"] + tolerance["rtol"] * gold.abs()
    worst = int(delta.flatten().argmax())
    failed = delta > allowed
    failing_index = int(torch.where(failed.flatten(), delta.flatten(), -1.0).argmax()) if bool(failed.any()) else None
    return {
        "query_shape": list(shaped_q.shape), "key_shape": list(keys.shape),
        "valid_store_roundtrip": True, "frozen_gate": tolerance,
        "identity_inverse_rotation_original_domain_oracle_checked": True,
        "qk": qledger, "pv": pledger,
        "integer_corrected_logits_all_zero": bool((logits == 0).all()),
        "max_abs_output_difference": float(delta.max()),
        "failed_elements": int((delta > allowed).sum()), "elements": delta.numel(),
        "worst_flat_index": worst, "reference_at_worst": float(gold.flatten()[worst]),
        "candidate_at_worst": float(candidate.flatten()[worst]),
        "worst_failing_element": None if failing_index is None else {
            "flat_index": failing_index, "reference": float(gold.flatten()[failing_index]),
            "candidate": float(candidate.flatten()[failing_index]),
            "abs_difference": float(delta.flatten()[failing_index]),
            "allowed": float(allowed.flatten()[failing_index])},
        "query_sha256": hashlib.sha256(bytes(query.contiguous().view(torch.uint8).flatten().tolist())).hexdigest(),
    }


def run() -> dict:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1753)
    oldq = torch.randn((1, 256), generator=generator); oldq[:, :2] = 0
    oldcodes = torch.ones((2, 256)); oldcodes[:, 0] = 0; oldcodes[:, 1] = 3
    old = compare(oldq, oldcodes * torch.tensor([[16.0], [1.0]]) - torch.tensor([[16.0], [1.0]]),
                  oldcodes * 2048 + torch.tensor([[0.0], [-4096.0]]))
    assert old["failed_elements"] == 0

    generator = torch.Generator().manual_seed(1753)
    rawq = torch.zeros((24, 256), dtype=torch.bfloat16)
    rawq[:, 128:] = torch.randn((24, 128), generator=generator).bfloat16()
    query = hadamard(rawq)
    base = torch.randint(0, 4, (128,), generator=generator).float(); base[0] = 0; base[1] = 3
    codes = torch.cat((base, base)).repeat(2, 1)
    keys = codes * torch.tensor([[128.0], [1.0]]) - torch.tensor([[128.0], [1.0]])
    vc = torch.ones((2, 256)); vc[:, [0, 2]] = 0; vc[:, [1, 3]] = 3
    values = vc * 2048 + torch.tensor([[0.0], [-4096.0]])
    for tensor in (keys, values):
        raw = hadamard(tensor)
        assert torch.equal(raw.bfloat16().float(), raw)
        assert torch.equal(hadamard(raw.bfloat16()), tensor)
    # The first rotated-space candidate is NOT a full-oracle counterexample:
    # the independent CV gate inverse-rotates K/V and consumes the original Q.
    logits, _ = exact_qk(query, quantize_int2(keys))
    assert bool((logits == 0).all())
    rotation = hadamard(torch.eye(256))
    original_gold = compressed_attention(rawq.reshape(4, 6, 256),
        encode_kv(keys[:, None], values[:, None]), rotation, rotation, causal=False).output.reshape(24, 256)
    original_exact = hadamard(values.mean(0, keepdim=True)).expand(24, -1)
    assert torch.equal(original_gold, original_exact)
    rotated_case = {"status": "not_a_full_oracle_counterexample", "original_domain_max_abs": 0.0,
                    "reason": "raw Q and inverse-rotated K have disjoint support; do not reject from rotated-space FP32 reduction alone"}

    # Identity R is an actual supported MTP rotation. Every raw Q/K/V input is
    # BF16, so this case does not mistake a rotated-domain dot for the CV oracle.
    generator = torch.Generator().manual_seed(1753)
    half = torch.randn((24, 128), generator=generator).bfloat16().float()
    exponents = torch.randint(-16, 17, (24, 128), generator=generator)
    half = (torch.ldexp(half, exponents) / 16384.0).bfloat16().float()
    query = torch.cat((half, -half), -1)
    base = torch.randint(0, 4, (128,), generator=generator).float(); base[0] = 0; base[1] = 3
    codes = torch.cat((base, base)).repeat(2, 1)
    keys = (codes - 1) * torch.tensor([[16.0], [1.0]])
    for tensor in (query, keys, values):
        assert torch.equal(tensor.bfloat16().float(), tensor)
    new = compare(query, keys, values, qlen=4)
    new.update({"rotation": "identity", "all_raw_qkv_bf16_exact": True,
                "query_norm": float(query.norm()), "query_abs_max": float(query.abs().max()),
                "input_generator": "BF16 randn * 2^randint(-16,17) / 16384, followed by exact negative half"})
    assert new["failed_elements"] > 0 and new["integer_corrected_logits_all_zero"]

    # One declared finite input for plane/space accounting; not a performance test.
    generator = torch.Generator().manual_seed(1753)
    normalq = hadamard(torch.randn((24, 256), generator=generator).bfloat16())
    logit = torch.randn((24, 256), generator=generator) * 4.0
    p = torch.exp(logit - logit.amax(-1, keepdim=True))
    vs = quantize_int2(hadamard(torch.randn((256, 256), generator=generator).bfloat16())).scale.flatten().tolist()
    qplanes = [len(split_row(row)[0]) for row in normalq.tolist()]
    pplanes = [len(split_row([v * s for v, s in zip(row, vs)])[0]) for row in p.tolist()]
    ledger = []
    for name, planes in (("qk", max(qplanes)), ("pv", max(pplanes))):
        ledger.append({"operation": name, "max_planes": planes, "stacked_M": 24 * planes,
                       "macs": 24 * planes * 256 * 256,
                       "int8_A_bytes": 24 * planes * 256, "int8_B_bytes": 256 * 256,
                       "int32_C_bytes": 24 * planes * 256 * 4})
    f32_bytes = 2 * 2 * (24 * 256 * 4 + 256 * 256 * 4 + 24 * 256 * 4)
    int_bytes = 2 * sum(row["int8_A_bytes"] + row["int8_B_bytes"] + row["int32_C_bytes"] for row in ledger)
    return {
        "status": "rejected_against_frozen_cpu_oracle", "seed": 1753,
        "environment": {"torch": str(torch.__version__), "machine": platform.machine(),
                        "system": platform.system(), "threads": torch.get_num_threads()},
        "scope": "CPU mathematics only; old A2 kernel on this new input unmeasured; no NPU regression claim",
        "decision": "Do not implement this unqualified exact-reassociation route as a frozen-oracle replacement. This is not a proof against every INT8 algorithm, nor a measured A2 regression; old A2 has not run this case.",
        "old_affine_case_repaired": old, "hadamard_rotated_case_not_admissible_for_rejection": rotated_case,
        "bf16_reachable_counterexample": new,
        "plane_accounting": {"scope": "one synthetic M24/K256/D256, BF16 Hadamard Q and exp(normal*4) P with valid V scales; not a universal bound",
                             "q_planes_per_row": qplanes, "pv_planes_per_row": pplanes, "ledger": ledger,
                             "logical_GM_publish_read_bytes": {"fp32_qk_plus_pv": f32_bytes,
                                 "int8_qk_plus_pv": int_bytes,
                                 "reduction_fraction": 1.0 - int_bytes / f32_bytes,
                                 "scope": "each A/B published+read and each C written+read once, without padding; not measured HBM traffic or speed"},
                             "excluded_costs": ["plane extraction/layout", "per-KV probability x scale", "zero reduction", "INT32 conversion/combination", "softmax/window/Rv", "padding and SDK tiling"]},
        "source_sha256": {str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
            "tools/experiments/int8_planes_counterexample.py", "oscar_ascend/ops/reference.py",
            "configs/acceptance.json", "csrc/kernels/rotate_clip_store.cpp")},
    }


if __name__ == "__main__":
    report = run()
    target = ROOT / "reports/int8_planes_counterexample_20261011.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(target),
                      "old_failed": report["old_affine_case_repaired"]["failed_elements"],
                      "new_max_abs": report["bf16_reachable_counterexample"]["max_abs_output_difference"],
                      "new_failed": report["bf16_reachable_counterexample"]["failed_elements"],
                      "planes": report["plane_accounting"]["ledger"]}, sort_keys=True))
