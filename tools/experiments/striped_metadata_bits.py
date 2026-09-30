"""Exhaust FP16 metadata predicates for an isolated striped CV experiment.

This checks the integer predicate against IEEE FP16 scalar semantics. It does
not execute CANN Compares, establish packed-mask lane order, or certify NPU.
Archive #126/#145/#154; startup D.4 rules out full-history restoration.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path


def signed16(bits: int) -> int:
    return bits if bits < 0x8000 else bits - 0x10000


def simd_predicates(bits: int) -> tuple[bool, bool]:
    signed = signed16(bits)
    scale_valid = signed > 0 and signed < 0x7C00
    zero_valid = signed < -1024 or (signed >= 0 and signed < 0x7C00)
    return scale_valid, zero_valid


def scalar_predicates(bits: int) -> tuple[bool, bool]:
    value = struct.unpack("<e", struct.pack("<H", bits))[0]
    return math.isfinite(value) and value > 0.0, math.isfinite(value)


def main() -> int:
    mismatches: list[dict[str, object]] = []
    categories = {"positive_subnormal": 0, "negative_zero": 0,
                  "positive_infinity": 0, "negative_nan": 0}
    for bits in range(1 << 16):
        if simd_predicates(bits) != scalar_predicates(bits):
            mismatches.append({"bits": bits, "simd": simd_predicates(bits),
                               "scalar": scalar_predicates(bits)})
        if 0 < bits < 0x400:
            categories["positive_subnormal"] += 1
        if bits == 0x8000:
            categories["negative_zero"] += 1
        if bits == 0x7C00:
            categories["positive_infinity"] += 1
        if 0xFC01 <= bits <= 0xFFFF:
            categories["negative_nan"] += 1
    report = {"experiment": "striped_metadata_signed_bits",
              "scope": "host_ieee_fp16_math_only",
              "cases": 65536, "mismatch_count": len(mismatches),
              "example_mismatches": mismatches[:4],
              "categories": categories,
              "cann_simd_compiled": False, "device_verified": False,
              "status": "passed" if not mismatches else "failed"}
    path = Path(__file__).resolve().parents[2] / "reports/striped_metadata_bits_20260930.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "cases": 65536,
                      "mismatch_count": len(mismatches),
                      "scope": report["scope"]}, sort_keys=True))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
