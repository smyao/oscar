"""Official CPU-debug goldens for the isolated B512/B1024 FP32 wide kernel.

Archive #126/#145/#148-155 and startup D.4: bounded test fixtures use the
normal BF16->rotate_clip->INT2/FP16-metadata store and exact FP32 decode.
The independent original-space oracle is test-only; no full-history recovery,
production route, or device-speed assertion is introduced.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path

import torch

from oscar_ascend.ops import reference
from tools.probe_history_reuse import _hadamard

ROWS = 126
DIM = 256
ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, data: torch.Tensor, kind: str) -> None:
    flat = data.contiguous().flatten().tolist()
    path.write_bytes(array(kind, flat).tobytes())


def write_case(directory: Path, kv: int, mode: str) -> dict:
    if kv not in (512, 1024) or mode not in ("normal", "cancel_high_value"):
        raise ValueError("isolated wide fixture admits B512/B1024 and two modes")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(8192 + kv)
    rk = _hadamard(torch, DIM)
    rv = rk.flip(1).contiguous()
    if mode == "normal":
        raw_q = (torch.randn((ROWS, 1, DIM), generator=generator) * 0.5).to(torch.bfloat16)
        raw_k = torch.randn((kv, 1, DIM), generator=generator).to(torch.bfloat16)
        raw_v = torch.randn((kv, 1, DIM), generator=generator).to(torch.bfloat16)
        visible = torch.full((ROWS,), kv, dtype=torch.int32)
        visible[:5] = torch.tensor([0, 1, kv // 4, kv // 2, kv - 1], dtype=torch.int32)
    else:
        base_k = torch.randn((1, 1, DIM), generator=generator).to(torch.bfloat16)
        raw_k = base_k.expand(kv, -1, -1).clone()
        base_v = torch.zeros((1, 1, DIM), dtype=torch.bfloat16)
        base_v[..., 0] = 2048
        base_v[..., 1] = -2048
        raw_v = base_v.expand(kv, -1, -1).clone()
        raw_v[1::2] = -raw_v[1::2]
        raw_q = torch.randn((ROWS, 1, DIM), generator=generator).to(torch.bfloat16)
        visible = torch.full((ROWS,), kv, dtype=torch.int32)
        visible[0] = 0
    slots = reference.encode_kv(reference.rotate_clip(raw_k.float(), rk),
                                reference.rotate_clip(raw_v.float(), rv))
    rotated_k, rotated_v = reference.decode_kv(slots, DIM)
    qrot = raw_q.float() @ rk
    expected = reference.compressed_attention(
        raw_q, slots, rk, rv, scale=DIM ** -0.5, causal=True,
        query_positions=visible.to(torch.int64) - 1)
    assert bool(torch.isfinite(qrot).all() and torch.isfinite(rotated_k).all()
                and torch.isfinite(rotated_v).all())
    assert bool(torch.isfinite(expected.output).all())
    # The prototype has no producer metadata/status input or private task tail.
    # Every one of the B K/V rows must be valid; guard arithmetic stays well
    # inside FP32 range. This is narrower than the production error domain.
    assert float(qrot.abs().max()) < 16
    assert float(rotated_k.abs().max()) < 16
    assert float(rotated_v.abs().max()) < 4096
    tolerance = json.loads((ROOT / "configs/acceptance.json").read_text())["fused_attention"]
    (directory / "tolerance.txt").write_text(
        f"{tolerance['atol']:.10g} {tolerance['rtol']:.10g}\n")
    _write(directory / "stored_int2.bin", slots, "B")
    for name, tensor, kind in (
        ("qrot.bin", qrot, "f"), ("krot.bin", rotated_k, "f"),
        ("vrot.bin", rotated_v, "f"), ("rv.bin", rv, "f"),
        ("visible.bin", visible, "i"),
        ("expected_output.bin", expected.output, "f"),
        ("expected_lse.bin", expected.lse, "f")):
        _write(directory / name, tensor, kind)
    (directory / "shape.txt").write_text(f"{ROWS} {kv} {DIM} {DIM ** -0.5:.10g}\n")
    return {"path": str(directory), "kv": kv, "mode": mode,
            "rows": ROWS, "dim": DIM, "stored_int2_slots": slots.numel(),
            "max_abs_raw_v": float(raw_v.float().abs().max()),
            "max_abs_decoded_v": float(rotated_v.abs().max()),
            "max_abs_expected_output": float(expected.output.abs().max()),
            "max_abs_qrot": float(qrot.abs().max()),
            "max_abs_decoded_k": float(rotated_k.abs().max()),
            "all_masked_rows": int((visible == 0).sum()),
            "frozen_tolerance": tolerance,
            "stored_int2_sha256": hashlib.sha256((directory / "stored_int2.bin").read_bytes()).hexdigest(),
            "oracle": "reference.compressed_attention original-space FP32"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/wide_fp32_cpu_cases"))
    args = parser.parse_args()
    cases = [write_case(args.output_dir / f"b{kv}_{mode}", kv, mode)
             for kv in (512, 1024) for mode in ("normal", "cancel_high_value")]
    manifest = {"status": "generated_cpu_only", "cases": cases,
                "domain": "M126/D256/grid1; all B K/V rows finite and store-reachable; qrot<16,krot<16,vrot<4096",
                "old_error_domain_equivalence": "not_established",
                "npu_precision": "not_run", "device_speed": "not_run"}
    (args.output_dir / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("WIDE_FP32_FIXTURE " + json.dumps({"cases": len(cases),
          "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
