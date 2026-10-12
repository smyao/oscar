"""Archive #13-22/#126/#140/#145/#148/#154-156 and full startup D.4.

Independent host-only FP32 Matmul fixtures, never a production path. Preserve
PR INT2 codes and FP16-first scale/zero rounding through ops.reference. The
three arms consume exactly the same A/B bytes; no arm supplies the oracle.
D.4: this isolates bounded QK/PV/Rv inside fused FIA, not historical 6.5-second
full-cache recovery. No history materialization enters serving, no numerical
or performance gate is inferred from fixture generation, and the frozen
5e-3 fused-attention tolerances remain unchanged. Changing K128 to K64 may
change accumulation; live old/new bitwise equality is the first experiment
rejection gate, followed by each arm against the independent FP32 oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from oscar_ascend.ops import reference

D = 256
PADDED = 32
SEED = 202610103264
SOURCE_PINS = {
    "csrc/kernels/attention_cv_decode_bundle.cpp":
        "d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401",
    "oscar_ascend/ops/reference.py":
        "8fa1d3a7912073d0e37fd3189a2467ea0d8e4ddeb4dfb7ea1cec3e6add43c20c",
    "configs/acceptance.json":
        "b044a5f485cd8cfed81dc76c7199717952853167b4e1d8e0aa9bff499b290811",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw(tensor: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise ValueError("fixture format requires a little-endian host")
    return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())


def check_sources() -> dict:
    for name, wanted in SOURCE_PINS.items():
        if sha((ROOT / name).read_bytes()) != wanted:
            raise ValueError("re-audit changed fixture source: " + name)
    return {
        **SOURCE_PINS,
        "references/oscar-vllm-pr46774/vllm/v1/attention/ops/triton_oscar_store.py":
            sha((ROOT / "references/oscar-vllm-pr46774/vllm/v1/attention/ops/triton_oscar_store.py").read_bytes()),
        str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).read_bytes()),
        "csrc/experiments/decode_mmad_barrier/kernel.cpp":
            sha((ROOT / "csrc/experiments/decode_mmad_barrier/kernel.cpp").read_bytes()),
        "csrc/kernels/oscar_common.h": sha((ROOT / "csrc/kernels/oscar_common.h").read_bytes()),
    }


def hadamard() -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < D:
        matrix = torch.cat((torch.cat((matrix, matrix), 1),
                            torch.cat((matrix, -matrix), 1)), 0)
    return matrix / 16.0


def quantized_evidence(name: str, value: torch.Tensor) -> tuple[torch.Tensor, dict]:
    vector = reference.quantize_int2(value)
    restored = reference.dequantize_int2(vector)
    codes = reference.unpack_int2(vector.packed, D)
    # The exact stored half bit patterns, scale then zero, low byte first.
    bits = torch.cat((vector.scale, vector.zero), -1).contiguous().view(torch.int16)
    unsigned = bits.to(torch.int32) & 65535
    metadata = torch.stack((unsigned & 255, unsigned >> 8), -1).to(torch.uint8).reshape(D, 4)
    return restored, {
        name + "_rotated": value,
        name + "_codes": codes,
        name + "_packed": vector.packed,
        name + "_scale_f16": vector.scale,
        name + "_zero_f16": vector.zero,
        name + "_metadata_le": metadata,
        name + "_restored": restored,
    }


def write_case(output: Path, name: str, live: int, layout: int,
               live_a: torch.Tensor, b: torch.Tensor, evidence: dict,
               notes: dict, sources: dict) -> dict:
    path = output / name
    path.mkdir()
    a = torch.zeros((PADDED, D), dtype=torch.float32)
    a[:live] = live_a
    b = b.to(torch.float32).contiguous()
    if live_a.shape != (live, D) or b.shape != (D, D):
        raise ValueError("incorrect Matmul fixture shape")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("these eight cases have a finite-input contract")
    if torch.count_nonzero(a[live:].view(torch.int32)):
        raise ValueError("inactive A rows must contain positive-zero bits")
    expected = torch.zeros((PADDED, D), dtype=torch.float32)
    operand = b if layout == 1 else b.T
    expected[:live] = torch.mm(a[:live], operand)
    if not torch.isfinite(expected).all():
        raise ValueError("FP32 oracle unexpectedly became nonfinite")
    tensors = {"a": a, "b": b, "expected": expected, **evidence}
    if notes["family"] == "cancellation":
        # Diagnostic only: do not replace the requested FP32 acceptance oracle
        # with a different dtype or pretend it specifies Cube's reduction tree.
        exact = torch.zeros((PADDED, D), dtype=torch.float64)
        exact[:live] = torch.mm(a[:live].double(), operand.double())
        tensors["expected_f64_diagnostic"] = exact
    files = {}
    for key, tensor in tensors.items():
        data = raw(tensor)
        filename = key + ".bin"
        (path / filename).write_bytes(data)
        files[filename] = {"bytes": len(data), "sha256": sha(data),
                           "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    shape = f"{live} {layout}\n".encode()
    (path / "shape.txt").write_bytes(shape)
    files["shape.txt"] = {"bytes": len(shape), "sha256": sha(shape)}
    tolerance = json.loads((ROOT / "configs/acceptance.json").read_text())["fused_attention"]
    row = {
        "case": name, "live_rows": live, "layout": layout,
        "shape": {"a": [PADDED, D], "b": [D, D], "expected": [PADDED, D]},
        "seed": SEED, "torch_version": str(torch.__version__), "torch_threads": 1,
        "source_sha256": sources, "files": files, "math": notes,
        "a_tail": "explicit FP32 positive-zero bytes",
        "oracle": "CPU Torch FP32 mm; transpose B for layout0/2, direct B for layout1",
        "oracle_scope": "only rows [0,live_rows) x 256; finite tail zeros are not old-arm output requirements",
        "paired_gate": "live output bits must match old arm before tolerance checks; no tolerance widening",
        "independent_oracle_tolerance": tolerance,
        "initial_output_and_guards": "driver-owned 0xAD output and front/back guards; fixture does not preinitialize kernel output",
        "cpu_debug": "not_run", "cann": "not_run", "camodel": "not_run",
        "npu": "not_run", "graph": "not_run", "whole_decode": "not_run",
    }
    (path / "fixture.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def export(output: Path) -> dict:
    sources = check_sources()
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    rotation = hadamard()
    q = (torch.randn((24, D), generator=generator) * 0.25).to(torch.bfloat16)
    key = (torch.randn((D, D), generator=generator) * 0.25).to(torch.bfloat16)
    value = (torch.randn((D, D), generator=generator) * 0.5).to(torch.bfloat16)
    qr = reference.rotate_clip(q, rotation, 0.0)
    k, k_evidence = quantized_evidence("key", reference.rotate_clip(key, rotation, 0.0))
    v, v_evidence = quantized_evidence("value", reference.rotate_clip(value, rotation, 0.0))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for live in (18, 24):
        scores = torch.mm(qr[:live], k.T)
        p = torch.softmax(scores * (D ** -0.5), dim=-1)
        pv = torch.mm(p, v)
        evidence = {"query_bf16": q[:live], "key_bf16": key, "value_bf16": value,
                    "rotation": rotation, "qrot": qr[:live], "qk_unscaled": scores,
                    "normalized_p": p, "normal_pv": pv, **k_evidence, **v_evidence}
        for label, layout, a, b in (("qk", 0, qr[:live], k),
                                     ("pv", 1, p, v), ("rv", 2, pv, rotation)):
            rows.append(write_case(output, f"normal_{label}_m{live}", live, layout, a, b,
                evidence, {"family": "normal", "operation": label,
                           "chain": "BF16 inputs -> FP32 normalized Hadamard -> pinned PR INT2/half metadata -> FP32 reconstruction",
                           "clip_ratio": 0.0, "qk_scale_for_p_only": D ** -0.5,
                           "p": "normalized softmax of independent QK; no candidate output used",
                           "rv": "same exact +/-1/16 normalized Hadamard, multiplied transposed"}, sources))

    # Four 64-wide K groups. Each row places large cancelling products and
    # small residuals in corresponding positions across all four groups.
    # Groups at 0/64/128/192 and 63/127/191/255 exercise both boundary sides.
    families = ((0, 64, 128, 192), (31, 95, 159, 223), (63, 127, 191, 255))
    key_codes = torch.ones((D, D), dtype=torch.float32)
    key_codes[:, 2] = 0.0
    key_codes[:, 3] = 3.0
    for j in range(D):
        for positions in families:
            key_codes[j, positions[0]] = key_codes[j, positions[2]] = float(1 + j % 3)
            key_codes[j, positions[1]] = float((j // 3) % 4)
            key_codes[j, positions[3]] = float((j // 12) % 4)
    cancel_key, cancel_evidence = quantized_evidence("key", key_codes)
    if not torch.equal(cancel_key, key_codes):
        raise ValueError("cancellation K must be exact legal INT2 with scale1/zero0")
    for live in (18, 24):
        a = torch.zeros((live, D), dtype=torch.float32)
        for row in range(live):
            indices = families[row % len(families)]
            magnitude = float(2 ** (20 + row % 6))
            residual = float(2 ** (-(row % 4)))
            sign = -1.0 if (row // 3) % 2 else 1.0
            a[row, list(indices)] = torch.tensor(
                [sign * magnitude, sign * residual, -sign * magnitude, -sign * residual],
                dtype=torch.float32)
        rows.append(write_case(output, f"cancel_qk_m{live}", live, 0, a, cancel_key,
            cancel_evidence, {"family": "cancellation", "operation": "qk",
                              "k_positions": [list(x) for x in families],
                              "terms": "+large,+small,-large,-small with row-dependent signs and powers of two",
                              "large_exponents": [20, 21, 22, 23, 24, 25],
                              "key_domain": "independent quantizer recreates exact codes0..3, FP16 scale1/zero0",
                              "scope": "finite arithmetic falsifier; not a prediction that either SDK reduction will fail",
                              "float64_diagnostic": "separate exact-product diagnostic, not the acceptance oracle"}, sources))
    report = {"schema": 1, "status": "inputs_exported_not_executed", "cases": rows,
              "case_count": len(rows), "source_sha256": sources,
              "scope": "eight isolated FP32 Matmul fixtures; not complete attention or service qualification"}
    (output / "cases.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps({"status": result["status"], "cases": [x["case"] for x in result["cases"]]}))
