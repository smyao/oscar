"""Archive G26/G27/#13-22/#126/#145/#154-156 and full startup D.4.

Host-only independent row-major FP32 Hadamard oracle for query rotation.
The stage order, Gather-before-Mul-before-Add, FP32 rounding and final scalar
normalization mirror the declared mathematics, never a candidate's output.
D.4: query rotation is bounded new-query work; no historical reconstruction
or extra production path is introduced. Frozen rotation/store 2e-3 is used
only for the independent finite oracle. All three arms must additionally
match complete output/status bytes, including NaN payloads and signed zero.
CPU, CANN, NPU, graph and performance qualification are not fixture generation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
SEED = 20261010864
SOURCE_PINS = {
    "csrc/kernels/rotate_clip_store.cpp":
        "e18bfc08673b3f5a49d24029c9ad34797e0d5b9114e79639ac01310658982748",
    "csrc/kernels/oscar_common.h":
        "20574a83f919df5bd6450cd581d1dba5a5a6f169768e7e237ba62f51b7499964",
    "configs/acceptance.json":
        "b044a5f485cd8cfed81dc76c7199717952853167b4e1d8e0aa9bff499b290811",
}
PLAN = (
    ("n128_h6_d256_bf16", 768, 256, 2, 6, "normal"),
    ("r1_h1_d64_bf16_signed_zero", 1, 64, 2, 1, "signed_zero"),
    ("r7_h1_d64_fp16", 7, 64, 1, 1, "normal"),
    ("r8_h1_d128_bf16", 8, 128, 2, 1, "normal"),
    ("r9_h3_d128_fp16_partial", 9, 128, 1, 3, "partial"),
    ("r7_h1_d256_bf16_nonfinite", 7, 256, 2, 1, "nonfinite"),
    ("r8_h1_d256_fp16_nonfinite", 8, 256, 1, 1, "nonfinite"),
    ("r9_h3_d64_bf16_allpadding", 9, 64, 2, 3, "allpadding"),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_source():
    for name, wanted in SOURCE_PINS.items():
        if sha((ROOT / name).read_bytes()) != wanted:
            raise ValueError("re-audit changed rotation source: " + name)
    return {**SOURCE_PINS,
            str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).read_bytes())}


def raw(tensor):
    if sys.byteorder != "little":
        raise ValueError("fixture bytes require a little-endian host")
    return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())


def signed16(word):
    return word if word < 32768 else word - 65536


def rotation_matrix(dim):
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < dim:
        matrix = torch.cat((torch.cat((matrix, matrix), 1),
                            torch.cat((matrix, -matrix), 1)), 0)
    # The exact FP32 constant stored at matrix[0,0] is also used by the oracle.
    normalization = torch.tensor(dim ** -0.5, dtype=torch.float32)
    return matrix * normalization


def inputs(rows, dim, dtype, heads, mode, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kind = torch.float16 if dtype == 1 else torch.bfloat16
    source = (torch.randn((rows, dim), generator=generator) * 0.25).to(kind)
    slots = torch.arange(rows // heads, dtype=torch.int64)
    words = source.view(torch.int16)
    qnan = 0x7E31 if dtype == 1 else 0x7FC5
    if mode == "signed_zero":
        words.zero_()
        words[:, 1::2] = signed16(0x8000)
    elif mode == "partial":
        slots[1] = -1
        words[heads:2 * heads].fill_(signed16(qnan))
    elif mode == "allpadding":
        slots.fill_(-1)
        words.fill_(signed16(qnan))
    elif mode == "nonfinite":
        special = (0x7C00, 0xFC00, qnan, 0x7C01) if dtype == 1 else (
            0x7F80, 0xFF80, qnan, 0x7F81)
        for row, word in enumerate(special):
            words[row, (row * 37) % dim] = signed16(word)
        words[4].zero_()
        words[4, 1::2] = signed16(0x8000)
        words[5].zero_()
        words[6].fill_(signed16(0x8000))
    return source, rotation_matrix(dim), slots


def oracle(source, rotation, slots, heads):
    rows, dim = source.shape
    output = torch.zeros((rows, dim), dtype=torch.float32)
    status = torch.zeros(rows, dtype=torch.int32)
    coordinates = torch.arange(dim, dtype=torch.int64)
    normalization = rotation[0, 0].clone()
    for row in range(rows):
        if int(slots[row // heads]) < 0:
            # Matches RotateOnly's explicit +0/status0 for masked query rows,
            # including an all-padding tile whose transform is never called.
            continue
        x = source[row].float().clone()
        distance = 1
        while distance < dim:
            gathered = x.index_select(0, coordinates ^ distance)
            signs = torch.where((coordinates & distance) != 0, -1.0, 1.0).to(torch.float32)
            product = torch.mul(x, signs)
            x = torch.add(product, gathered)
            distance *= 2
        output[row] = torch.mul(x, normalization)
        # This suite's finite inputs are bounded; no finite sum can overflow.
        # Each active special row has one Inf/NaN input, giving a nonfinite
        # final reduction under any legal tree. This does not replace the
        # kernel's original ReduceSum with a per-element production check.
        status[row] = 0 if bool(torch.isfinite(output[row].sum(dtype=torch.float32))) else 2
    return output, status


def export(output):
    source_pins = check_source()
    tolerance = json.loads((ROOT / "configs/acceptance.json").read_text())["store_dequant"]
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (name, rows, dim, dtype, heads, mode) in enumerate(PLAN):
        source, rotation, slots = inputs(rows, dim, dtype, heads, mode, SEED + index)
        expected, status = oracle(source, rotation, slots, heads)
        grid = min(32, (rows + 7) // 8)
        path = output / name
        path.mkdir()
        tensors = {"source": source, "rotation": rotation, "slots": slots,
                   "expected_output": expected, "expected_status": status}
        files = {}
        for label, tensor in tensors.items():
            data = raw(tensor)
            filename = label + ".bin"
            (path / filename).write_bytes(data)
            files[filename] = {"bytes": len(data), "sha256": sha(data),
                               "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        shape = f"{rows} {dim} {dtype} {heads} 1 {grid}\n".encode()
        (path / "shape.txt").write_bytes(shape)
        files["shape.txt"] = {"bytes": len(shape), "sha256": sha(shape)}
        row = {"case": name, "rows": rows, "tokens": rows // heads, "dim": dim,
               "dtype": dtype, "dtype_name": "fp16" if dtype == 1 else "bf16",
               "heads": heads, "hadamard": True, "grid": grid,
               "shape": [rows, dim, dtype, heads, 1, grid], "mode": mode,
               "seed": SEED + index, "torch_version": str(torch.__version__),
               "source_sha256": source_pins, "files": files,
               "normalization_bits": hex(struct.unpack("<I", raw(rotation[0, 0:1]))[0]),
               "oracle": "independent row-major FP32 Gather(i xor distance), then Mul(sign), then Add; one FP32 multiply by stored rotation[0,0] after all stages",
               "independent_finite_tolerance": tolerance,
               "independent_nonfinite": "NaN/+Inf/-Inf classes and exact row status; host NaN payload is not an NPU golden",
               "cross_arm_acceptance": "every output and status byte identical to saved arm0, including NaN payload and signed zero; no equal_nan comparison",
               "mask_contract": "slots always present; negative slot masks all its heads to FP32 +0 and status0",
               "status_counts": {str(code): int((status == code).sum()) for code in (0, 2)},
               "input_output_guards": "driver-owned; compare input source/rotation/slots bytes and both guards",
               "cpu_debug": "not_run", "cann": "not_run", "camodel": "not_run",
               "npu": "not_run", "graph": "not_run", "performance": "not_run"}
        (path / "fixture.json").write_text(json.dumps(row, indent=2) + "\n")
        records.append(row)
    report = {"schema": 1, "status": "inputs_exported_not_executed", "cases": records,
              "source_sha256": source_pins, "scope": "eight isolated query-rotation cases; no writer, CV or service speed claim"}
    (output / "cases.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps({"status": result["status"], "cases": [row["case"] for row in result["cases"]]}))
