"""Archive #126/#140/#145/#148/#154-156 and full startup D.4.

Two host-only operand-throughput fixtures, not a production BF16 conversion.
FP32 and BF16 operands are exactly identical integer/64 values; independent
goldens use integer dot products, never a floating Matmul implementation.
D.4: measure only bounded Cube math with an exactly representable input set.
This cannot qualify arbitrary FP32 inputs, history reconstruction, an NPU
speedup, or a service change. Keep all original precision/error gates intact.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
SEED = 194126256
DIM = 256
POISON = 0x85858585
PLAN = (("qk_m24", 24, 32, 0), ("pv_m126", 126, 128, 1))
SOURCE_PINS = {
    "csrc/kernels/attention_cv_decode_bundle.cpp":
        "d8171f0403cf176087e1c76c64e454e6fb1e74fd79252aab3b59e848cca19401",
    "csrc/kernels/oscar_common.h":
        "20574a83f919df5bd6450cd581d1dba5a5a6f169768e7e237ba62f51b7499964",
    "configs/acceptance.json":
        "b044a5f485cd8cfed81dc76c7199717952853167b4e1d8e0aa9bff499b290811",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def little_array(code, values):
    data = array(code, values)
    required = {"h": 2, "H": 2, "I": 4}[code]
    if data.itemsize != required:
        raise ValueError("unexpected host array element size")
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def check_source():
    for name, expected in SOURCE_PINS.items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError("re-audit changed reference source: " + name)
    text = (ROOT / "csrc/kernels/attention_cv_decode_bundle.cpp").read_text()
    start = text.index("__aicore__ constexpr MatmulConfig CvConfig()")
    end = text.index("constexpr auto kCubeConfig", start)
    config = text[start:end]
    return {"sha256": {**SOURCE_PINS,
                       str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).read_bytes())},
            "current_decode_config": config,
            "current_decode_config_sha256": sha(config.encode()),
            "kernel_binding": "experiment kernel is bound separately by its build/replay; no missing or future kernel is represented as verified"}


def exact_operands(integers):
    f32 = []
    bf16 = []
    for numerator in integers:
        if not -64 <= numerator <= 64:
            raise ValueError("operand left the declared integer/64 domain")
        word = struct.unpack("<I", struct.pack("<f", numerator / 64.0))[0]
        if word & 65535:
            raise ValueError("FP32 input is not exactly representable in BF16")
        f32.append(word)
        bf16.append(word >> 16)
    return little_array("I", f32), little_array("H", bf16)


def integer_dot(a, b, live, padded, layout):
    # Physical B is [256,256] in both cases. QK uses B.T; PV uses B.
    columns = ([b[column * DIM:(column + 1) * DIM] for column in range(DIM)]
               if layout == 0 else
               [[b[k * DIM + column] for k in range(DIM)] for column in range(DIM)])
    words = [POISON] * (padded * DIM)
    maximum = 0
    nonzero = 0
    for row in range(live):
        left = a[row * DIM:(row + 1) * DIM]
        for column, right in enumerate(columns):
            numerator = sum(x * y for x, y in zip(left, right))
            if abs(numerator) > 2 ** 20:
                raise ValueError("integer dot exceeds its proven FP32 exact range")
            maximum = max(maximum, abs(numerator))
            nonzero += numerator != 0
            exact = numerator / 4096.0  # exact binary rational in Python FP64
            encoded = struct.pack("<f", exact)
            if struct.unpack("<f", encoded)[0] != exact:
                raise ValueError("integer golden lost precision when encoded as FP32")
            words[row * DIM + column] = struct.unpack("<I", encoded)[0]
    return little_array("I", words), {"max_abs_live_integer_numerator": maximum,
                                      "nonzero_live_outputs": nonzero}


def export(output):
    source = check_source()
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, (name, live, padded, layout) in enumerate(PLAN):
        rng = random.Random(SEED + index)
        a = [rng.randint(-64, 64) for _ in range(live * DIM)] + [0] * ((padded - live) * DIM)
        b = [rng.randint(-64, 64) for _ in range(DIM * DIM)]
        a32, a16 = exact_operands(a)
        b32, b16 = exact_operands(b)
        golden, observed = integer_dot(a, b, live, padded, layout)
        path = output / name
        path.mkdir()
        payloads = {"a_fp32.bin": (a32, "float32", [padded, DIM]),
                    "a_bf16.bin": (a16, "bfloat16", [padded, DIM]),
                    "b_fp32.bin": (b32, "float32", [DIM, DIM]),
                    "b_bf16.bin": (b16, "bfloat16", [DIM, DIM]),
                    "a_int16.bin": (little_array("h", a), "int16 numerator", [padded, DIM]),
                    "b_int16.bin": (little_array("h", b), "int16 numerator", [DIM, DIM]),
                    "expected_output.bin": (golden, "float32", [padded, DIM])}
        files = {}
        for filename, (data, dtype, shape) in payloads.items():
            (path / filename).write_bytes(data)
            files[filename] = {"bytes": len(data), "sha256": sha(data), "dtype": dtype, "shape": shape}
        shape = f"{live} {padded} {layout}\n".encode()
        (path / "shape.txt").write_bytes(shape)
        files["shape.txt"] = {"bytes": len(shape), "sha256": sha(shape)}
        row = {"case": name, "live_rows": live, "padded_rows": padded,
               "layout": layout, "shape": [live, padded, layout], "n": DIM, "k": DIM,
               "operation": "A @ B.T" if layout == 0 else "A @ B",
               "seed": SEED + index, "source": source, "files": files,
               "math_proof": {
                   "operands": "integer numerator in [-64,64] divided by64; every FP32 bit pattern has low16 bits zero and equals its BF16 operand exactly",
                   "products": "integer numerator in [-4096,4096] divided by4096, exactly representable in FP32",
                   "every_partial_sum": "sum of any subset of at most256 products has |numerator|<=256*4096=2^20; denominator4096; all intermediate sums are exactly representable in FP32 regardless of reduction tree",
                   "quantization_error": "zero on every saved input element, verified by FP32-to-BF16 raw-bit equality",
                   "golden": "Python integer dot product then exact division by4096 and checked FP32 encoding; no Torch/NumPy or floating Matmul oracle",
                   **observed},
               "a_padding": "positive-zero input bytes in both operand dtypes",
               "output_padding_word": "0x85858585",
               "acceptance": "all live output FP32 bits equal exact integer golden; padded output unchanged; input/guard bytes unchanged",
               "scope": "controlled operand-throughput ratio only; cannot justify general FP32-to-BF16 replacement",
               "cpu_debug": "not_run", "cann": "not_run", "camodel": "not_run",
               "npu": "not_run", "graph": "not_run", "performance": "not_run"}
        (path / "fixture.json").write_text(json.dumps(row, indent=2) + "\n")
        cases.append(row)
    report = {"schema": 1, "status": "inputs_exported_not_executed", "cases": cases,
              "source": source, "scope": "two cases for four runs: each same-byte-value input evaluated with FP32 or BF16 operands and FP32 accumulation"}
    (output / "cases.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps({"status": result["status"], "cases": [row["case"] for row in result["cases"]]}))
