"""Archive #126/#145/#154, D.4: independent revision3 atomic-state recovery.

Reuse revision2 random P/V/alpha bytes. Two separately initialized ordinary
Matmul outputs test whether the same Matmul object returns to default atomic0.
This is host input/source preparation, not NPU, graph or performance evidence.
No v1/v2 source, input, result, tolerance or failure record is overwritten.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.c4_atomic_state import fixture as base
from csrc.experiments.c4_atomic_state import fixture_readback as v2

CASES = ("random",)
KERNEL = "oscar_c4_atomic_state_recovery_kernel"
LIBRARY = "liboscar_c4_atomic_state_recovery.so"
KERNEL_SHA = "d225fbeda5f2f354a5cb9de02c737e271e5d474ef073921b75027af31ec5cb43"
POINTERS = ("p", "v", "alpha", "pv", "reference", "atomic", "status",
            "atomic_readback", "normal_before", "normal_after")
V2_RANDOM_SHA = {
    "p": "9ffbf21ef7ff398a5d794decdc6938b09e6f7789a14dc622e75c5ade79d9ba05",
    "v": "fda14bd335362312f72e144e1efbb82734638073b8c09495720a2c11fa992b0b",
    "alpha": "cd8fbd92f3da64c4840e6b41e04b7255dd1d68375d76f609bd3349fc5a9444b2",
}


def check_source():
    from csrc.experiments.c4_atomic_state import generate_recovery_v3 as revision
    original = v2.check_source()
    revision.check()
    kernel = (HERE / "kernel_recovery.cpp").read_bytes()
    if base.sha(kernel) != KERNEL_SHA:
        raise ValueError("revision3 kernel changed; re-audit source before generating inputs")
    if base.config_block(kernel.decode()) != base.config_block((HERE / "kernel.cpp").read_text()):
        raise ValueError("revision3 changed the established static Matmul configuration")
    return {"revision": 3, "revision2_source": original,
            "kernel": KERNEL, "library_name": LIBRARY,
            "kernel_sha256": base.sha(kernel),
            "generator_sha256": base.sha((HERE / "generate_recovery_v3.py").read_bytes()),
            "fixture_source_sha256": base.sha(Path(__file__).read_bytes()),
            "replay_source_sha256": base.sha((HERE / "replay_recovery.py").read_bytes()),
            "gm_pointer_count": 10, "gm_pointer_order": list(POINTERS)}


def initial_outputs():
    size = 128 * 256 * 4
    return {"normal_before": b"\x85" * size,
            "normal_after": struct.pack("<f", 1.0) * (128 * 256)}


def normal_transposed_proof(inputs):
    # In this fixed random fixture both operands are integer multiples of
    # 1/64 in [-1,1]. Every product and every possible partial sum is exactly
    # representable in FP32: integer numerator <=256*64*64=2^20, /2^12.
    integers = {}
    for name in ("p", "v"):
        count = len(inputs[name]) // 4
        values = struct.unpack("<" + "f" * count, inputs[name])
        if any(not -1.0 <= value <= 1.0 or value * 64 != int(value * 64) for value in values):
            raise ValueError("random operand no longer satisfies its exact finite lattice")
        integers[name] = [int(value * 64) for value in values]
    witness = None
    for column in range(256):
        numerator = sum(integers["p"][k] * integers["v"][column * 256 + k] for k in range(256))
        if numerator:
            value = numerator / 4096.0
            word = struct.unpack("<I", struct.pack("<f", value))[0]
            plus_one = struct.unpack("<I", struct.pack("<f", value + 1.0))[0]
            if word == plus_one:
                raise ValueError("atomic+1 recovery witness is not FP32-distinguishable")
            witness = {"row": 0, "column": column, "exact_numerator": numerator,
                       "denominator": 4096, "value": value,
                       "normal_bits": hex(word), "plus_one_bits": hex(plus_one)}
            break
    if witness is None:
        raise ValueError("random ordinary Matmul has no nonzero recovery witness")
    return {"operation": "P @ V.T", "all_live_outputs_finite": True,
            "reason": "operands are integer/64 within [-1,1]; each 256-term product sum is integer/4096 with |numerator|<=2^20",
            "maximum_abs_output": 256.0, "nonzero_plus_one_witness": witness,
            "scope": "host exact-arithmetic input proof, not execution or hardware acceptance"}


def export(output):
    source = check_source()
    inputs = base.inputs("random")
    if {name: base.sha(data) for name, data in inputs.items()} != V2_RANDOM_SHA:
        raise ValueError("revision3 random inputs differ from the saved revision2 bytes")
    proof = normal_transposed_proof(inputs)
    initial = initial_outputs()
    output.mkdir(parents=True, exist_ok=False)
    directory = output / "random"
    directory.mkdir()
    for name, data in inputs.items():
        (directory / (name + ".bin")).write_bytes(data)
    initial_records = {}
    for name, filename in (("normal_before", "before_initial.bin"),
                           ("normal_after", "after_initial.bin")):
        data = initial[name]
        (directory / filename).write_bytes(data)
        initial_records[name] = {"path": filename, "bytes": len(data), "sha256": base.sha(data),
                                 "word": "0x85858585" if name == "normal_before" else "0x3f800000"}
    (directory / "shape.txt").write_text("126 128 256 256\n")
    row = {"case": "random", "revision": 3, "shape": [126, 128, 256, 256],
           "cube_cores": 1, "aiv_actors": 2, "gm_pointer_count": 10,
           "gm_pointer_order": list(POINTERS), "input_sha256": V2_RANDOM_SHA,
           "input_bytes": {name: len(data) for name, data in inputs.items()},
           "initial_outputs": initial_records, "normal_transposed_proof": proof,
           "acceptance": "preserve all revision2 paired/readback/actor/input/guard bitwise gates; live normal_before==normal_after, both finite, padding independently retains initial bytes",
           "readback_acceptance": "all128 rows bitwise equal to both atomic and reference; padding separately counted",
           "nonfinite_classification_separate": False, "expected_anti_k128_live_reference_bits": None,
           "npu": "not_run", "graph": "not_run", "whole_c4_error_contract": "not_run"}
    (directory / "fixture.json").write_text(json.dumps(row, indent=2) + "\n")
    report = {"schema": 1, "revision": 3, "source": source, "cases": [row],
              "scope": "same-object default atomic0 recovery around one original paired atomic/readback experiment; random only",
              "cpu_debug": "not_run", "cann_compile": "not_run", "camodel": "not_run",
              "npu": "not_run", "performance": "not_run"}
    (output / "cases.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check_source:
        print(json.dumps(check_source(), indent=2))
    elif args.output:
        print(json.dumps({"status": "revision3_inputs_exported_not_executed", "cases": len(export(args.output)["cases"])}))
    else:
        parser.error("choose --check-source or --output NEW_DIRECTORY")
