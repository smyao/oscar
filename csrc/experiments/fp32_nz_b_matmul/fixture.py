"""Archive #126/#140/#145/#148/#154-156 and full startup D.4.

Host-only B-layout experiment: keep the fixed FP32 A/B values and exact
integer golden from fp32_bf16_mmad_ratio. Only B's physical bytes are permuted.
D.4: this prepares one bounded 256x256 B, never a full restored history cache.
SDK C0=8 and direct GM NZ copying are source-proven below. Host prepacking is
outside the eventual Matmul timing; it is not free work in a serving kernel.
No CANN, NPU, graph, numerical or performance execution is implied here.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
MOTHER = ROOT / "artifacts/fp32_bf16_mmad_ratio_20261010"
SDK = "artifacts/cann_sdk_matmul_move_20261010/"
DIM, C0 = 256, 8
CASE_PINS = {
    "qk_m24": "a6dd0911c7487aca367ea5d2a8b2345221deb91832ae0a34848d245b5a0b12bc",
    "pv_m126": "8eb5debfc9299ee84c02d466a7992c9709002ad0aaa8ad36696e0ee6420ccbc9",
}
MANIFEST_SHA = "9d019545775b5193484392abf669fc0cace15c6caf619b7835de0deedf484052"
SDK_PINS = {
    SDK + "utils/mx_matmul_utils.h":
        "0a197444976e793a378e4b5d448f474270814d2e71650630a9579c3cb7b36de5",
    SDK + "utils/matmul_config_utils.h":
        "282de06d4a8e6464a8afb6619c05141dd47515b1dd324088e652da4958b3088a",
    SDK + "stage/copy_cube_in/copy_tile_to_cube/copy_tile_to_cube_common.h":
        "2cf9bb4fc0c5581584900d5dc9f00d8b128a1d6f4df53d3778e5ec1490550307",
    SDK + "stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_nz.h":
        "dcaf0276d57bae56265094217732fdd30f9217a302be83dbccac9c48843e101c",
    SDK + "stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_utils.h":
        "4b625a8757c4691e1032bfa9c6f0291cc0d6a65a1c502d89f291f9566e29e3e6",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def words_to_bytes(words):
    data = array("I", words)
    if data.itemsize != 4:
        raise ValueError("32-bit uint representation required")
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def nz_index(row, column):
    return (column // C0) * DIM * C0 + row * C0 + column % C0


def pack_nz(data):
    if len(data) != DIM * DIM * 4:
        raise ValueError("only the declared physical B[256,256] is supported")
    source = struct.unpack("<65536I", data)
    packed = [0] * (DIM * DIM)
    for row in range(DIM):
        for column in range(DIM):
            packed[nz_index(row, column)] = source[row * DIM + column]
    return words_to_bytes(packed)


def unpack_nz(data):
    if len(data) != DIM * DIM * 4:
        raise ValueError("invalid NZ byte count")
    packed = struct.unpack("<65536I", data)
    return words_to_bytes(packed[nz_index(row, column)]
                          for row in range(DIM) for column in range(DIM))


def check_bijection():
    # Unique uint32 tags expose duplicates/holes that repeated numeric values
    # could hide. This checks all physical positions, not a random sample.
    labels = words_to_bytes(range(DIM * DIM))
    packed = pack_nz(labels)
    packed_words = struct.unpack("<65536I", packed)
    positions = [nz_index(row, column) for row in range(DIM) for column in range(DIM)]
    if sorted(positions) != list(range(DIM * DIM)):
        raise ValueError("NZ formula is not a full-domain bijection")
    for logical, physical in enumerate(positions):
        if packed_words[physical] != logical:
            raise ValueError("NZ spatial tag moved to the wrong position")
    if unpack_nz(packed) != labels:
        raise ValueError("full-domain uint32 inverse differs")
    return {"status": "host_passed", "unique_uint32_tags": DIM * DIM,
            "physical_positions": DIM * DIM, "nd_tags_sha256": sha(labels),
            "nz_tags_sha256": sha(packed), "inverse_raw_bytes_exact": True}


def check_source():
    if sha((MOTHER / "cases.json").read_bytes()) != MANIFEST_SHA:
        raise ValueError("fixed mother manifest changed")
    manifest = json.loads((MOTHER / "cases.json").read_text())
    for name, expected in manifest["source"]["sha256"].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError("mother source changed: " + name)
    for name, expected in SDK_PINS.items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError("re-audit changed SDK definition: " + name)
    return {"mother_manifest": str((MOTHER / "cases.json").relative_to(ROOT)),
            "mother_manifest_sha256": MANIFEST_SHA,
            "mother_source": manifest["source"], "sdk_sha256": SDK_PINS,
            "fixture_source_sha256": sha(Path(__file__).read_bytes()),
            "sdk_proof": "mx_matmul_utils.h:102-105 -> B32_C0SIZE; matmul_config_utils.h:41 -> 8; copy_tile_to_cube_common.h:484-487 -> direct CopyNZ2NZ; data_copy_wrapper_utils.h:302-334 -> source offset and stride"}


def export(output):
    source = check_source()
    bijection = check_bijection()
    manifest = json.loads((MOTHER / "cases.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name, fixture_sha in CASE_PINS.items():
        mother = MOTHER / name
        if sha((mother / "fixture.json").read_bytes()) != fixture_sha:
            raise ValueError("fixed mother case changed: " + name)
        old = json.loads((mother / "fixture.json").read_text())
        if [row for row in manifest["cases"] if row["case"] == name] != [old]:
            raise ValueError("mother case differs from its manifest")
        live, padded, layout = old["shape"]
        if (live, padded, layout) != ((24, 32, 0) if name == "qk_m24" else (126, 128, 1)):
            raise ValueError("undeclared case geometry")
        preserved = {}
        for filename in ("a_fp32.bin", "b_fp32.bin", "expected_output.bin", "shape.txt"):
            data = (mother / filename).read_bytes()
            record = old["files"][filename]
            if len(data) != record["bytes"] or sha(data) != record["sha256"]:
                raise ValueError("mother bytes/hash changed: " + filename)
            preserved[filename] = data
        a = preserved["a_fp32.bin"]
        b = preserved["b_fp32.bin"]
        golden = preserved["expected_output.bin"]
        nz = pack_nz(b)
        if unpack_nz(nz) != b:
            raise ValueError("actual B inverse failed raw uint32 equality")
        if a[live * DIM * 4:] != bytes((padded - live) * DIM * 4):
            raise ValueError("A positive-zero padding changed")
        if golden[live * DIM * 4:] != b"\x85" * ((padded - live) * DIM * 4):
            raise ValueError("golden padding poison changed")
        path = output / name
        path.mkdir()
        payloads = {"a.bin": (a, "float32", [padded, DIM]),
                    "b_nd.bin": (b, "float32", [DIM, DIM]),
                    "b_nz.bin": (nz, "float32 raw-bit permutation", [32, 16, 16, 8]),
                    "expected_output.bin": (golden, "float32", [padded, DIM]),
                    "shape.txt": (preserved["shape.txt"], "text", [3])}
        files = {}
        for filename, (data, dtype, shape) in payloads.items():
            (path / filename).write_bytes(data)
            files[filename] = {"bytes": len(data), "sha256": sha(data), "dtype": dtype, "shape": shape}
        row = {"case": name, "live_rows": live, "padded_rows": padded,
               "layout": layout, "shape": [live, padded, layout], "n": DIM, "k": DIM,
               "b_transpose": layout == 0,
               "b_physical_axes": ["N=KV", "K=D"] if layout == 0 else ["K=KV", "N=D"],
               "mother_fixture_sha256": fixture_sha,
               "mother_files": {key: old["files"][key] for key in preserved},
               "source": source, "files": files, "c0": C0,
               "pack_formula": "offset(r,c)=(c//8)*256*8+r*8+c%8; no host transpose",
               "pack_dtype_policy": "uint32 raw bits only; no float conversion, rounding, clipping or quantization",
               "host_bijection": bijection,
               "actual_b_roundtrip": {"uint32_elements": 65536, "raw_bytes_exact": True},
               "math_proof": old["math_proof"],
               "acceptance": old["acceptance"], "output_padding_word": "0x85858585",
               "timing_scope": "host prepack excluded; A and output remain ND; prepack cost is not claimed free in production",
               "cpu_debug": "not_run", "cann": "not_run", "camodel": "not_run",
               "npu": "not_run", "graph": "not_run", "performance": "not_run"}
        (path / "fixture.json").write_text(json.dumps(row, indent=2) + "\n")
        cases.append(row)
    report = {"schema": 1, "status": "inputs_exported_not_executed", "source": source,
              "host_bijection": bijection, "cases": cases,
              "scope": "two fixed pure-Matmul cases, B GM ND versus prepacked NZ, unchanged FP32 values and K128 mathematics"}
    (output / "cases.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps({"status": result["status"], "cases": [row["case"] for row in result["cases"]],
                      "host_bijection": result["host_bijection"]["status"]}))
