"""Archive #126/#140/#145/#148/#154-156 and full startup D.4.

Host-only matched-shape ND/B-NZ/A+B-NZ experiment: reuse fixed FP32 A/B operands from fp32_bf16_mmad_ratio and compute new
integer goldens for the crossed QK/PV shapes. Only A/B physical bytes are permuted; C remains ND and live M is unchanged.
D.4: this prepares bounded A[32/128,256] and B[256,256], never a full history cache.
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


def validate_geometry(rows, cols):
    if rows <= 0 or cols <= 0 or rows % 16 or cols % C0:
        raise ValueError('NZ fixture requires physical rows multiple16 and columns multiple8')


def nz_index(row, column, rows=DIM):
    return (column // C0) * rows * C0 + row * C0 + column % C0


def pack_nz(data, rows=DIM, cols=DIM):
    validate_geometry(rows, cols)
    if len(data) != rows * cols * 4:
        raise ValueError('physical matrix byte count differs')
    source = struct.unpack('<' + str(rows * cols) + 'I', data)
    packed = [0] * (rows * cols)
    for row in range(rows):
        for column in range(cols):
            packed[nz_index(row, column, rows)] = source[row * cols + column]
    return words_to_bytes(packed)


def unpack_nz(data, rows=DIM, cols=DIM):
    validate_geometry(rows, cols)
    if len(data) != rows * cols * 4:
        raise ValueError('invalid NZ byte count')
    packed = struct.unpack('<' + str(rows * cols) + 'I', data)
    return words_to_bytes(packed[nz_index(row, column, rows)]
                          for row in range(rows) for column in range(cols))


def check_bijection(rows=DIM, cols=DIM):
    validate_geometry(rows, cols)
    labels = words_to_bytes(range(rows * cols))
    packed = pack_nz(labels, rows, cols)
    packed_words = struct.unpack('<' + str(rows * cols) + 'I', packed)
    positions = [nz_index(row, column, rows) for row in range(rows) for column in range(cols)]
    if sorted(positions) != list(range(rows * cols)):
        raise ValueError('NZ formula is not a full-domain bijection')
    for logical, physical in enumerate(positions):
        if packed_words[physical] != logical:
            raise ValueError('NZ spatial tag moved to the wrong position')
    if unpack_nz(packed, rows, cols) != labels:
        raise ValueError('full-domain uint32 inverse differs')
    return {'status': 'host_passed', 'rows': rows, 'cols': cols,
            'unique_uint32_tags': rows * cols, 'physical_positions': rows * cols,
            'nd_tags_sha256': sha(labels), 'nz_tags_sha256': sha(packed),
            'inverse_raw_bytes_exact': True}


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


def exact_golden(a_nd, b_nd, live, pad, layout):
    if (live, pad, layout) not in ((126, 128, 0), (24, 32, 1)):
        raise ValueError('only declared matched shapes are allowed')
    if len(a_nd) != pad * DIM * 4 or len(b_nd) != DIM * DIM * 4:
        raise ValueError('FP32 input size differs')
    integers = []
    for data in (a_nd, b_nd):
        values = struct.unpack('<' + str(len(data) // 4) + 'f', data)
        words = []
        for value in values:
            if not -1.0 <= value <= 1.0:
                raise ValueError('input left finite integer/64 domain')
            numerator = int(value * 64)
            if value != numerator / 64.0 or not -64 <= numerator <= 64:
                raise ValueError('input is not exactly integer/64')
            words.append(numerator)
        integers.append(words)
    if a_nd[live * DIM * 4:] != bytes((pad - live) * DIM * 4):
        raise ValueError('A padding must be positive zero bits')
    left, right = integers
    columns = ([right[c * DIM:(c + 1) * DIM] for c in range(DIM)] if layout == 0 else
               [[right[k * DIM + c] for k in range(DIM)] for c in range(DIM)])
    output = [0x85858585] * (pad * DIM)
    for row in range(live):
        a = left[row * DIM:(row + 1) * DIM]
        for column, b in enumerate(columns):
            numerator = sum(x * y for x, y in zip(a, b))
            if abs(numerator) > 2 ** 20:
                raise ValueError('integer dot exceeded proven exact FP32 range')
            value = numerator / 4096.0
            encoded = struct.pack('<f', value)
            if struct.unpack('<f', encoded)[0] != value:
                raise ValueError('exact integer golden changed during FP32 encoding')
            output[row * DIM + column] = struct.unpack('<I', encoded)[0]
    return words_to_bytes(output)


def export(output):
    source = check_source()
    bijection = {str(rows): check_bijection(rows) for rows in (32, 128, 256)}
    manifest = json.loads((MOTHER / 'cases.json').read_text())
    mothers = {}
    for name, digest in CASE_PINS.items():
        path = MOTHER / name
        if sha((path / 'fixture.json').read_bytes()) != digest:
            raise ValueError('fixed mother case changed: ' + name)
        record = json.loads((path / 'fixture.json').read_text())
        if [row for row in manifest['cases'] if row['case'] == name] != [record]:
            raise ValueError('mother case differs from registered manifest')
        mothers[name] = record
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    plan = (('qk_m126', 126, 128, 0, 'pv_m126', 'qk_m24'),
            ('pv_m24', 24, 32, 1, 'qk_m24', 'pv_m126'))
    for name, live, pad, layout, a_case, b_case in plan:
        inputs = {}
        mother_files = {}
        for label, mother_case, filename in (('a', a_case, 'a_fp32.bin'),
                                              ('b', b_case, 'b_fp32.bin')):
            data = (MOTHER / mother_case / filename).read_bytes()
            record = mothers[mother_case]['files'][filename]
            if len(data) != record['bytes'] or sha(data) != record['sha256']:
                raise ValueError('fixed mother input changed: ' + mother_case + '/' + filename)
            inputs[label] = data
            mother_files[filename] = record
        a, b = inputs['a'], inputs['b']
        golden = exact_golden(a, b, live, pad, layout)
        a_nz, b_nz = pack_nz(a, pad), pack_nz(b)
        if unpack_nz(a_nz, pad) != a or unpack_nz(b_nz) != b:
            raise ValueError('A/B raw-bit NZ inverse failed')
        packed_a = struct.unpack('<' + str(pad * DIM) + 'I', a_nz)
        if any(packed_a[nz_index(r, c, pad)] != 0
               for r in range(live, pad) for c in range(DIM)):
            raise ValueError('NZ A padding changed')
        path = output / name
        path.mkdir()
        payloads = {'a_nd.bin': (a, 'float32', [pad, DIM]),
                    'a_nz.bin': (a_nz, 'float32 raw-bit permutation', [32, pad // 16, 16, 8]),
                    'b_nd.bin': (b, 'float32', [DIM, DIM]),
                    'b_nz.bin': (b_nz, 'float32 raw-bit permutation', [32, 16, 16, 8]),
                    'expected_output.bin': (golden, 'float32', [pad, DIM]),
                    'shape.txt': (f'{live} {pad} {layout}\n'.encode(), 'text', [3])}
        files = {}
        for filename, (data, dtype, shape) in payloads.items():
            (path / filename).write_bytes(data)
            files[filename] = {'bytes': len(data), 'sha256': sha(data), 'dtype': dtype, 'shape': shape}
        row = {'case': name, 'live_rows': live, 'padded_rows': pad, 'layout': layout,
               'shape': [live, pad, layout], 'n': DIM, 'k': DIM,
               'operation': 'A @ B.T' if layout == 0 else 'A @ B',
               'a_transpose': False, 'b_transpose': layout == 0,
               'a_physical_rows': pad, 'org_m': {'nd': live, 'b_nz': live, 'ab_nz': pad},
               'single_m': live, 'mother_a_case': a_case, 'mother_b_case': b_case,
               'mother_case_sha256': CASE_PINS, 'mother_files': mother_files,
               'b_physical_axes': ['N=KV', 'K=D'] if layout == 0 else ['K=KV', 'N=D'],
               'source': source, 'files': files, 'c0': C0,
               'pack_formula': 'offset(r,c)=(c//8)*physical_rows*8+r*8+c%8; no host transpose',
               'pack_dtype_policy': 'uint32 raw bits only; no float conversion or rounding',
               'host_bijection': {'a': bijection[str(pad)], 'b': bijection['256']},
               'actual_a_roundtrip': {'uint32_elements': pad * DIM, 'raw_bytes_exact': True,
                                      'padding_positive_zero_bits_exact': True},
               'actual_b_roundtrip': {'uint32_elements': DIM * DIM, 'raw_bytes_exact': True},
               'math_proof': {
                   'operands': 'exactly the fixed mother FP32 values, verified integer/64 in [-1,1]',
                   'products': 'integer numerator in [-4096,4096] divided by4096',
                   'every_partial_sum': 'any subset of at most256 products has |numerator|<=2^20; integer/4096 and every FP32 intermediate are exact',
                   'golden': 'new Python integer dot for this crossed shape/layout, then checked exact FP32 encoding; not reused from a differently oriented case and not a floating Matmul oracle'},
               'acceptance': 'live output bytes equal exact integer golden; output padding85; all input and guard bytes unchanged',
               'output_padding_word': '0x85858585',
               'timing_scope': 'prepacking excluded; ND and B-only NZ use production live orgM, A+B NZ uses physical padded orgM; all arms singleM=live and C ND',
               'cpu_debug': 'not_run', 'cann': 'not_run', 'camodel': 'not_run',
               'npu': 'not_run', 'graph': 'not_run', 'performance': 'not_run'}
        (path / 'fixture.json').write_text(json.dumps(row, indent=2) + '\n')
        cases.append(row)
    report = {'schema': 1, 'status': 'inputs_exported_not_executed', 'source': source,
              'host_bijection': bijection, 'cases': cases,
              'scope': 'two additional matched pure-Matmul shapes, ND/B-NZ/A+B-NZ, fixed FP32 values and K128; not a serving or producer-inclusive speed result'}
    (output / 'cases.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps({"status": result["status"], "cases": [row["case"] for row in result["cases"]],
                      "host_bijection": {key: value["status"] for key, value in result["host_bijection"].items()}}))
