"""Archive #126/#140/#145/#148/#154-156 and full startup D.4.

Host-only fixed inputs for SDK versus K-major FP32 Matmul, both A/B in NZ.
Reuse existing integer-exact QK/PV and a frozen cancellation FP32 oracle.
Never replace the cancellation oracle with its FP64 diagnostic. Preserve the
per-output K128 order and gate live paired bytes before independent accuracy.
D.4: bounded matrix fixtures are not full-history restoration or proof of a
service speedup. These three finite-input cases are not full precision-domain,
error-domain, graph or target-NPU qualification. No old evidence is modified.
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
MOTHER = ROOT / "artifacts/fp32_nz_ab_matmul_20261010"
CANCEL = ROOT / "artifacts/decode_mmad_barrier_20261010"
SDK = "artifacts/cann_sdk_matmul_move_20261010/"
DIM, C0 = 256, 8
CASE_PINS = {
    "qk_m24": "7ed910bbc54b0ea41de7f290356eb39b25399aa2093adcd262d4fdad0884e487",
    "pv_m126": "920e6f83fe77b3df5117c66da4e11aa9c012f46f893eafcb65f164689585c662",
}
MANIFEST_SHA = "742fe0bbd3dadc64cb75d373fa46c006dd0c6612cbfc1fe5811e838ef38be49b"
CANCEL_MANIFEST_SHA = "61b9737ce8fc15c0843556d96f728be5a87a2f9e61c702fa6ccce29723241c55"
CANCEL_CASE_SHA = "a0c4526dae84d4ab889222963f69afe0f9ac76289a1daa78c28d371a4a0fe984"
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
    if sha((MOTHER / 'cases.json').read_bytes()) != MANIFEST_SHA:
        raise ValueError('fixed ABNZ mother manifest changed')
    if sha((CANCEL / 'cases.json').read_bytes()) != CANCEL_MANIFEST_SHA:
        raise ValueError('fixed cancellation manifest changed')
    manifest = json.loads((MOTHER / 'cases.json').read_text())
    normal_sources = manifest['source']['mother_source']['sha256']
    cancel_meta = json.loads((CANCEL / 'cancel_qk_m24' / 'fixture.json').read_text())
    if sha((CANCEL / 'cancel_qk_m24' / 'fixture.json').read_bytes()) != CANCEL_CASE_SHA:
        raise ValueError('frozen cancellation metadata changed')
    for name, wanted in {**normal_sources, **cancel_meta['source_sha256'], **SDK_PINS}.items():
        if sha((ROOT / name).read_bytes()) != wanted:
            raise ValueError('re-audit changed source: ' + name)
    acceptance = ROOT / 'configs/acceptance.json'
    if sha(acceptance.read_bytes()) != normal_sources['configs/acceptance.json']:
        raise ValueError('frozen acceptance changed')
    return {'mother_manifest': str((MOTHER / 'cases.json').relative_to(ROOT)),
            'mother_manifest_sha256': MANIFEST_SHA, 'mother_source': manifest['source']['mother_source'],
            'abnz_source': manifest['source'],
            'cancel_manifest': str((CANCEL / 'cases.json').relative_to(ROOT)),
            'cancel_manifest_sha256': CANCEL_MANIFEST_SHA,
            'cancel_case_sha256': CANCEL_CASE_SHA,
            'cancel_source_sha256': cancel_meta['source_sha256'],
            'sdk_sha256': SDK_PINS, 'fixture_source_sha256': sha(Path(__file__).read_bytes()),
            'kernel_binding': 'new K-major kernel is bound by its independent build/replay; not yet accepted here'}


def origin_case(directory, name, expected_sha):
    path = directory / name
    if sha((path / 'fixture.json').read_bytes()) != expected_sha:
        raise ValueError('origin fixture identity changed: ' + name)
    meta = json.loads((path / 'fixture.json').read_text())
    registry = json.loads((directory / 'cases.json').read_text())
    if [row for row in registry['cases'] if row['case'] == name] != [meta]:
        raise ValueError('origin fixture differs from registered manifest: ' + name)
    data = {}
    evidence = {}
    for filename, record in meta['files'].items():
        payload = (path / filename).read_bytes()
        if len(payload) != record['bytes'] or sha(payload) != record['sha256']:
            raise ValueError('origin evidence bytes changed: ' + filename)
        data[filename] = payload
        evidence[filename] = {'path': str((path / filename).relative_to(ROOT)), **record,
                              'acceptance_role': 'diagnostic_only_not_oracle' if filename == 'expected_f64_diagnostic.bin' else 'frozen_origin_evidence'}
    return meta, data, evidence


def finite_payload(data):
    return all((word & 0x7fffffff) < 0x7f800000
               for word in struct.unpack('<' + str(len(data) // 4) + 'I', data))


def export(output):
    source = check_source()
    bijection = {str(rows): check_bijection(rows) for rows in (32, 128, 256)}
    tolerance = json.loads((ROOT / 'configs/acceptance.json').read_text())['fused_attention']
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    plan = (('qk_m24', 24, 32, 0, True), ('pv_m126', 126, 128, 1, True),
            ('cancel_qk_m24', 24, 32, 0, False))
    for name, live, pad, layout, exact in plan:
        directory = MOTHER if exact else CANCEL
        origin_sha = CASE_PINS[name] if exact else CANCEL_CASE_SHA
        meta, data, evidence = origin_case(directory, name, origin_sha)
        if exact:
            a, b, a_nz, b_nz = (data[key] for key in ('a_nd.bin', 'b_nd.bin', 'a_nz.bin', 'b_nz.bin'))
            golden = data['expected_output.bin']
            math = meta['math_proof']
            golden_note = 'all original integer-golden bytes unchanged'
        else:
            a, b = data['a.bin'], data['b.bin']
            a_nz, b_nz = pack_nz(a, pad), pack_nz(b)
            # Only inactive output padding adopts the new driver's poison.
            # Every active FP32 expected byte remains the frozen original.
            golden = data['expected.bin'][:live * DIM * 4] + b'\x85' * ((pad - live) * DIM * 4)
            math = meta['math']
            golden_note = 'live bytes copied from original FP32 expected.bin; inactive tail replaced by driver poison85; FP64 diagnostic excluded'
            if meta['independent_oracle_tolerance'] != tolerance:
                raise ValueError('cancellation tolerance differs from frozen gate')
        if len(a) != pad * DIM * 4 or len(b) != DIM * DIM * 4 or len(golden) != pad * DIM * 4:
            raise ValueError('fixed case byte geometry differs')
        if not finite_payload(a) or not finite_payload(b) or not finite_payload(golden[:live * DIM * 4]):
            raise ValueError('this minimal suite is declared finite-input/finite-oracle only')
        if unpack_nz(a_nz, pad) != a or unpack_nz(b_nz) != b:
            raise ValueError('NZ operands differ from original FP32 values')
        if pack_nz(a, pad) != a_nz or pack_nz(b) != b_nz:
            raise ValueError('origin NZ packing does not match the pinned SDK geometry')
        if a[live * DIM * 4:] != bytes((pad - live) * DIM * 4):
            raise ValueError('A input padding must be positive zero')
        if golden[live * DIM * 4:] != b'\x85' * ((pad - live) * DIM * 4):
            raise ValueError('expected output padding must be poison85')
        path = output / name
        path.mkdir()
        payloads = {'a_nz.bin': (a_nz, [32, pad // 16, 16, 8]),
                    'b_nz.bin': (b_nz, [32, 16, 16, 8]),
                    'expected_output.bin': (golden, [pad, DIM]),
                    'shape.txt': (f'{live} {pad} {layout}\n'.encode(), [3])}
        files = {}
        for filename, (payload, shape) in payloads.items():
            (path / filename).write_bytes(payload)
            files[filename] = {'bytes': len(payload), 'sha256': sha(payload), 'shape': shape,
                               'dtype': 'text' if filename == 'shape.txt' else 'float32 raw bits'}
        row = {'case': name, 'live_rows': live, 'padded_rows': pad, 'layout': layout,
               'shape': [live, pad, layout], 'n': DIM, 'k': DIM, 'format': 'NZ',
               'a_transpose': False, 'b_transpose': layout == 0,
               'a_physical_rows': pad, 'org_m': {'sdk': pad, 'kmajor': pad}, 'single_m': live,
               'arms': ['sdk', 'kmajor'], 'exact_integer_oracle': exact,
               'source': source, 'origin_directory': str((directory / name).relative_to(ROOT)),
               'origin_fixture_sha256': origin_sha, 'origin_evidence': evidence,
               'math': math, 'oracle_bytes_policy': golden_note,
               'live_oracle_sha256': sha(golden[:live * DIM * 4]),
               'files': files, 'c0': C0,
               'pack_formula': 'offset(r,c)=(c//8)*physical_rows*8+r*8+c%8; no numerical casts or host transpose',
               'host_bijection': {'a': bijection[str(pad)], 'b': bijection['256']},
               'actual_a_roundtrip': {'raw_bytes_exact': True, 'padding_positive_zero_bits_exact': True},
               'actual_b_roundtrip': {'raw_bytes_exact': True},
               'order_contract': 'for every output preserve K[0:128] first, then K[128:256]; only N traversal may change',
               'paired_gate': 'compare all live output raw bits sdk versus kmajor first, including signed zero; any difference fails',
               'independent_oracle': 'exact live FP32 integer-golden bits' if exact else 'frozen original FP32 expected.bin live bytes, never FP64',
               'independent_oracle_tolerance': tolerance,
               'output_padding_word': '0x85858585',
               'acceptance': 'paired live bitwise first; normal exact-integer bits or cancellation frozen5e-3 oracle per arm; padding/input/guards exact',
               'timing_scope': 'same prepacked ABNZ inputs; host prepack excluded; pure bounded Matmul only',
               'domain_limit': 'only three finite input/oracle cases; no NaN/Inf, complete error-domain, full attention or model-quality qualification',
               'cpu_debug': 'not_run', 'cann': 'not_run', 'camodel': 'not_run',
               'npu': 'not_run', 'graph': 'not_run', 'performance': 'not_run'}
        (path / 'fixture.json').write_text(json.dumps(row, indent=2) + '\n')
        cases.append(row)
    report = {'schema': 1, 'status': 'inputs_exported_not_executed', 'source': source,
              'host_bijection': bijection, 'cases': cases,
              'scope': 'three frozen finite cases, SDK versus K-major full-FP32 Matmul; not whole precision/error/serving acceptance'}
    (output / 'cases.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps({"status": result["status"], "cases": [row["case"] for row in result["cases"]],
                      "host_bijection": {key: value["status"] for key, value in result["host_bijection"].items()}}))
