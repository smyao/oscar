"""Archive #111/#126/#145/#148/#157/#164: v3 codec/attention device gate.

D.4: bounded new-token writer and KV256 fused reader; no whole-history device
restore. All goldens come from independent predeclared v3 math. This command
requires a real NPU, never substitutes host execution, and never times kernels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .phase import atomic_json
from .probe_bf16_attention import WORKSPACE, CLUSTER, allocate, require_real_npu

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'oscar_bf16_attention_v3'
CACHE_FORMAT = 'striped_fp32_v2'
OPERATORS = {'q4': 'attention_cv_aligned_decode_out', 'q1': 'attention_cv_aligned_decode_q1_out',
             'c4': 'attention_cv_aligned_cluster4_out', 'c16': 'attention_cv_aligned_cluster16_out'}
WRITER = 'store_aligned_int2_out'
QUERY_COPY = 'copy_validate_bf16_query_out'
SOURCE_PATHS = (
 'csrc/kernels/attention_cv_aligned_decode.cpp', 'csrc/kernels/attention_cv_aligned_cluster4.cpp',
 'csrc/kernels/attention_cv_aligned_cluster16.cpp', 'csrc/kernels/attention_aligned_unpack.h',
 'csrc/kernels/store_aligned_int2.cpp', 'csrc/include/oscar_aligned_decode_launch.h',
 'csrc/include/oscar_aligned_cluster_launch.h', 'csrc/include/oscar_aligned_store_launch.h',
 'csrc/kernels/attention_bf16_cast.h', 'csrc/kernels/attention_tasks.cpp', 'csrc/kernels/merge_lse.cpp',
 'csrc/striped_attention_bindings.cpp', 'csrc/torch_bindings.cpp',
 'csrc/aligned_attention_bindings.cpp', 'csrc/aligned_store_bindings.cpp',
 'tools/aligned_reference.py', 'tools/aligned_runtime_oracle.py', 'tools/aligned_fixture.py',
 'tools/bf16_reference.py', 'oscar_ascend/ops/reference.py',
 'tools/probe_history_reuse.py', 'tools/probe_bf16_attention.py',
 'tools/probe_mdl_decode.py', 'tools/probe_striped_cache.py',
 'configs/acceptance_aligned_bf16.json', 'oscar_ascend/runtime.py',
 'oscar_ascend/integration/impl.py', 'oscar_ascend/integration/aligned_math.py',
 'oscar_ascend/integration/cache_storage_ops.py', 'oscar_ascend/layout.py', 'oscar_ascend/integration/specs.py',
 'oscar_ascend/ops/cv_dispatch.py', 'oscar_ascend/ops/contracts.py',
 'tools/serving_variants.py', 'tools/probe_aligned_attention.py', 'tools/probe_current_only.py',
 'tools/probe_first_mtp_current.py', 'tools/probe_mixed_decode_split.py', 'tools/probe_prefix_cache.py',
 'tools/observe_serve.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_binding(root=ROOT):
    return {name: digest(root / name) for name in SOURCE_PATHS}


def case_plan():
    from dataclasses import replace
    from .probe_bf16_attention import case_plan as legacy_plan
    rows = []
    for case in legacy_plan():
        for mode in (0, 1):
            shape = replace(case['shape'], name=case['shape'].name.replace('bf16_', 'aligned_') + f'_mode{mode}')
            rows.append({**case, 'shape': shape, 'dequant_mode': mode})
    return tuple(rows)


def launch(ops, family, tensors, fixture, buffers, cores, *, dequant_mode, merge=True):
    from . import probe_history_reuse as reuse
    if dequant_mode not in (0, 1):
        raise ValueError('explicit prefill0/decode1 dequant_mode required')
    spec = fixture['spec']
    args = [tensors[k] for k in ('q', 'qr', 'ck', 'cv', 'rv', 'raw', 'table', 'wk', 'wv', 'tags')]
    args += [buffers[k] for k in ('tasks', 'partial', 'lse', 'status', 'workspace')]
    if CLUSTER[family] > 1:
        args.append(buffers['cluster_stats'])
    args += [reuse.BLOCK_TOKENS, fixture['blocks'], reuse.PREFIX, fixture['stride'],
             reuse.SINK, spec.recent_tokens, reuse.SPECULATIVE, spec.splits,
             fixture['scale'], dequant_mode, cores]
    getattr(ops, OPERATORS[family])(*args)
    if merge:
        n, h, d, s = fixture['tokens'], spec.heads, spec.dim, 3 * spec.splits
        ops.merge_lse_out(buffers['partial'].view(n*h, s, d), buffers['lse'].view(n*h, s),
                          buffers['output'], buffers['output_lse'], buffers['merge_status'])
        buffers['bf16'].copy_(buffers['output'])


def expected_fault_status(torch, fixture, tasks, family):
    """Declared task/lane error domain; no candidate status is used as golden."""
    spec = fixture['spec']; rows = torch.zeros((tasks.shape[0], 2), dtype=torch.int32)
    # C4/C16 H32 unpack batches preserve the original alternating H16 lane
    # ownership (lane0:0..15,32..47; lane1:16..31,48..63).
    lane_rows, half_kv = (16, 64) if family in ('q4', 'q1') else (64, 16)
    group = spec.heads // spec.kv_heads
    for index, task in enumerate(tasks.tolist()):
        qb, count, head, begin, end, request, segment, kind = task[:8]
        if count <= 0 or kind != 0:
            continue
        if spec.corrupt_metadata and request == 0 and head == spec.kv_heads - 1 and begin <= spec.corrupt_position < end:
            # KV publication alternates two AIV lanes within each 2*HalfKv.
            lane = ((spec.corrupt_position - begin) % (2 * half_kv)) // half_kv
            rows[index, lane] = 3
        if spec.corrupt_qr and head == 0 and qb <= spec.corrupt_qr_token < qb + count:
            rows[index, ((spec.corrupt_qr_token - qb) * group) // lane_rows] = 2
    if not bool(rows.ne(0).any()):
        raise RuntimeError('fault fixture failed to address a real source0 task')
    return rows


def check_outputs(torch, fixture, buffers, golden, tasks, family):
    from .bf16_reference import error_metrics
    if not torch.equal(buffers['tasks'].cpu(), tasks):
        raise RuntimeError('aligned reader modified prepared tasks')
    spec = fixture['spec']; invalid = spec.corrupt_metadata or spec.corrupt_qr
    if invalid:
        want = expected_fault_status(torch, fixture, tasks, family)
        if not torch.equal(buffers['status'].cpu(), want):
            raise RuntimeError('aligned exact task/lane status differs from declared error domain')
        mask = torch.zeros_like(buffers['lse'], device='cpu', dtype=torch.bool)
        lane_rows = 16 if family in ('q4', 'q1') else 64
        group = spec.heads // spec.kv_heads
        for index, row in enumerate(tasks.tolist()):
            qb, count, head, _, _, _, segment = row[:7]
            if count <= 0:
                continue
            for lane in (0, 1):
                if want[index, lane]:
                    for r in range(lane * lane_rows, min((lane + 1) * lane_rows, count * group)):
                        mask[qb + r // group, head * group + r % group, segment] = True
        if not torch.equal(torch.isnan(buffers['lse'].cpu()), mask):
            raise RuntimeError('aligned error LSE classification changed')
        if not torch.equal(torch.isnan(buffers['partial'].cpu()), mask[..., None].expand_as(buffers['partial'])):
            raise RuntimeError('aligned error output classification changed')
        return {'status': 'passed', 'precision': 'exact_declared_task_lane_status_and_NaN'}
    for name in ('status', 'merge_status'):
        if bool(buffers[name].ne(0).any()):
            raise RuntimeError('aligned valid case has nonzero ' + name)
    details = {}
    for name in ('partial', 'lse', 'output', 'output_lse'):
        actual = buffers[name].cpu(); want = golden[name].reshape_as(actual)
        details[name] = error_metrics(actual, want)
        if not details[name]['passed']:
            raise RuntimeError('independent aligned v3 oracle mismatch ' + name + ': ' + str(details[name]))
    if not torch.equal(buffers['bf16'], buffers['output'].to(torch.bfloat16)):
        raise RuntimeError('BF16 final copy differs from actual FP32 merge cast')
    dead = fixture['cpu']['slots'] < 0
    if bool(dead.any()) and (bool(buffers['partial'].cpu()[dead].view(torch.int32).ne(0).any()) or
                             not bool(torch.isneginf(buffers['lse'].cpu()[dead]).all())):
        raise RuntimeError('aligned padded output/LSE not exactly initialized')
    return {'status': 'passed', 'precision': 'independent_v3_all_outputs', 'detail': details}


def run_case(torch, ops, case, device, cores):
    from dataclasses import replace
    from . import probe_history_reuse as reuse
    from . import probe_mdl_decode as mdl
    from .aligned_fixture import make_fixture
    from .aligned_reference import fixture_tasks_reference
    from .probe_striped_cache import _store_bits_equal
    family, mode = case['family'], case['dequant_mode']
    fixture = make_fixture(torch, case['shape'])
    tensors = {name: value.to(device) for name, value in fixture['cpu'].items()}
    buffers = allocate(torch, fixture, device, cores, family)
    reuse._prepare(torch, ops, tensors, fixture, buffers)
    tasks = buffers['tasks'].cpu(); invalid = case['shape'].corrupt_metadata or case['shape'].corrupt_qr
    golden = None if invalid else fixture_tasks_reference(fixture, tasks, mode)
    before = {name: value.clone() for name, value in tensors.items()}
    def execute():
        launch(ops, family, tensors, fixture, buffers, cores, dequant_mode=mode, merge=not invalid)
    mdl.poison(torch, buffers); execute(); torch.npu.synchronize()
    check = check_outputs(torch, fixture, buffers, golden, tasks, family)
    if not all(_store_bits_equal(torch, before[name], value) for name, value in tensors.items()):
        raise RuntimeError('aligned reader mutated an input')
    if not invalid and CLUSTER[family] > 1:
        count = int(buffers['cluster_stats'][:, 0].sum().cpu())
        if bool(count) != bool(case['shape'].expect_clusters):
            raise RuntimeError('declared mature/generic aligned schedule was not exercised')
    graph_report = None
    if case['graph']:
        addresses = {name: value.data_ptr() for name, value in tensors.items()}
        addresses.update({'buffer_' + name: value.data_ptr() for name, value in buffers.items() if value is not None})
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode='thread_local', auto_dispatch_capture=True):
            execute()
        states = []
        for state in ('original', 'changed_query', 'changed_context'):
            if state == 'changed_query':
                fixture['cpu']['q'][0].neg_(); fixture['cpu']['qr'][0].neg_()
                tensors['q'].copy_(fixture['cpu']['q']); tensors['qr'].copy_(fixture['cpu']['qr'])
            elif state == 'changed_context':
                shape = replace(case['shape'], contexts=tuple(x + 32 for x in case['shape'].contexts))
                fixture = make_fixture(torch, shape)
                for name, value in fixture['cpu'].items():
                    if value.shape != tensors[name].shape:
                        raise RuntimeError('aligned graph allocation changed ' + name)
                    tensors[name].copy_(value)
                reuse._prepare(torch, ops, tensors, fixture, buffers); tasks = buffers['tasks'].cpu()
            golden = fixture_tasks_reference(fixture, tasks, mode)
            mdl.poison(torch, buffers); graph.replay(); torch.npu.synchronize()
            check_outputs(torch, fixture, buffers, golden, tasks, family)
            snapshot = mdl.snapshot(buffers); mdl.poison(torch, buffers); execute(); torch.npu.synchronize()
            mdl.same_bits(torch, snapshot, buffers)
            states.append({'state': state, 'v3_oracle': 'passed', 'eager_replay_bitwise': 'passed'})
        current = {name: value.data_ptr() for name, value in tensors.items()}
        current.update({'buffer_' + name: value.data_ptr() for name, value in buffers.items() if value is not None})
        if current != addresses:
            raise RuntimeError('aligned graph input/workspace address changed')
        graph_report = {'capture': 'passed', 'replay': 'passed', 'same_addresses': True, 'states': states}
    return {'case': case['shape'].name, 'family': family, 'operator': OPERATORS[family],
            'dequant_mode': mode, 'status': 'passed', 'all_tasks': tasks.shape[0],
            'check': check, 'graph': graph_report, 'timing': None}


def writer_rotation_gate(torch, ops, device, cores):
    """Exact input-driven codec/window checks plus BF16 rotation and Q guard."""
    from .aligned_reference import pack_kv, rotate_bf16
    from . import probe_history_reuse as reuse
    from .probe_striped_cache import _store_bits_equal
    n, h, d, block, blocks, prefix = 17, 2, 256, 512, 2, 64
    generator = torch.Generator().manual_seed(188144)
    original_k = torch.randn(n, h, d, generator=generator).bfloat16()
    original_v = torch.randn(n, h, d, generator=generator).bfloat16()
    rotation = reuse._hadamard(torch, d).bfloat16()
    kr, vr = rotate_bf16(original_k, rotation), rotate_bf16(original_v, rotation)
    slots = torch.arange(n, dtype=torch.int64); slots[-1] = -1
    positions = torch.arange(n, dtype=torch.int64)
    stride = block * h * 144
    initial_raw = torch.full((prefix + blocks * stride + 64,), 0xA5, dtype=torch.uint8)
    wk = torch.full((blocks, 64 + 259, h, d), float('nan'), dtype=torch.bfloat16)
    wv = wk.clone(); tags = torch.full((blocks, 323), -1, dtype=torch.int64)
    inputs = [x.to(device).clone() for x in (kr, vr, original_k, original_v, slots, positions)]
    raw, key, value, tag = [x.to(device).clone() for x in (initial_raw, wk, wv, tags)]
    status = torch.empty(n, h, dtype=torch.int32, device=device)
    def execute():
        getattr(ops, WRITER)(*inputs, raw, key, value, tag, status,
            block, blocks, prefix, stride, 64, 259, .96, .92)
    want = initial_raw.clone(); encoded = pack_kv(kr, vr, .96, .92)
    for token in range(n - 1):
        begin = prefix + token * h * 144
        want[begin:begin + h * 144] = encoded[token].reshape(-1)
        wk[0, token] = original_k[token]; wv[0, token] = original_v[token]; tags[0, token] = token
    execute(); torch.npu.synchronize()
    if not torch.equal(raw.cpu(), want) or bool(status.ne(0).any()):
        raise RuntimeError('aligned writer packed bytes/FP32 metadata/status differ from independent v3 oracle')
    for actual, expected in ((key.cpu(), wk), (value.cpu(), wv), (tag.cpu(), tags)):
        if not _store_bits_equal(torch, actual, expected):
            raise RuntimeError('aligned writer changed precise original-space windows or canaries')
    graph = torch.npu.NPUGraph()
    addresses = [x.data_ptr() for x in (*inputs, raw, key, value, tag, status)]
    with torch.npu.graph(graph, capture_error_mode='thread_local', auto_dispatch_capture=True):
        execute()
    inputs[0].neg_(); inputs[1].neg_()
    changed = pack_kv(-kr, -vr, .96, .92)
    for token in range(n - 1):
        begin = prefix + token * h * 144; want[begin:begin + h * 144] = changed[token].reshape(-1)
    graph.replay(); torch.npu.synchronize()
    if not torch.equal(raw.cpu(), want) or bool(status.ne(0).any()):
        raise RuntimeError('aligned writer changed-input graph did not publish exact new bytes')
    if addresses != [x.data_ptr() for x in (*inputs, raw, key, value, tag, status)]:
        raise RuntimeError('aligned writer graph changed addresses')
    protected = [x.clone() for x in (raw, key, value, tag)]
    inputs[4][0] = block * blocks; inputs[5][1] = -1
    inputs[0][2, 0, 0] = float('nan'); inputs[0][-1].fill_(float('nan'))
    execute(); torch.npu.synchronize()
    writer_status = torch.zeros(n, h, dtype=torch.int32, device=device)
    writer_status[0] = 1; writer_status[1] = 4; writer_status[2, 0] = 2
    if not torch.equal(status, writer_status):
        raise RuntimeError('writer slot/position/nonfinite/padding exact status differs')
    if not all(_store_bits_equal(torch, a, b) for a, b in zip((raw, key, value, tag), protected)):
        raise RuntimeError('invalid writer row changed a cache payload/window')
    inputs[4].copy_(slots.to(device))
    # Rotation is the actual native torch BF16 matmul used by the external
    # provider. The Q copy must preserve its BF16 payload and strict statuses.
    rotated = torch.matmul(original_k.to(device), rotation.to(device))
    torch.testing.assert_close(rotated.float().cpu(), kr.float(), atol=.005, rtol=.005)
    qr = torch.empty(n, h, d, dtype=torch.bfloat16, device=device)
    qstatus = torch.empty(n, h, dtype=torch.int32, device=device)
    getattr(ops, QUERY_COPY)(rotated, inputs[4], qr, qstatus, cores)
    torch.npu.synchronize()
    expected_qr = rotated.clone(); expected_qr[-1].zero_()
    if not torch.equal(qr, expected_qr) or bool(qstatus.ne(0).any()):
        raise RuntimeError('BF16 rotated Q publication/status differs')
    rotated[0, 0, 0] = float('nan'); rotated[-1].fill_(float('nan'))
    getattr(ops, QUERY_COPY)(rotated, inputs[4], qr, qstatus, cores)
    torch.npu.synchronize(); want_status = torch.zeros_like(qstatus); want_status[0, 0] = 2
    if not torch.equal(qstatus, want_status) or bool(qr[-1].ne(0).any()):
        raise RuntimeError('Q nonfinite/padding error domain differs')
    return {'status': 'passed', 'writer': WRITER, 'query_copy': QUERY_COPY,
            'codec_bytes': 'exact', 'original_windows': 'exact', 'canaries': 'exact',
            'query_status': 'exact', 'writer_status': 'exact', 'rotation': 'BF16_operands_FP32acc_BF16_output',
            'graph_capture': 'passed', 'graph_replay': 'passed', 'graph_same_addresses': True}


def validate_evidence(report):
    required = {'status': 'passed', 'contract': CONTRACT, 'cache_format': CACHE_FORMAT,
                'slot_bytes': 144, 'device_completion': 'passed', 'precision': 'passed',
                'error_domain': 'passed', 'graph_capture': 'passed', 'graph_replay': 'passed',
                'performance': 'not_run_user_requested'}
    bad = [k for k, v in required.items() if report.get(k) != v]
    for field, path in (('acceptance_sha256', 'configs/acceptance_aligned_bf16.json'),
                        ('oracle_sha256', 'tools/aligned_reference.py'), ('probe_sha256', 'tools/probe_aligned_attention.py')):
        if report.get(field) != digest(ROOT / path):
            bad.append(field)
    if report.get('source_binding') != source_binding(): bad.append('source_binding')
    if report.get('operator_families') != OPERATORS: bad.append('operator_families')
    writer = report.get('writer_rotation', {})
    for key, value in {'status': 'passed', 'writer': WRITER, 'query_copy': QUERY_COPY,
                       'codec_bytes': 'exact', 'original_windows': 'exact', 'canaries': 'exact',
                       'query_status': 'exact', 'writer_status': 'exact', 'graph_capture': 'passed', 'graph_replay': 'passed',
                       'graph_same_addresses': True}.items():
        if writer.get(key) != value: bad.append('writer_rotation.' + key)
    rows, plan = report.get('cases', []), case_plan()
    if len(rows) != len(plan): return bad + ['complete_case_plan']
    for row, case in zip(rows, plan):
        if any(row.get(k) != v for k, v in {'case': case['shape'].name, 'family': case['family'],
                'operator': OPERATORS[case['family']], 'dequant_mode': case['dequant_mode'], 'status': 'passed'}.items()):
            bad.append(case['shape'].name + '.identity')
        tasks = (case['shape'].padded_tokens or sum(case['shape'].qlens)) * case['shape'].kv_heads * 3 * case['shape'].splits
        if row.get('all_tasks') != tasks or row.get('check', {}).get('status') != 'passed': bad.append(case['shape'].name + '.outputs')
        if row.get('timing') is not None: bad.append(case['shape'].name + '.timing')
        if case['graph']:
            graph = row.get('graph') or {}
            if (graph.get('capture') != 'passed' or graph.get('replay') != 'passed' or graph.get('same_addresses') is not True or
                    [x.get('state') for x in graph.get('states', [])] != ['original', 'changed_query', 'changed_context'] or
                    any(x.get('v3_oracle') != 'passed' or x.get('eager_replay_bitwise') != 'passed' for x in graph.get('states', []))):
                bad.append(case['shape'].name + '.graph')
    return bad


def run_npu(config_path=ROOT / 'configs/target.json'):
    import gc
    from . import probe_history_reuse as reuse
    from .aligned_reference import load_acceptance, ACCEPTANCE
    load_acceptance(); config_path = Path(config_path).resolve(); config = json.loads(config_path.read_text())
    if config.get('experimental_aligned_math') is not True or config.get('experimental_bf16_attention') is not True:
        raise RuntimeError('aligned gate requires experimental_aligned_math=True and experimental_bf16_attention=True')
    reuse._active_device(config)
    torch = require_real_npu(); torch.npu.set_device(0); torch.set_num_threads(min(4, torch.get_num_threads()))
    from .build_ops import normalize_soc
    if normalize_soc(torch.npu.get_device_name(0)) != config['soc_version']:
        raise RuntimeError('aligned target SoC differs from compiled contract')
    from oscar_ascend.ops.loader import validate_build_artifacts, require_capabilities
    manifest = validate_build_artifacts(ROOT / 'build/ascendc/build_manifest.json')
    require_capabilities(set(OPERATORS.values()) | {WRITER, QUERY_COPY, 'prepare_attention_tasks_out', 'merge_lse_out'})
    bindings, config_hash = source_binding(), digest(config_path)
    cores = reuse._core_count(torch, config); device = torch.device('npu:0')
    writer = writer_rotation_gate(torch, torch.ops.oscar_ascend_ops, device, cores)
    rows = []
    for case in case_plan():
        row = run_case(torch, torch.ops.oscar_ascend_ops, case, device, cores); rows.append(row)
        print('ALIGNED_PROBE ' + json.dumps({'case': row['case'], 'status': row['status']}), flush=True)
        gc.collect(); torch.npu.empty_cache()
    if bindings != source_binding() or digest(config_path) != config_hash:
        raise RuntimeError('aligned source/config changed during device gate')
    report = {'status': 'passed', 'contract': CONTRACT, 'cache_format': CACHE_FORMAT, 'slot_bytes': 144,
        'source_binding': bindings, 'effective_config_sha256': config_hash, 'operator_families': OPERATORS,
        'acceptance_sha256': digest(ACCEPTANCE), 'oracle_sha256': digest(ROOT / 'tools/aligned_reference.py'),
        'probe_sha256': digest(__file__), 'artifact_signature': manifest['signature'], 'artifact_sha256': manifest['sha256'],
        'writer_rotation': writer, 'cases': rows, 'device_completion': 'passed', 'precision': 'passed',
        'error_domain': 'passed', 'graph_capture': 'passed', 'graph_replay': 'passed',
        'performance': 'not_run_user_requested', 'model_quality': 'not_established'}
    bad = validate_evidence(report)
    if bad: report.update(status='failed', evidence_mismatches=bad)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    npu = sub.add_parser('npu'); npu.add_argument('--config', type=Path, default=ROOT / 'configs/target.json')
    npu.add_argument('--output', type=Path, required=True); args = parser.parse_args(argv)
    try:
        report = run_npu(args.config)
    except Exception as error:
        import traceback
        traceback.print_exc()
        report = {'status': 'failed', 'error': str(error), 'NPU': 'not_passed', 'contract': CONTRACT,
                  'cache_format': CACHE_FORMAT, 'performance': 'not_run_user_requested'}
    atomic_json(args.output, report)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
