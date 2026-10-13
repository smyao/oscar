"""Archive #126/#129/#140/#145/#148/#154-156 and complete startup D.4.

Same-input production-H16 versus existing H32/member-pipeline composition.
Only namespace/include/driver fixture bindings change; no operator math edit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from csrc.experiments.decode_phase_profile.generate import function

SOURCES = {
    'baseline': ('csrc/kernels/attention_cv_window_range_cluster4.cpp',
        '14b28f9181220432a15bcc3f23679e539473b12e01d858bbb00ff3345522ba65'),
    'candidate': ('csrc/kernels/attention_cv_member_pipeline_cluster4.cpp',
        '0795a07b1db0ba3d12e7ae762929b752d3bfb6d7475a925cf9e25e62ddeeb4bc'),
}
HARNESS = {
    'cv_probe.cpp': 'e79ff6d3219fbc73dc0dc4b63c7cc06129ede6f1bec4959b4325a61c0268d823',
    'cv_replay.py': '71c38e9dd2fb545de1cb4514ddf256984587c62fb3397560cbe73e7a44eb8091',
    'cv_compare.py': '9fb3b466e5e0a8e5ba4c2986abaed7af6443cd4d674495a7a2670816083e3dc1',
}
METHODS = ('Process', 'IsClusterMember', 'Matmul', 'CubeTask', 'CubeCluster',
           'VectorTask', 'VectorCluster', 'LoadQueries', 'LoadPacked', 'Unpack',
           'LoadWindowRuns', 'LoadPrecise', 'Softmax', 'RestoreState', 'SpillState',
           'PublishRows', 'PublishStatus', 'TaskValid')
HEADER = '''// GENERATED isolated complete C4 composition. Archive #126/#129/#140/#145/#148/#154-156.
// Full D.4 four questions: (1) fused history dequant+FIA and exact window.
// (2) Old full-history restore6499.8-6655.1ms and store209-215ms are forbidden.
// (3) Exact existing production kernels, bounded KV256 slots and FP32 math;
//     H16 versus H32+member pipeline, no history-sized tensor or changed ABI.
// (4) Complete Q336/C1341, all1008 tasks, source0+1/current suppressed:
//     >=1.15 complete CV speed is a predeclared gate, NOT a predicted pass.
'''


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def pair_sources():
    return {arm: digest for arm, (_, digest) in SOURCES.items()}


def source_dependencies():
    """Bind current repository headers transitively; SDK is a build input."""
    pending = [ROOT / path for path, _ in SOURCES.values()]
    found = {}
    while pending:
        path = pending.pop().resolve()
        key = str(path.relative_to(ROOT))
        if key in found: continue
        text = path.read_text(); found[key] = sha(text)
        for name in re.findall(r'^\s*#include\s+"([^"]+)"', text, re.MULTILINE):
            child = (path.parent / name).resolve()
            if child.is_file(): pending.append(child)
    return found


def transform(original, edits):
    text = original
    for before, after, count in edits:
        if text.count(before) != count:
            raise ValueError(('edit occurrence', before[:100], text.count(before), count))
        text = text.replace(before, after)
    restored = text
    for before, after, count in reversed(edits):
        if restored.count(after) != count: raise ValueError('inverse ambiguity ' + after[:100])
        restored = restored.replace(after, before)
    if restored != original: raise ValueError('full original source restoration failed')
    return text


def generate(candidate):
    arm = 'candidate' if candidate else 'baseline'
    name, digest = SOURCES[arm]
    original = (ROOT / name).read_text()
    if sha(original) != digest: raise ValueError('production mother changed: ' + name)
    stem = 'member_pipeline_cluster4' if candidate else 'window_range_cluster4'
    kernel = 'oscar_attention_cv_member_pipeline_cluster4_kernel' if candidate else 'oscar_attention_window_range_cluster4_kernel'
    edits = [(f'attention_cv_{stem}_launch', f'c4_composed_{arm}_launch', 1),
             (kernel, f'oscar_c4_composed_{arm}_kernel', 1),
             ('"../include/', '"../../include/', 3 if candidate else 2)]
    headers = ('oscar_common.h', 'attention_fast_unpack.h', 'attention_striped_unpack.h', 'attention_range_mask.h')
    if candidate: headers += ('attention_striped_unpack32.h',)
    for header in headers:
        edits.append((f'#include "{header}"', f'#include "../../kernels/{header}"',
                      2 if header == 'attention_fast_unpack.h' else 1))
    changed = transform(original, edits)
    hashes = {}
    for method in METHODS:
        before = original[slice(*function(original, method))]
        after = changed[slice(*function(changed, method))]
        if before != after: raise ValueError('method changed ' + method)
        hashes[method] = sha(before)
    final = HEADER + changed
    return final, {'arm': arm, 'symbol': f'oscar_c4_composed_{arm}_kernel',
        'source': name, 'actual_source_sha256': digest, 'source_sha256': SOURCES['baseline'][1],
        'sha256': sha(final), 'production_bytes_restored': True,
        'method_hashes': hashes, 'workspace_per_core': 2494464 if candidate else 1839104,
        'explicit_UB_per_AIV': 157760 if candidate else 157632}


def emit(name, edits):
    source = HERE.parent / 'c4_member_pipeline' / name
    original = source.read_text()
    if sha(original) != HARNESS[name]: raise ValueError('qualified harness drift ' + name)
    changed = transform(original, edits)
    return changed, {'template': str(source.relative_to(ROOT)), 'template_sha256': HARNESS[name],
                     'sha256': sha(changed), 'template_bytes_restored': True}


def artifacts():
    generated = {arm: generate(arm == 'candidate')[1]['sha256'] for arm in SOURCES}
    binding = '''  {std::ofstream binding(out/"source-binding.json");
    binding<<"{\\"arm\\":\\""<<arm<<"\\",\\"kernel_sha256\\":\\""
       <<(arm=="baseline"?"%s":"%s")<<"\\"}\\n";
    if(!binding.flush())throw std::runtime_error("source binding save");}
''' % (generated['baseline'], generated['candidate'])
    probe = emit('cv_probe.cpp', [
        ('oscar_c4_member_pipeline_', 'oscar_c4_composed_', 4),
        ('C4_MEMBER_PIPELINE_', 'C4_COMPOSED_', 2),
        ('bad!=-1&&bad!=80&&bad!=96&&bad!=336', 'bad!=-1', 1),
        ('(context!=113&&context!=337)', '(context!=1341)', 1),
        ('  auto taskBefore=std::vector<uint8_t>(tasks.p,tasks.p+tasks.n);',
         '  if(Read(dir/"expected_tasks.bin",1008*128)!=std::vector<uint8_t>(tasks.p,tasks.p+tasks.n))throw std::runtime_error("source task formula differs from real prepare");\n  auto taskBefore=std::vector<uint8_t>(tasks.p,tasks.p+tasks.n);', 1),
        ('  std::ofstream report(out/"report.json");', binding + '  std::ofstream report(out/"report.json");', 1),
    ])
    replay = emit('cv_replay.py', [
        ('from csrc.experiments.c4_member_pipeline import generate as generate_c4',
         'from csrc.experiments.c4_composed_long import generate as generate_c4', 1),
        ('bad not in (-1,80,96,336) or bad>=context or context not in (113,337)',
         'bad!=-1 or context!=1341', 1),
        ('real complete Q336/C113 CPU metadata/evidence required', 'real complete Q336/C1341 CPU metadata/evidence required', 1),
        ("    out=a.output.resolve();", "    if a.library.name!='liboscar_c4_composed_long.so':raise ValueError('explicit composed same-library pair required')\n    if json.loads((a.library.parent.parent/'source_binding.json').read_text())!=generate_c4.build_binding():raise ValueError('build source binding differs')\n    out=a.output.resolve();", 1),
        ("    inputs={name:(a.fixture/(name+'.bin')).read_bytes()", "    expected_binding={'arm':a.variant,'kernel_sha256':identity['sha256']}\n    if receipt.get('arm')!=a.variant or json.loads((a.prepared/'source-binding.json').read_text())!=expected_binding:raise ValueError('CPU reference does not bind this production arm')\n    inputs={name:(a.fixture/(name+'.bin')).read_bytes()", 1),
        ("    config=Path(os.environ['CAMODEL_CONFIG_PATH']);", "    if inputs['tasks']!=(a.fixture/'expected_tasks.bin').read_bytes():raise ValueError('CPU tasks differ from frozen host formula')\n    config=Path(os.environ['CAMODEL_CONFIG_PATH']);", 1),
        ("'H32_source_sha256':identity['H32_source_sha256']", "'pair_sources':generate_c4.pair_sources()", 1),
    ])
    compare = emit('cv_compare.py', [
        ('import struct\n', 'import struct\nimport sys\nROOT=Path(__file__).resolve().parents[3]\nif str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))\nfrom csrc.experiments.c4_composed_long import generate\n', 1),
        ("report['context'] not in (113,337)", "report['context']!=1341", 1),
        ("    hashes={}\n", "    deadline=900 if 'outputs' in report else 120\n    if host.get('status')!='completed' or host.get('supervisor_returncode')!=0 or host['timeout_seconds']!=deadline:raise ValueError('declared completion/deadline gate')\n    if 'outputs' not in report and json.loads((path/'source-binding.json').read_text())!={'arm':arm,'kernel_sha256':generate.generate(arm=='candidate')[1]['sha256']}:raise ValueError('CPU production source binding')\n    if 'outputs' in report and report['generated_sha256']!=generate.generate(arm=='candidate')[1]['sha256']:raise ValueError('CAModel source binding')\n    hashes={}\n", 1),
        ("'H32_source_sha256'", "'pair_sources'", 1),
        ("        result.update(old_ticks=old['task_ticks'],new_ticks=new['task_ticks'],old_over_new=old['task_ticks']/new['task_ticks'])", "        result.update(old_ticks=old['task_ticks'],new_ticks=new['task_ticks'],old_over_new=old['task_ticks']/new['task_ticks'])\n        result.update(minimum_complete_CV_speedup=1.15,combination_gate='passed' if result['old_over_new']>=1.15 else 'failed',denominator='current_production_H16',candidate='existing_H32_plus_member_pipeline')", 1),
        ("a.output.write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))", "a.output.write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))\n    raise SystemExit(2 if r.get('combination_gate')=='failed' else 0)", 1),
    ])
    return {'cv_probe.cpp': probe, 'cv_replay.py': replay, 'cv_compare.py': compare}


def build_binding():
    return {'schema': 1, 'pair_sources': pair_sources(),
        'source_dependencies': source_dependencies(),
        'sources': {f'c4_{arm}.cpp': generate(arm == 'candidate')[1]['sha256'] for arm in SOURCES},
        'harness': {name: value[1]['sha256'] for name, value in artifacts().items()},
        'generator_sha256': sha(Path(__file__).read_bytes())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true'); p.add_argument('--build-binding', type=Path)
    args = p.parse_args(argv)
    files, arms = {}, {}
    for arm in SOURCES: files[f'c4_{arm}.cpp'], arms[arm] = generate(arm == 'candidate')
    harness = artifacts()
    for name, (text, _) in harness.items(): files[name] = text
    manifest = {'scope': 'Q336_C1341_all1008_tasks_source0_source1_legal_current_suppression',
        'arms': arms, 'harness': {name: value[1] for name, value in harness.items()},
        'minimum_complete_CV_speedup': 1.15, 'source_sha256': pair_sources(),
        'source_dependencies': source_dependencies(),
        'validation': {k: 'not_run' for k in ('CPU', 'CANN', 'CAModel', 'NPU', 'graph', 'whole_service')}}
    files['manifest.json'] = json.dumps(manifest, indent=2) + '\n'
    for name, text in files.items():
        path = HERE / name
        if args.check:
            if path.read_text() != text: raise ValueError('stale isolated source ' + name)
        else: path.write_text(text)
    if args.build_binding:
        if not args.check: raise ValueError('build binding requires --check')
        args.build_binding.write_text(json.dumps(build_binding(), indent=2) + '\n')
    print(json.dumps({'status': 'checked' if args.check else 'generated',
        'production_unchanged': True, 'device_execution': 'not_run', 'gate': 1.15}))


if __name__ == '__main__': main()
