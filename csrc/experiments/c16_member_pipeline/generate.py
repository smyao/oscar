"""Archive #126/#129/#145/#148/#154-157; D.4 four questions in generated source.

Extend the qualified H32/two-slot member pipeline to the actual C16 route.
Preserve the stable C16 task ownership and eligibility; freeze both mothers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from csrc.experiments.decode_phase_profile.generate import function

BASE = ROOT / 'csrc/kernels/attention_cv_window_range_cluster16.cpp'
PIPE = ROOT / 'csrc/kernels/attention_cv_member_pipeline_cluster4.cpp'
HEADER = '''// GENERATED C16 H32 MEMBER PIPELINE. Archive #126/#129/#145/#148/#154-157.
// D.4 four questions (complete original log reread):
// 1. Fused history dequant/FIA: sixteen members share one bounded KV256 tile.
// 2. Old full-history dequant6499.8-6655.1ms, host725ms and store209-215ms
//    must never return; CPU parity does not establish asynchronous safety.
// 3. Preserve stable C16 eligibility/ownership and every member's FP32/K128
//    reduction, errors, masks, and final rotation. The qualified C4 two-slot
//    pipeline extends to16 members without new flags: QK(i+2) follows PV(i)
//    before reusing P's slot. Atomic complete-K accumulation precedes the
//    next RestoreState; next-KV writes a separate fixed buffer. H32 retains
//    the original alternating16-row error ownership. No history-sized tensor.
// 4. Adds655360B/core atD256 (total5652480B/core), no context-sized reserve;
//    numerical/bitwise/full-task and full-operator pairing decide benefit.
//    Old C4 speed and D.4 timings are not C16 or model speed evidence.
'''


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def generate(candidate, production=False):
    base, pipeline = BASE.read_text(), PIPE.read_text()
    pins = json.loads((HERE / 'pins.json').read_text())
    if sha(base) != pins['stable_c16'] or sha(pipeline) != pins['qualified_c4_pipeline']:
        raise ValueError('frozen C16/C4 mother source changed')
    original = pipeline if candidate else base
    text = original
    edits = []
    def edit(old, new, count=1):
        nonlocal text
        if text.count(old) != count: raise ValueError(('edit identity', old[:100], text.count(old), count))
        text = text.replace(old, new); edits.append((old, new, count))
    if candidate:
        for name in ('Process', 'IsClusterMember'):
            edit(text[slice(*function(text, name))], base[slice(*function(base, name))])
        edit('attention_cluster4_workspace_per_core(D)', 'attention_cluster16_workspace_per_core(D)')
        edit('kOffset=4*kQueryRows*D', 'kOffset=16*kQueryRows*D')
        edit('statStateOffset=accStateOffset+4*kQueryRows*D', 'statStateOffset=accStateOffset+16*kQueryRows*D')
        edit('member<4', 'member<16', 5)
        edit('member+2<4', 'member+2<16')
        edit('memberErrors[4]', 'memberErrors[16]')
        edit('Four Q tiles', 'Sixteen Q tiles')
        edit('and four online states', 'and sixteen online states')
        edit('// QK0,QK1,PV0,QK2,PV1,QK3,PV2,PV3.', '// QK0,QK1,PV0,QK2,...,PV14,PV15.')
        edit('// P3 and all alpha states are published.', '// P15 and all alpha states are published.')
        edit('current PV3.', 'current PV15.')
        # Exact stable C16 scheduling and cluster domain are independently bound.
        for name in ('Process', 'IsClusterMember'):
            if text[slice(*function(text, name))] != base[slice(*function(base, name))]:
                raise ValueError('C16 task/domain drift ' + name)
    arm = 'candidate' if candidate else 'baseline'
    old_launch = ('attention_cv_member_pipeline_cluster4_launch' if candidate else
                  'attention_cv_window_range_cluster16_launch')
    old_kernel = ('oscar_attention_cv_member_pipeline_cluster4_kernel' if candidate else
                  'oscar_attention_window_range_cluster16_kernel')
    launch = 'attention_cv_member_pipeline_cluster16_launch' if production else 'c16_member_pipeline_' + arm + '_launch'
    kernel = 'oscar_attention_cv_member_pipeline_cluster16_kernel' if production else 'oscar_c16_member_pipeline_' + arm + '_kernel'
    edit(old_launch, launch); edit(old_kernel, kernel)
    if candidate and production:
        edit('"../include/oscar_member_pipeline_launch.h"', '"../include/oscar_c16_member_pipeline_launch.h"')
    if not production:
        for header in ('attention_fast_unpack.h', 'attention_striped_unpack.h',
                       'attention_range_mask.h', 'oscar_common.h'):
            edit('#include "' + header + '"', '#include "../../kernels/' + header + '"',
                 2 if header == 'attention_fast_unpack.h' else 1)
        if candidate:
            edit('#include "attention_striped_unpack32.h"', '#include "../../kernels/attention_striped_unpack32.h"')
        edit('"../include/', '"../../include/', text.count('"../include/'))
    restored = text
    for old, new, count in reversed(edits):
        if restored.count(new) != count: raise ValueError('inverse edit drift: ' + new[:80])
        restored = restored.replace(new, old)
    if restored != original: raise ValueError('mother bytes not restored')
    unchanged = {}
    for name in ('Matmul', 'MatmulAtomic', 'LoadQueries', 'LoadPacked', 'Unpack',
                 'PublishHistoryKv', 'LoadWindowRuns', 'LoadPrecise', 'PublishKv',
                 'Softmax', 'VectorTask', 'SpillState', 'RestoreState', 'LoadClusterKv',
                 'PublishRows', 'PublishEmpty', 'PublishStatus', 'TaskValid') if candidate else ():
        before = original[slice(*function(original, name))]
        after = text[slice(*function(text, name))]
        if before != after: raise ValueError('qualified math/domain changed ' + name)
        unchanged[name] = sha(before)
    text = (HEADER if candidate else '// Stable C16 baseline; archive #126/#145/#154 and D.4.\n') + text
    return text, {'arm': arm, 'symbol': kernel, 'source_sha256': sha(base),
        'H32_source_sha256': sha(pipeline), 'generated_sha256': sha(text),
        'mother_bytes_restored': True, 'unchanged_qualified_methods': unchanged,
        'workspace_per_core': 5652480 if candidate else 4997120,
        'explicit_UB_per_AIV': 157760 if candidate else 157632,
        'production': production, 'validation': 'not_executed'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true')
    p.add_argument('--production', action='store_true'); a = p.parse_args()
    manifest = {}
    for candidate in (False, True):
        source, record = generate(candidate)
        path = HERE / ('c16_' + record['arm'] + '.cpp')
        if a.check:
            if path.read_text() != source: raise ValueError('generated source drift')
        else: path.write_text(source)
        manifest[record['arm']] = record
    content = json.dumps(manifest, indent=2) + '\n'
    if a.check:
        if (HERE / 'manifest.json').read_text() != content: raise ValueError('manifest drift')
    else: (HERE / 'manifest.json').write_text(content)
    if a.production:
        source, record = generate(True, production=True)
        (ROOT / 'csrc/kernels/attention_cv_member_pipeline_cluster16.cpp').write_text(source)
    print(json.dumps({'status': 'checked' if a.check else 'generated', 'CANN': 'not_run', 'NPU': 'not_run'}))


if __name__ == '__main__': main()
