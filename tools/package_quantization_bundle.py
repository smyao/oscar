"""Archive #13-22/#70-73/#126/#145/#148/#154-156 and startup D.4.

Map the measured Init+Batch8 rotation and unchanged SIMD-pack+Init writer into
existing production symbols. Only include/symbol/banner relocation is allowed.
Local integration does not rewrite historical promotion decisions or claim a
fresh full build, target-NPU, graph, model-quality or service speed result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.package_request_window_kernels import package_packed_writer, relocated_helper
from csrc.experiments.request_windows_store.generate import generate as generate_packed

PREFIX='csrc/experiments/hadamard_batch8/'
OLD_CANONICAL='csrc/kernels/rotate_clip_store.cpp'
OLD_STRIPED='csrc/kernels/rotate_clip_store_striped.cpp'
SNAPSHOT=ROOT/'artifacts/quantization_bundle_baseline_20261011'
REPORT=ROOT/'reports/query_batch8_reopen_20261015/integration.json'
EVIDENCE=ROOT/'reports/hadamard_batch8_20261010/reports'
QUERY_REPORT=EVIDENCE/'batch8-rotate-camodel128-v2/batch8/report.json'
WRITER_REPORT=EVIDENCE/'batch8-writer-camodel128/current/report.json'
WRITER_COMPARISON=ROOT/'reports/quant_store_simd_pack_20261010/reports/store-camodel-n128/comparison.json'
WRITER_SOURCE='csrc/experiments/striped_store_simd_pack/writer_pack_init.cpp'
PINS={
    PREFIX+'rotate_batch.cpp':'8f34370289fe77e357e38ffe757b9076519b96c7167c30378df4f40b908c1302',
    WRITER_SOURCE:'7cbb1b1205c380afcaa3893f23d9ef1337f468c13f7c437c2af8ef91bf123a17',
}
MEASURED_HEADER=''.join((ROOT/(PREFIX+'rotate_batch.cpp')).read_text().splitlines(keepends=True)[:6])
QUERY_BANNER="""// Production candidate: exact Hadamard Init + Batch8; archive #13-22/#126/#145/#154-156.
// D.4: bounded rotate phase, never full-history recovery6499.8-6655.1ms.
// Every row retains FP32 Gather/Mul/Add order; eight rows share vector fences.
// User2026-10-15 accepts complete net gains: Init68082 -> Batch8 64537 ticks.
// Fixed Hadamard UB increases7KiB/core; no GM/ABI change, writer unchanged.
// Historical CPU/CAModel qualification is local; fresh full build/NPU/graph pending.
"""
MAPS=(
    (PREFIX+'rotate_batch.cpp','csrc/kernels/rotate_vector_init.cpp',(
        (MEASURED_HEADER,QUERY_BANNER),
        ('#include "../../kernels/oscar_common.h"','#include "oscar_common.h"'),
        ('#include "../hadamard_indices/vector_indices.h"','#include "hadamard_indices.h"'),
        ('#include "../../include/oscar_rotation_launch.h"','#include "../include/oscar_rotation_vector_init_launch.h"'),
        ('oscar_rotate_batch8_kernel','oscar_rotate_vector_init_kernel'),
        ('oscar_store_batch8_aux_kernel','oscar_store_vector_init_aux_kernel'),
        ('rotate_batch8_launch','rotate_vector_init_launch'),
        ('rotate_clip_store_batch8_aux_launch','rotate_clip_store_vector_init_aux_launch'),
    )),
    (WRITER_SOURCE,OLD_STRIPED,(
        ('// Independent C arm: SIMD packing plus exact Hadamard Init; Transform unchanged.',
         '// Candidate writer: qualified SIMD pack + exact Init; NPU/graph pending.\n// Independent C arm: SIMD packing plus exact Hadamard Init; Transform unchanged.'),
        ('#include "../../kernels/oscar_common.h"','#include "oscar_common.h"'),
        ('#include "pack.h"','#include "striped_store_pack.h"'),
        ('#include "../hadamard_indices/vector_indices.h"','#include "hadamard_indices.h"'),
        ('#include "../../include/oscar_rotation_launch.h"','#include "../include/oscar_rotation_launch.h"'),
        ('rotate_clip_store_striped_simd_pack_init','rotate_clip_store_striped'),
        ('rotate_striped_simd_pack_init_aux','rotate_striped_aux'),
    )),
)


def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def evidence():
    query=json.loads(QUERY_REPORT.read_text());writer=json.loads(WRITER_REPORT.read_text())
    if any(r['status']!='passed' or r['failures'] for r in (query,writer)):
        raise ValueError('measured arm lacks successful evidence')
    if query['source_binding']!=writer['source_binding'] or query['library_sha256']!=writer['library_sha256']:
        raise ValueError('measured arms did not use the same frozen source/library')
    binding=query['source_binding']
    for source,pin in PINS.items():
        if sha((ROOT/source).read_bytes())!=pin or binding['sources'][source]!=pin:
            raise ValueError('measured experiment source drift: '+source)
    for source,pin in {**binding['dependencies'],**binding['frozen_writer_binding']['dependencies']}.items():
        if sha((ROOT/source).read_bytes())!=pin:raise ValueError('measured dependency drift: '+source)
    for src,dst in (('csrc/experiments/striped_store_simd_pack/pack.h','csrc/kernels/striped_store_pack.h'),
                    ('csrc/experiments/hadamard_indices/vector_indices.h','csrc/kernels/hadamard_indices.h')):
        if (ROOT/dst).read_text()!=relocated_helper((ROOT/src).read_text()):raise ValueError('shared helper drift')
    return binding


def mapped(source,edits):
    original=(ROOT/source).read_text();result=original;counts=[]
    for old,new in edits:
        count=result.count(old)
        expected=3 if old in ('rotate_clip_store_striped_simd_pack_init','rotate_striped_simd_pack_init_aux') else (2 if old.endswith('_kernel') else 1)
        if count!=expected:raise ValueError(f'non-unique mapping {old}: {count}!={expected}')
        result=result.replace(old,new);counts.append(count)
    restored=result
    for (old,new),count in reversed(list(zip(edits,counts))):
        if restored.count(new)!=count:raise ValueError('ambiguous reverse mapping '+new)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('source mapping is not byte reversible')
    return result,counts


def freeze(binding,check):
    manifest_path=SNAPSHOT/'manifest.json'
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text())
        if manifest['measured_binding']!=binding:raise ValueError('frozen measured binding changed')
        for item in manifest['files']:
            if sha((SNAPSHOT/item['snapshot']).read_bytes())!=item['sha256']:raise ValueError('baseline snapshot changed')
        return manifest
    if check:raise ValueError('baseline snapshot has not been created')
    canonical=(ROOT/OLD_CANONICAL).read_bytes()
    if sha(canonical)!=binding['sources'][OLD_CANONICAL]:raise ValueError('cannot freeze changed canonical baseline')
    # The current writer must still be the recorded SIMD pack+Init arm before
    # the first bundle installation; this is historical identity, not new routing.
    measured_writer_path='csrc/experiments/striped_store_simd_pack/writer_pack_init.cpp'
    measured_writer=(ROOT/measured_writer_path).read_text()
    if sha(measured_writer)!=binding['sources'][measured_writer_path]:
        raise ValueError('frozen writer baseline changed')
    historical=measured_writer
    for old,new in (('#include "../../kernels/oscar_common.h"','#include "oscar_common.h"'),
                    ('#include "pack.h"','#include "striped_store_pack.h"'),
                    ('#include "../hadamard_indices/vector_indices.h"','#include "hadamard_indices.h"'),
                    ('#include "../../include/oscar_rotation_launch.h"','#include "../include/oscar_rotation_launch.h"'),
                    ('rotate_clip_store_striped_simd_pack_init','rotate_clip_store_striped'),
                    ('rotate_striped_simd_pack_init_aux','rotate_striped_aux')):
        historical=historical.replace(old,new)
    historical='// Candidate writer: qualified SIMD pack + exact Init; NPU/graph pending.\n'+historical
    if (ROOT/OLD_STRIPED).read_text()!=historical:raise ValueError('cannot freeze changed striped baseline')
    SNAPSHOT.mkdir(parents=True,exist_ok=False)
    items=[]
    for source in (OLD_CANONICAL,OLD_STRIPED):
        data=(ROOT/source).read_bytes();name=Path(source).name
        (SNAPSHOT/name).write_bytes(data)
        items.append({'source_at_measurement':source,'snapshot':name,'sha256':sha(data)})
    manifest={'scope':'original source bytes for replay in the frozen historical source tree',
              'files':items,'measured_binding':binding,
              'replay_note':'Old hadamard_batch8 generate/check/CMake/fixtures require these originals at their original source-tree paths. Current production is a new bundle; old source-drift checks should fail here. Use a separate frozen research source tree, never overwrite live production for replay.'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def package(*,check=False):
    binding=evidence();prepared=[(s,t,*mapped(s,e)) for s,t,e in MAPS]
    snapshot=freeze(binding,check)
    if sha((ROOT/OLD_CANONICAL).read_bytes())!=binding['sources'][OLD_CANONICAL]:
        raise ValueError('Frozen FE0 canonical changed; never re-pin it for this candidate')
    records=[]
    for source,target,result,counts in prepared:
        if check:
            if (ROOT/target).read_text()!=result:raise ValueError('production source differs: '+target)
        else:(ROOT/target).write_text(result)
        records.append({'experiment':source,'experiment_sha256':sha((ROOT/source).read_bytes()),
                        'production':target,'production_sha256':sha(result),
                        'changes':'only named include/symbol/banner mappings','mapping_counts':counts,
                        'reverse_mapping_full_bytes_identical':True,'ABI':'measured Init+Batch8 signature retained; original rotate_out/legacy store preserved' if target.endswith('/rotate_vector_init.cpp') else 'existing writer symbols/signatures unchanged'})
    packed=generate_packed(check=check)
    packaged=package_packed_writer(check=check)
    report={'status':'local_source_bundle_integrated_not_fresh_device_acceptance',
            'mapping':records,'request_packed_generation':packed,'request_packed_mapping':packaged,
            'measured_source_binding':binding,
            'historical_evidence':{
                'query':{'report':str(QUERY_REPORT.relative_to(ROOT)),'sha256':sha(QUERY_REPORT.read_bytes()),
                         'same_library_ticks':{'old':110343,'init':68082,'batch8':64537},'init_over_batch8':68082/64537,'scope':'complete N128/H6/D256 query rotation only; not a model percentage'},
                'writer':{'report':str(WRITER_REPORT.relative_to(ROOT)),'sha256':sha(WRITER_REPORT.read_bytes()),
                          'same_library_ticks':{'old':173419,'pack_init':96566},'old_over_pack_init':173419/96566,
                          'full_writer_comparison':str(WRITER_COMPARISON.relative_to(ROOT)),
                          'comparison_sha256':sha(WRITER_COMPARISON.read_bytes())},
                'scope':'historical CPU/CAModel evidence retained verbatim; primitive ratios are not combined into service speed'},
            'frozen_baseline':{'directory':str(SNAPSHOT.relative_to(ROOT)),'manifest_sha256':sha((SNAPSHOT/'manifest.json').read_bytes()),
                               'files':snapshot['files'],'old_scripts_in_current_tree':'expected explicit source drift; replay requires frozen source tree'},
            'activation':'shared experimental_quantization_bundle internal candidate flag selects rotate_vector_init_out in both entrypoints; original rotate_out/legacy store remain baseline; striped/request-packed bundle retained; no new user option',
            'canonical_fe0_sha256':sha((ROOT/OLD_CANONICAL).read_bytes()),
            'correction':'initial local canonical replacement was rejected by frozen FE0 source gate; original bytes restored, optimization moved to independent optional operator',
            'scope_note':'independent rotate_vector_init.cpp retains an unused auxiliary store symbol; no torch registration or route selects it; canonical fused store remains frozen',
            'new_selection_20261015':{'query_batch8_beyond_init':1.0549297302322698,'extra_Hadamard_UB_per_core_bytes':7168,'extra_GM_bytes':0,'decision':'user waived fixed per-point15percent; query Batch8 accepted; writer Batch8 remains excluded','historical_reports':'retained without rewriting their original decisions'},
            'fresh_full_build':'pending','fresh_CPU_integration':'pending','NPU':'pending',
            'graph_capture':'pending','graph_replay':'pending','full_model_quality':'pending','service_performance':'pending'}
    if not check:
        REPORT.parent.mkdir(parents=True,exist_ok=True)
        REPORT.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');a=parser.parse_args()
    result=package(check=a.check)
    print(json.dumps({'status':result['status'],'check_only':a.check,'report':str(REPORT),'NPU':'pending','graph':'pending'}))
