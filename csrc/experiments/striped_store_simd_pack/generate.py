"""Archive G25-34/#13-22/#70-73/#126/#145/#151/#154-156, D.4.

Independent three-arm writer experiment. Only packing changes in B; C also
uses the separately verified integer Hadamard-index initializer. No rotation,
clipping, half metadata, quantizer, slot/window/error order is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=HERE/'writer_baseline.cpp'
PRODUCTION=ROOT/'csrc/kernels/rotate_clip_store_striped.cpp'
FROZEN_SHA='674b6115f93b1c8387bdc8ba8440218c7b1e78e3563cb4f76e37624e0eae62eb'
SOURCE_SHA='0515fd2a527812dc6e7c09e46ebae50fd19be62e7645705c8caffbb132a31d29'


def replace(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError('source drift: '+old[:90])
    return text.replace(old,new)


def writers():
    original=frozen_original()
    begin=original.index('    // Eight lanes at the same column')
    end=original.index('    auto words=meta.ReinterpretCast<uint16_t>();',begin)
    changed=original[:begin]+'''    oscar_striped_store_simd_pack::PackCodes(
        codes,work,packed,dim_,packedOffset);
'''+original[end:]
    changed=replace(changed,'#include "oscar_common.h"','#include "../../kernels/oscar_common.h"\n#include "pack.h"')
    changed=replace(changed,'#include "../include/oscar_rotation_launch.h"','#include "../../include/oscar_rotation_launch.h"')
    pack=changed.replace('rotate_clip_store_striped','rotate_clip_store_striped_simd_pack').replace('rotate_striped_aux','rotate_striped_simd_pack_aux')
    pack='// Independent B arm: only code packing changed; rotation Init is original.\n'+pack
    start=changed.index('      int32_t stage=0;')
    finish=changed.index('      Fence<HardEvent::S_V>();',start)+len('      Fence<HardEvent::S_V>();')
    both=changed[:start]+'''      oscar_hadamard_indices_trial::Build(indices,signs,
          inputBuf_.template Get<int32_t>(),workBuf_.template Get<int32_t>(),
          redBuf_.template Get<int32_t>(),dim_);'''+changed[finish:]
    both=both.replace('#include "pack.h"','#include "pack.h"\n#include "../hadamard_indices/vector_indices.h"')
    both=both.replace('rotate_clip_store_striped','rotate_clip_store_striped_simd_pack_init').replace('rotate_striped_aux','rotate_striped_simd_pack_init_aux')
    both='// Independent C arm: SIMD packing plus exact Hadamard Init; Transform unchanged.\n'+both
    return {'writer_pack.cpp':pack,'writer_pack_init.cpp':both}


def frozen_manifest():
    manifest=json.loads((HERE/'frozen_sources.json').read_text())
    if (manifest['original_sha256']!=SOURCE_SHA or manifest['frozen_sha256']!=FROZEN_SHA or
            manifest['frozen_source']!='writer_baseline.cpp'):
        raise ValueError('frozen baseline identity changed')
    return manifest


def frozen_original():
    manifest=frozen_manifest();data=SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest()!=FROZEN_SHA:raise ValueError('frozen baseline bytes changed')
    original=data.decode()
    for relocation in manifest['include_relocations']:
        original=replace(original,relocation['frozen'],relocation['original'])
    if hashlib.sha256(original.encode()).hexdigest()!=SOURCE_SHA:
        raise ValueError('include relocation changed original baseline body')
    return original


def driver():
    # This harness was measured together with these exact three kernels.
    # It is not regenerated from the mutable active_identity experiment.
    return (HERE/'store_cpu.cpp').read_text()


def source_binding():
    manifest=frozen_manifest();frozen_original()
    for name,wanted in manifest['measured_research_sources'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=wanted:
            raise ValueError('measured research source changed: '+name)
    for name,wanted in manifest['dependencies'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=wanted:
            raise ValueError('frozen research dependency changed: '+name)
    return {'scope':manifest['scope'],'manifest_sha256':hashlib.sha256((HERE/'frozen_sources.json').read_bytes()).hexdigest(),
            'original_sha256':SOURCE_SHA,'frozen_sha256':FROZEN_SHA,
            'research_sources':manifest['measured_research_sources'],'dependencies':manifest['dependencies']}


def verify_production():
    # Qualification of a new production mapping is deliberately separate
    # from replaying the frozen experiment. Never regenerate A/B/C from it.
    banner='// Candidate writer: qualified SIMD pack + exact Init; NPU/graph pending.\n'
    production=PRODUCTION.read_text()
    if not production.startswith(banner):raise ValueError('production qualification banner differs')
    production=production[len(banner):]
    for old,new in (('#include "oscar_common.h"','#include "../../kernels/oscar_common.h"'),
                    ('#include "striped_store_pack.h"','#include "pack.h"'),
                    ('#include "hadamard_indices.h"','#include "../hadamard_indices/vector_indices.h"'),
                    ('#include "../include/oscar_rotation_launch.h"','#include "../../include/oscar_rotation_launch.h"')):
        production=replace(production,old,new)
    production=production.replace('rotate_clip_store_striped','rotate_clip_store_striped_simd_pack_init').replace('rotate_striped_aux','rotate_striped_simd_pack_init_aux')
    if production!=(HERE/'writer_pack_init.cpp').read_text():
        raise ValueError('production writer differs from qualified frozen C arm')


def check():
    source_binding()
    expected={**writers(),'store_cpu.cpp':driver()}
    for name,text in expected.items():
        if not (HERE/name).is_file() or (HERE/name).read_text()!=text:raise ValueError('generated source differs: '+name)
    return expected


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--frozen-only',action='store_true',help='reproduce fixed A/B/C independently of later production changes')
    parser.add_argument('--manifest',type=Path,help='write the verified frozen source identity for this research build')
    a=parser.parse_args()
    if a.check:
        check()
        if not a.frozen_only:verify_production()
        print('frozen writer A/B/C verified'+('' if a.frozen_only else '; production C-arm mapping verified')+'; no execution')
    else:
        # Regeneration uses only A at SOURCE_SHA; B/C must keep their measured
        # hashes. A changed production source is never a new old baseline.
        manifest=frozen_manifest()
        for name,text in writers().items():
            if hashlib.sha256(text.encode()).hexdigest()!=manifest['measured_research_sources'][name]:
                raise ValueError('regeneration changes measured research arm: '+name)
            (HERE/name).write_text(text)
        check()
        print('fixed B/C writer sources restored from frozen baseline; no execution')
    if a.manifest:
        a.manifest.write_text(json.dumps(source_binding(),indent=2)+'\n')
