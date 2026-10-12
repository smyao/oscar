"""Extend the proven window-DMA algorithm and range mask to current C4/C16.

The mixed/prefill kernels are separate current production sources; a q4/q1
SIMD speedup must not be assigned to them. Clone each actual striped wrapper
body without modifying the production originals. History reuse, FP32 math,
group statistics, producer checks and task domain remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_phase_profile.generate import function
OUT=Path(__file__).resolve().parent


def full_method(text,name):
    a,b=function(text,name)
    return text[text.rfind('\n',0,a)+1:b]


def main():
    window=(ROOT/'csrc/experiments/window_batch_decode/window_batch_decode.cpp').read_text()
    replacement=full_method(window,'LoadWindowRuns')+'\n'+full_method(window,'LoadPrecise')
    source_hashes={}
    for suffix in ('','_balanced','_cluster4','_cluster16'):
        name='attention_cv_fast'+suffix
        source=ROOT/'csrc/kernels'/f'{name}.cpp';text=source.read_text()
        source_hashes[str(source.relative_to(ROOT))]=hashlib.sha256(text.encode()).hexdigest()
        def edit(old,new,count=1):
            nonlocal text
            assert text.count(old)==count,(name,old[:60],text.count(old),count)
            text=text.replace(old,new)
        edit(full_method(text,'LoadPrecise'),replacement)
        edit('''        Fence<HardEvent::S_V>();
        for(int32_t j=0;j<kKvRows;++j) {
          const bool visible=(j>=lo0&&j<hi0)||(j>=lo1&&j<hi1);
          if(!visible)scores.SetValue(r*kKvRows+j,kNegativeInfinity);
        }
        Fence<HardEvent::S_V>();''',
             '''        oscar_range_mask::MaskComplement(scores[r*kKvRows],lo0,hi0,lo1,hi1);
        Fence<HardEvent::V_S>();''')
        edit('#include "oscar_common.h"','#include "../../kernels/oscar_common.h"\n#include "../range_mask/range_mask.h"')
        edit('#include "attention_fast_unpack.h"','#include "../../kernels/attention_fast_unpack.h"')
        # All currently referenced project headers are preserved, only relative
        # include resolution changes for this isolated copy.
        include_count=text.count('"../include/')
        assert include_count>=1
        edit('"../include/','"../../include/',include_count)
        stem='base' if not suffix else suffix[1:]
        (OUT/f'{stem}_body.cpp').write_text(text)
        header='''// Isolated actual striped candidate clone. Archive #126/#129/#140/#145/#148/#151/#154/#155.
// D.4: full restore cost6499.8-6655.1ms is not repeated. This changes bounded
// window reads and score-mask writes only, with original FP32 history math,
// source/group boundaries and statuses. No current q4 gain is assumed here.
#include "../../kernels/attention_fast_unpack.h"
#include "../../kernels/attention_striped_unpack.h"
#define oscar_ascend_fast oscar_ascend_striped
#define OSCAR_STRIPED_D256_ONLY 1
'''
        wrapper=(header+f'#define {name}_launch attention_cv_window_range{suffix}_launch\n'
            +f'#define oscar_{name}_kernel oscar_attention_window_range{suffix}_kernel\n'
            +f'#include "{stem}_body.cpp"\n#undef OSCAR_STRIPED_D256_ONLY\n')
        (OUT/f'{stem}.cpp').write_text(wrapper)
    (OUT/'source_identity.json').write_text(json.dumps({'sources':source_hashes,
        'window_source_sha256':hashlib.sha256(window.encode()).hexdigest(),
        'production_edited':False,'npu':'not_run','performance':'not_established'},indent=2)+'\n')
    print('PREFILL_WINDOW_RANGE_GENERATED separate current base/balanced/C4/C16 paths')


if __name__=='__main__':main()
