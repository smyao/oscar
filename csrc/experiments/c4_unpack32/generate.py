"""Archive #13-22/#70-73/#126/#144-145/#148/#151/#154-156; full startup D.4.

Isolated 32-row extension of the CURRENT padded decoder helper. No production
edit. Preserve INT2 levels, FP16 metadata decode, FP32 Mul then Add and guards.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PINS={
 'csrc/kernels/attention_striped_unpack_padded.h':'1551af143666d7465352426dbbe7024af8a86d81ee12d7ccfd9840c328e4ceee',
 'csrc/kernels/attention_striped_unpack.h':'2f03998b9ac66474acaab70563b1608c4f3646369b7767a76ccd171eecbf3e21',
 'csrc/kernels/attention_cv_window_range_cluster4.cpp':'14b28f9181220432a15bcc3f23679e539473b12e01d858bbb00ff3345522ba65',
}
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()


def helper():
    for p,h in PINS.items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('source drift: '+p)
    original=(ROOT/next(iter(PINS))).read_text();text=original;edits=[]
    def replace(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError(('edit drift',old,text.count(old)))
        text=text.replace(old,new);edits.append((old,new,count))
    replace('HalfKv==16 || HalfKv==64','HalfKv==16 || HalfKv==32 || HalfKv==64')
    replace('striped HalfKv must be 16 or 64','striped HalfKv must be 16, 32 or 64')
    replace('compareRows=HalfKv==16?64:128','compareRows=HalfKv<=32?64:128')
    replace('''  } else {
    ltZero=ltMax.ReinterpretCast<uint64_t>().GetValue(1);''','''  } else if constexpr (HalfKv==32) {
    gtScale&=0xffffffffULL;ltScale&=0xffffffffULL;
    ltZero=(ltMax.ReinterpretCast<uint64_t>().GetValue(0)>>32)&0xffffffffULL;
    negZero=(ltNegInf.ReinterpretCast<uint64_t>().GetValue(0)>>32)&0xffffffffULL;
    geZero=(ge0.ReinterpretCast<uint64_t>().GetValue(0)>>32)&0xffffffffULL;
  } else {
    ltZero=ltMax.ReinterpretCast<uint64_t>().GetValue(1);''')
    replace('oscar_ascend_padded_natural','oscar_ascend_unpack32',2)
    replace('#include "oscar_common.h"','#include "../../kernels/oscar_common.h"')
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse identity drift')
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('helper source identity failed')
    return '''// RESEARCH ONLY. Archive #13-22/#126/#144-145/#148/#151/#154-156; D.4.
// 1. Fused source0 INT2 dequant. 2. Prior whole-history restore cost6.5s.
// 3. Bounded HalfKv32, exact code/meta/Mul->Add; no BF16 restoration or skip.
// 4. C4 helper calls halve; bytes/math count stay. Precision before timing.
// The original header's HalfKv64 byte comment describes its decoder, not C4.
'''+text


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    text=helper();dest=HERE/'unpack32.h'
    if a.check:
        if dest.read_text()!=text:raise ValueError('generated helper stale')
    else:dest.write_text(text)
    manifest={'scope':'isolated helper only','dependency_sha256':PINS,'helper_sha256':sha(text),
        'inverse_source_bytes_identical_after_edits':True,'half_rows':32,'compare_rows':64,
        'metadata_zero_mask_bits':[32,63],'scratch_bytes':1024,'proposed_C4_explicit_UB_bytes':157760,
        'softmax_temporary_view_must_stay_bytes':16384,
        'cann_compile':'not_run','CPU_debug':'not_run','CAModel':'not_run','NPU':'not_run','production_enabled':False}
    encoded=json.dumps(manifest,indent=2)+'\n';path=HERE/'manifest.json'
    if a.check:
        if path.read_text()!=encoded:raise ValueError('manifest stale')
    else:path.write_text(encoded)
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False}))
if __name__=='__main__':main()
