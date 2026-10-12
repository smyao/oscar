"""Package verified local candidates without changing their math/control body.

Only project-owned include paths and public symbol names change. Old production
kernels remain available for real same-input gates. No reference tree is edited.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]


def main():
    pairs=[]
    def write(source,destination,changes):
        old=(ROOT/source).read_text();new=old
        for before,after in changes:
            assert before in new,(source,before)
            new=new.replace(before,after)
        (ROOT/destination).write_text(new)
        pairs.append({'source':source,'destination':destination,
            'source_sha256':hashlib.sha256(old.encode()).hexdigest(),
            'destination_sha256':hashlib.sha256(new.encode()).hexdigest(),
            'changes':'include paths and public symbols only'})
    write('csrc/experiments/decode_bundle/attention_bundle.cpp','csrc/kernels/attention_cv_decode_bundle.cpp',[
        ('"../../kernels/oscar_common.h"','"oscar_common.h"'),
        ('"../decode_natural_padding/padded_unpack.h"','"attention_striped_unpack_padded.h"'),
        ('"../range_mask/range_mask.h"','"attention_range_mask.h"'),
        ('"../../include/','"../include/'),
        ('oscar_decode_bundle_q4_kernel','oscar_attention_cv_decode_bundle_kernel'),
        ('oscar_decode_bundle_q1_kernel','oscar_attention_cv_decode_bundle_q1_kernel'),
        ('decode_bundle_q4_launch','attention_cv_decode_bundle_launch'),
        ('decode_bundle_q1_launch','attention_cv_decode_bundle_q1_launch')])
    write('csrc/experiments/decode_natural_padding/padded_unpack.h','csrc/kernels/attention_striped_unpack_padded.h',[
        ('"../../kernels/oscar_common.h"','"oscar_common.h"')])
    write('csrc/experiments/range_mask/range_mask.h','csrc/kernels/attention_range_mask.h',[
        ('"../../kernels/oscar_common.h"','"oscar_common.h"')])
    base=ROOT/'csrc/experiments/prefill_window_range'
    for stem,suffix in (('base',''),('balanced','_balanced'),('cluster4','_cluster4'),('cluster16','_cluster16')):
        wrapper=(base/f'{stem}.cpp').read_text();body=(base/f'{stem}_body.cpp').read_text()
        needle=f'#include "{stem}_body.cpp"'
        assert wrapper.count(needle)==1
        text=wrapper.replace(needle,body)
        text=text.replace('"../../kernels/','"').replace('"../../include/','"../include/')
        text=text.replace('"../range_mask/range_mask.h"','"attention_range_mask.h"')
        destination=ROOT/f'csrc/kernels/attention_cv_window_range{suffix}.cpp'
        destination.write_text(text)
        pairs.append({'source_wrapper':str((base/f'{stem}.cpp').relative_to(ROOT)),
            'source_body':str((base/f'{stem}_body.cpp').relative_to(ROOT)),
            'wrapper_sha256':hashlib.sha256(wrapper.encode()).hexdigest(),
            'body_sha256':hashlib.sha256(body.encode()).hexdigest(),
            'destination':str(destination.relative_to(ROOT)),
            'destination_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'changes':'same wrapper with body include expanded and include paths rebased'})
    (ROOT/'reports/decode_bundle_packaging_20261007.json').write_text(json.dumps({
        'files':pairs,'production_selection':'pending explicit feature registration',
        'npu':'not_run','whole_service':'not_established'},indent=2)+'\n')
    print('PACKAGED_CANDIDATE_SOURCES',len(pairs))


if __name__=='__main__':main()
