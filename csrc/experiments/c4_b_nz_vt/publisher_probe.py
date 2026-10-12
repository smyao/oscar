"""Reuse the completed publisher harness with K-NZ/Vt-NZ address oracles.

Five methods are byte-bound to the new candidate. No old gate is edited.
Archive #126/#145/#148/#154 and startup D.4: fixed KV256 bit movement only.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_b_nz import publisher_probe as base
from csrc.experiments.decode_phase_profile.generate import function
sha=base.sha
SOURCE=HERE/'c4_candidate.cpp'
NAMES=('BuildNz16Indices','BuildVt16Indices','PublishNz16','PublishHistoryKv','PublishKv')


def change(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError('qualified publisher harness drift: '+old)
    return text.replace(old,new)


def generated():
    source=SOURCE.read_text();methods=[];bindings={}
    for name in NAMES:
        a,b=function(source,name);marker='  __aicore__ void '+name+'('
        if source.count(marker)!=1:raise ValueError('method changed: '+name)
        start=source.rfind(marker,0,a)
        if start<0:raise ValueError('missing declaration')
        exact=source[start:b];methods.append(exact);bindings[name]=sha(exact)
    identity={'source':'csrc/experiments/c4_b_nz_vt/c4_candidate.cpp',
              'source_sha256':sha(source),'method_sha256':bindings,
              'harness_mother_sha256':sha(Path(base.__file__).read_bytes()),
              'formats':{'K':'NZ','V':'Vt-NZ'}}
    prefix=change(base.PREFIX,'four methods','five methods')
    prefix=change(prefix,'pipe.InitBuffer(dequantBuf,32768);BuildNz16Indices();',
                  'pipe.InitBuffer(dequantBuf,32768);pipe.InitBuffer(naturalBuf,17408);\n    pipe.InitBuffer(vtIndexBuf,8192);BuildNz16Indices();BuildVt16Indices();')
    suffix=change(base.SUFFIX,'nzIndexBuf,scoreBuf,dequantBuf;',
                  'nzIndexBuf,vtIndexBuf,naturalBuf,scoreBuf,dequantBuf;')
    suffix=change(suffix,'side*65536+(col/8)*256*8+row*8+col%8',
                  'side*65536+(side==0?((col/8)*256*8+row*8+col%8):((row/8)*256*8+col*8+row%8))')
    suffix=suffix.replace('oscar_c4_b_nz_publisher','oscar_c4_b_nz_vt_publisher')
    binding=json.dumps(identity,sort_keys=True)
    content=('// Source binding: '+binding+'\nconstexpr const char* kPublisherSourceBinding=R"publisher('
             +binding+')publisher";\n'+prefix+'\n'.join(methods)+suffix)
    for exact in methods:
        if content.count(exact)!=1:raise ValueError('extracted method bytes differ')
    return content,identity


def verify(directory,identity):
    source=base.guarded(directory/'source.guarded.bin');words=struct.unpack('<131072I',source)
    special=(0,0x80000000,0x7fc00001,0x7fa12345,0x7f800000,0xff800000,1,0x80000001)
    expected=tuple(special[i%65536] if i%65536<8 else 0x3f000000+i for i in range(131072))
    if words!=expected:raise ValueError('source tags differ')
    for name in ('history','precise'):
        receipt=json.loads((directory/(name+'.json')).read_text())
        if receipt.get('status')!='passed' or receipt.get('checked_words')!=131072 or receipt.get('source_binding')!=identity:
            raise ValueError('incomplete or differently bound CPU gate')
        if base.guarded(directory/(name+'-source.guarded.bin'))!=source:raise ValueError('source mutated')
        out=struct.unpack('<131072I',base.guarded(directory/(name+'.guarded.bin')))
        for side in range(2):
            for row in range(256):
                for col in range(256):
                    target=(col//8)*256*8+row*8+col%8 if side==0 else (row//8)*256*8+col*8+row%8
                    if out[side*65536+target]!=words[side*65536+row*256+col]:
                        raise ValueError('K-NZ/Vt-NZ independent address mismatch')
    report={'status':'passed','scope':'publisher_only_raw_bits_not_full_C4',
            'source_binding':identity,'modes':['history_old_lane16','precise16'],
            'words_per_mode':131072,'bytes_per_mode':524288,
            'CPU':'passed','full_candidate_CPU':'not_established','NPU':'not_run','performance':'not_measured'}
    (directory/'publisher-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--verify',type=Path);a=p.parse_args()
    text,identity=generated();path=HERE/'publisher_probe.cpp'
    if a.check or a.verify:
        if path.read_text()!=text:raise ValueError('publisher source drift')
    else:path.write_text(text)
    print(json.dumps(verify(a.verify,identity) if a.verify else {
        'status':'checked' if a.check else 'generated','source_binding':identity,'execution':'not_run'}))


if __name__=='__main__':main()
