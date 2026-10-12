"""Rebind the full Kt/Vt byte gate to exact u32-index methods; old gate untouched."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_kvt_nz import publisher_probe as mother
from csrc.experiments.decode_phase_profile.generate import function
sha=mother.sha
SOURCE=HERE/'c4_candidate.cpp'
NAMES=mother.NAMES
verify=mother.verify


def method(source,name):
    a,b=function(source,name);marker='  __aicore__ void '+name+'('
    if source.count(marker)!=1:raise ValueError('method changed '+name)
    start=source.rfind(marker,0,a)
    if start<0:raise ValueError('missing declaration')
    return source[start:b]


def generated():
    text,old_identity=mother.generated();old_source=mother.SOURCE.read_text();source=SOURCE.read_text();bindings={}
    for name in NAMES:
        old=method(old_source,name);new=method(source,name)
        if text.count(old)!=1:raise ValueError('mother publisher method drift')
        text=text.replace(old,new);bindings[name]=sha(new)
    old='pipe.InitBuffer(vtIndexBuf,8192);'
    if text.count(old)!=1:raise ValueError('publisher buffer allocation changed')
    text=text.replace(old,'pipe.InitBuffer(vtIndexBuf,16384);')
    text=text.replace('oscar_c4_kvt_nz_publisher','oscar_c4_kvt_nz_u32_publisher')
    identity={**old_identity,'source':'csrc/experiments/c4_kvt_nz_u32/c4_candidate.cpp',
        'source_sha256':sha(source),'method_sha256':bindings,'harness_v1_sha256':sha(Path(mother.__file__).read_bytes()),
        'index_dtype':'uint32','index_bytes':16384}
    old=json.dumps(old_identity,sort_keys=True)
    if text.count(old)!=2:raise ValueError('embedded source identity changed')
    text=text.replace(old,json.dumps(identity,sort_keys=True))
    for name in NAMES:
        if text.count(method(source,name))!=1:raise ValueError('extracted method differs')
    return text,identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--verify',type=Path);a=p.parse_args()
    text,identity=generated();path=HERE/'publisher_probe.cpp'
    if a.check or a.verify:
        if path.read_text()!=text:raise ValueError('publisher source drift')
    else:path.write_text(text)
    print(json.dumps(verify(a.verify,identity) if a.verify else {'status':'checked' if a.check else 'generated','source_binding':identity,'execution':'not_run'}))


if __name__=='__main__':main()
