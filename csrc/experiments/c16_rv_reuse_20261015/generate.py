"""Archive #126/#145/#148/#154; D.4: bounded final Rv residency only.

The accepted C16 B1 source is the sole denominator. Strict inverse edits
bind both arms, every Vector method, resource declaration and existing flag.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c16_l1_reuse_20261015.safe_logging_runner import transform

MOTHER=ROOT/'csrc/experiments/c16_l1_reuse_20261015/c16_candidate.cpp'
MOTHER_SHA='1deb6ab15a34ed263000c9a99530d0f9ee1a7e682a84ba7657b2c968d2ae6ec0'
PREVIOUS=ROOT/'csrc/experiments/c16_l1_reuse_20261015'
HEADER='''// GENERATED C16 FINAL RV RESIDENCY; archive #126/#145/#148/#154.
// D.4 four questions, complete original timing logs reread:
// 1. Fused history FIA final inverse rotation reloads one Rv for16 members.
// 2. Old whole-history restore6499.8-6655.1ms and store209-215ms are
//    historical forbidden structures, not current measured phase shares.
// 3. After final PV, original MTE1->MTE2 and MTE2->MTE1 fence loads Rv
//    once into the existing256KiB external B1. Sixteen original FP32/K128
//    MatmulResidentB calls, per-member waits and outputs remain unchanged.
//    No extra buffer, no flag change, no mathematical rotation relocation.
// 4. Removes15 Rv loads/cluster (3.75MiB logical GM->B1), not measured
//    HBM or a speed claim. GM7487488B/core, UB157760B/AIV unchanged.
//    Complete same-library CPU/oracle/CAModel pairing required; no NPU,
//    graph, long-context or model acceptance is implied.
'''


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()


def replace_exact(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError(('fixed edit anchor',old[:70],text.count(old),count))
    return text.replace(old,new)


def function(text,name):
    matches=list(re.finditer(r'__aicore__[^\n]*\b'+re.escape(name)+r'\(',text))
    if len(matches)!=1:raise ValueError(('unique method',name,len(matches)))
    begin=text.index('{',matches[0].end());depth=1;end=begin+1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return begin,end


def generate_arm(arm):
    if arm not in ('baseline','candidate'):raise ValueError('explicit two-arm identity required')
    original=MOTHER.read_text()
    if sha(original)!=MOTHER_SHA:raise ValueError('accepted B1 mother changed')
    text=original;edits=[]
    def edit(old,new,count=1):
        nonlocal text
        text=replace_exact(text,old,new,count);edits.append((old,new,count))
    if arm=='candidate':
        start,end=function(text,'CubeCluster');body=text[start:end]
        old='''    SelectClusterSlots(0,0);
    for(int32_t member=0;member<16;++member) {'''
        new='''    SelectClusterSlots(0,0);
    // Final PV has finished issuing; the original ownership fence inside
    // LoadResidentB orders its last MTE1 read before this one Rv load.
    // MatmulImpl::End does not own or overwrite the external B1 buffer.
    LoadResidentB(0,true);
    for(int32_t member=0;member<16;++member) {'''
        changed=replace_exact(body,old,new)
        changed=replace_exact(changed,'Matmul(qOffset,0,rotOffset,rows,D,D,true);',
            'MatmulResidentB(qOffset,rotOffset,rows,D,D,true);')
        edit(body,changed)
    edit('c16_l1_reuse_candidate',f'c16_rv_reuse_{arm}',original.count('c16_l1_reuse_candidate'))
    restored=text
    for old,new,count in reversed(edits):
        restored=replace_exact(restored,new,old,count)
    if restored!=original:raise ValueError('full accepted mother not restored')
    names=re.findall(r'__aicore__\s+(?:inline\s+)?(?:constexpr\s+)?[\w:<>]+\s+(\w+)\(',original)
    unchanged={}
    for name in sorted(set(names)-{'CubeCluster'}):
        before=original[slice(*function(original,name))]
        after=text[slice(*function(text,name))]
        if before!=after:raise ValueError('unintended method/resource/math/flag change '+name)
        unchanged[name]=sha(before)
    old_cube=original[slice(*function(original,'CubeCluster'))]
    new_cube=text[slice(*function(text,'CubeCluster'))]
    flags=lambda s:re.findall(r'CrossCore(?:WaitFlag|SetFlag)[^;]*;',s)
    if flags(old_cube)!=flags(new_cube):raise ValueError('cross-core protocol changed')
    text=HEADER+text
    return text,{'arm':arm,'symbol':f'oscar_c16_rv_reuse_{arm}_kernel',
        'source_sha256':MOTHER_SHA,'H32_source_sha256':'0795a07b1db0ba3d12e7ae762929b752d3bfb6d7475a925cf9e25e62ddeeb4bc',
        'generated_sha256':sha(text),'full_inverse_mother_bytes':True,
        'all_other_methods_unchanged':unchanged,'crosscore_flags_unchanged':True,
        'workspace_per_core':7487488,'explicit_UB_per_AIV':157760,'external_B1_bytes':262144,
        'additional_workspace_UB_L1_bytes':0,'Rv_loads_per_mature_cluster':1 if arm=='candidate' else 16,
        'NPU':'not_run','production_changed':False}


def harness(name):
    path=PREVIOUS/name;original=path.read_text();pins=json.loads((HERE/'pins.json').read_text())
    if sha(original)!=pins['harness'][name]:raise ValueError('fixed harness changed '+name)
    text=original
    if name.endswith('.cpp'):
        text=replace_exact(text,'extern "C" void oscar_c16_l1_reuse_stable_kernel(CV_ARGS);\n','')
        text=replace_exact(text,'arm!="stable"&&','')
        text=replace_exact(text,'if(arm=="stable"){RUN(oscar_c16_l1_reuse_stable_kernel);}else ','')
        text=text.replace('c16_l1_reuse','c16_rv_reuse')
    else:
        text=transform(text)
        text=replace_exact(text,'from csrc.experiments.c16_l1_reuse_20261015 import generate as generate_c4',
            'from csrc.experiments.c16_rv_reuse_20261015 import generate as generate_c4')
        text=replace_exact(text,"choices=('stable','baseline','candidate')","choices=('baseline','candidate')")
        text=replace_exact(text,"stock=json.loads(","if configuration['config.json']['LOG']['core_enable_mask']!=['0x0']:raise ValueError('official zero-mask required')\n    stock=json.loads(")
        if 'os.dup2' in text or '/proc/self/fd' in text:raise ValueError('unsafe polling remained')
    return text,{'mother':str(path.relative_to(ROOT)),'mother_sha256':sha(original),'sha256':sha(text),
        'numeric_status_guard_and_complete_profile_gates':'unchanged','arm_symbols':'independent two-arm binding',
        'logging':'official mask0, no FD polling daemon'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    manifest={'arms':{},'harness':{}}
    for arm in ('baseline','candidate'):
        text,identity=generate_arm(arm);manifest['arms'][arm]=identity;path=HERE/f'c16_{arm}.cpp'
        if a.check:
            if path.read_text()!=text:raise ValueError('source drift '+arm)
        else:path.write_text(text)
    for name in ('cv_probe_small.cpp','cv_probe_qr.cpp','cv_replay_small.py'):
        text,identity=harness(name);manifest['harness'][name]=identity;path=HERE/name
        if a.check:
            if path.read_text()!=text:raise ValueError('harness drift '+name)
        else:path.write_text(text)
    text=json.dumps(manifest,indent=2)+'\n';path=HERE/'manifest.json'
    if a.check:
        if path.read_text()!=text:raise ValueError('manifest drift')
    else:path.write_text(text)
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False,'resource_increment_bytes':0}))


if __name__=='__main__':main()
