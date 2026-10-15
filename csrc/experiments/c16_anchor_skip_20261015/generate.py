"""Archive #34/#36/#126/#129/#144/#145/#148/#152/#154/#157; full D.4.

Independent second-pass anchor traversal only. Keep the first pass and every
numerical method unchanged. Native prepared-task reachability is proved by
host_replay; this source edit alone is not CANN/NPU/performance qualification.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_phase_profile.generate import masked
SOURCE='csrc/experiments/c16_rv_reuse_20261015/c16_candidate.cpp'
SOURCE_SHA='9603817dddfba0e6357b42756a9956880137348ce8076c61aff58884acb6210b'
HEADER='''// ISOLATED C16 VERIFIED-CLUSTER ANCHOR SKIP. Archive #34/#36/#126/#129/
// #144/#145/#148/#152/#154/#157; startup D.4 full original logs read.
// D.4 four questions: (1) fused FIA task metadata traversal, not KV math.
// (2) old host prepare725ms/full restore6499.8-6655.1ms remain forbidden.
// (3) After all16 full same-request/head/split/context groups were verified
// and executed, skip only covered non-anchor rows in this second-pass bucket.
// First-pass padding/error/follower status writes and all CV math/flags stay
// byte-identical. No qcount=0 discard inference, host readback or new buffer.
// (4) Saves metadata loads/fences, not a measured phase share or speedup.
// Native-reachable task proof, full CPU/oracle and combined Rv+B1 timing
// remain separate gates. No production/NPU/graph/performance acceptance.
'''
BEFORE='''            CubeCluster(members);
          } else {VectorCluster(members);}
'''
AFTER=BEFORE+'''          // The native prepare ABI gives sixteen adjacent full groups in
          // one request. Their union covers [token, token+bucketWidth),
          // so no additional request-relative cluster anchor is inside it.
          // First-pass errors/statuses have already retained every row.
          // VectorCluster mutates qbegin: use the unchanged loop token.
          // Clamp to this global bucket and avoid token+width overflow.
          token += Min64(bucketWidth,end-token)-1;
'''

def sha(value):return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()

def method(text,name):
    clean=masked(text)
    matches=list(re.finditer(r'__aicore__[^\n{;]*\b'+re.escape(name)+r'\([^{};]*\)\s*(?:const\s*)?\{',clean))
    if len(matches)!=1:raise ValueError(('unique method',name,len(matches)))
    first=matches[0].start();start=matches[0].end()-1;depth=0
    for end in range(start,len(clean)):
        if clean[end]=='{':depth+=1
        elif clean[end]=='}':
            depth-=1
            if not depth:return first,end+1
    raise ValueError('unclosed method '+name)

PROCESS_SHA='2bb45e7998fc855e4c4ac8e7e6e238df68931ca4f0ba9feb6f1f804876c088f1'

def apply_skip(text):
    """Strict reusable transform for the Rv+finite+skip combined source chain."""
    start,end=method(text,'Process');before=text[start:end]
    if sha(before)!=PROCESS_SHA or before.count(BEFORE)!=1:
        raise ValueError('qualified original C16 Process changed; re-prove native reachability')
    after=before.replace(BEFORE,AFTER)
    if after.replace(AFTER,BEFORE)!=before:
        raise ValueError('Process inverse failed')
    result=text[:start]+after+text[end:]
    return result,{'old_Process_sha256':sha(before),'new_Process_sha256':sha(after),
                   'first_pass_byte_identical':before.split('// Pass two')[0]==after.split('// Pass two')[0],
                   'outside_Process_byte_identical':True,'native_prepared_task_precondition':True}

def generate(arm,source=SOURCE,expected_sha=SOURCE_SHA):
    if arm not in ('baseline','candidate'):raise ValueError('explicit arm required')
    original=(ROOT/source).read_text()
    if sha(original)!=expected_sha:raise ValueError('fixed C16 base source changed')
    text=original;edits=[]
    def edit(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError(('anchor',old[:80],text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count))
    if arm=='candidate':
        rewritten,proof=apply_skip(text)
        edit(text[slice(*method(text,'Process'))],rewritten[slice(*method(rewritten,'Process'))])
    edit('c16_rv_reuse_candidate',f'c16_anchor_skip_{arm}',2)
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse edit ambiguity')
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('full original bytes not restored')
    names=re.findall(r'__aicore__\s+(?:inline\s+)?(?:constexpr\s+)?[\w:<>]+\s+(\w+)\(',masked(original))
    unchanged={}
    for name in sorted(set(names)-{'Process'}):
        before=original[slice(*method(original,name))];after=text[slice(*method(text,name))]
        if before!=after:raise ValueError('changed non-Process method '+name)
        unchanged[name]=sha(before)
    old_process=original[slice(*method(original,'Process'))]
    new_process=text[slice(*method(text,'Process'))]
    if old_process.split('// Pass two')[0]!=new_process.split('// Pass two')[0]:raise ValueError('first pass changed')
    body=HEADER+text
    return body,{'arm':arm,'source':source,'source_sha256':expected_sha,'generated_sha256':sha(body),
        'full_inverse_mother_bytes':True,'first_pass_byte_identical':True,'all_other_methods_unchanged':unchanged,
        'workspace_per_core':7487488,'additional_workspace_UB_L1_bytes':0,
        'scope':'native reachable prepared task tables only; full floating-point kernel not executed',
        'base_qualification':'Rv candidate pending separate full qualification; no old stable/pipeline pairing',
        'production_changed':False,'NPU':'not_run','graph':'not_run','performance':'not_established'}

def host_source():
    base=generate('baseline')[0];candidate=generate('candidate')[0]
    template=(HERE/'host_template.cpp').read_text()
    for name in ('CurrentTask','UseTask','IsClusterMember','TaskValid'):
        template=template.replace('@@'+name+'@@',base[slice(*method(base,name))])
    for key,text in (('Baseline',base),('Candidate',candidate)):
        body=text[slice(*method(text,'Process'))].replace('void Process(',f'void Process{key}(',1)
        template=template.replace('@@Process'+key+'@@',body)
    if '@@' in template:raise ValueError('unfilled host template')
    return template

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    outputs={};manifest={}
    for arm in ('baseline','candidate'):
        outputs[f'c16_{arm}.cpp'],manifest[arm]=generate(arm)
    if (HERE/'host_template.cpp').exists():outputs['host_replay.cpp']=host_source()
    outputs['manifest.json']=json.dumps(manifest,indent=2)+'\n'
    for name,body in outputs.items():
        path=HERE/name
        if a.check:
            if path.read_text()!=body:raise ValueError('generated source drift '+name)
        else:path.write_text(body)
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False}))
if __name__=='__main__':main()
