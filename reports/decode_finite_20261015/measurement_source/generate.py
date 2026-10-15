"""Archive #126/#134/#140/#145/#148/#154-157; full D.4 read.

Two independent current-production decode denominators. Only Softmax finite
result readback changes; no production source, math, publisher or flags change.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from csrc.experiments.c16_anchor_skip_20261015.generate import method

SOURCES={
 'q4':('attention_cv_hybrid_l1_decode.cpp','03df3835144e26ad399e9be8cd41f684b2705ed280c77029bf78c4e74bc9bad0'),
 'q1':('attention_cv_mdl_decode.cpp','1d451bd9c0c47e1d61a5531c785ef5b0a72a1978c79ee972dca8f7a1c6cf7b0f')}
SOFTMAX_SHA='d3b684ef21c9427b2356f17eb32d962a85b6e2ce7dc0ec811e45d04bacdbf5ac'
LOOP_SHA='059084595499e2819d529ad30ff137db945c335aa56fd955860a102b207ea1d7'
HEADER='''// ISOLATED DECODE FINITE READBACK. Archive #126/#134/#140/#145/#148/#154-157.
// D.4 complete original logs read; four questions:
// 1. Fused FIA per-row finite-check V->S synchronization only.
// 2. Historical full recovery6499.8-6655.1ms/store209-215ms are forbidden
//    failure bounds, never current phase shares; no history restoration.
// 3. Fully-visible kind0 KV256 retains every original FP32 ReduceSum/order,
//    tmp[16], PIPE_V and finite/error decisions. Dead natural UB stores one
//    result/32B; read after one V_S and finish S_V before alpha reuse.
//    Partial/precise/empty paths, future-error commit and all flags unchanged.
// 4. Zero new GM/UB/L1; full independent q4/q1 multi-KV CPU/CAModel pairs
//    and actual-method exceptional cases required. No NPU/graph/E2E claim.
'''

def sha(v):return hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
def digest(p):return sha(Path(p).read_bytes())

def apply_finite(text):
    start,end=method(text,'Softmax');before=text[start:end]
    if sha(before)!=SOFTMAX_SHA:raise ValueError('complete decode Softmax drift')
    a=before.index('    for(int32_t r=0;r<activeRows;++r) {')
    b=before.index('    // The two 16-row max/sum arrays',a)
    loop=before[a:b]
    if sha(loop)!=LOOP_SHA:raise ValueError('original finite/mask loop drift')
    batch='''    const int64_t earliestPosition=context+qbegin+rowBase/group-requestBegin;
    const int64_t earliestCut=Max64(g.sink,earliestPosition+1-g.recent);
    const bool fullyVisibleHistory=kind==0 && jEnd==kKvRows && start>=g.sink &&
        tileEnd<=Min64(context,Min64(earliestCut,earliestPosition+1));
    if(fullyVisibleHistory) {
      // Next K publication completed before Softmax. Packed next V and
      // pending errors live elsewhere; natural UB is dead until alpha Brcb.
      auto finiteSums=naturalBuf.Get<float>(kHalfRows*8);
      for(int32_t r=0;r<activeRows;++r) {
        PipeBarrier<PIPE_V>();
        ReduceSum(finiteSums[r*8],scores[r*kKvRows],tmp[16],kKvRows);
      }
      Fence<HardEvent::V_S>();
      for(int32_t r=0;r<activeRows;++r)
        if(!Finite(finiteSums.GetValue(r*8)))error=2;
      Fence<HardEvent::S_V>();
    } else {
'''+loop+'''    }
'''
    after=before[:a]+batch+before[b:]
    if after.replace(batch,loop)!=before:raise ValueError('Softmax inverse failed')
    return text[:start]+after+text[end:],{'old_Softmax_sha256':sha(before),'new_Softmax_sha256':sha(after),
        'original_loop_sha256':sha(loop),'fallback_loop_byte_identical':True,
        'original_FP32_ReduceSum256_and_tmp16':True,'result_slot_bytes':512,'natural_capacity_bytes':34816}

def generate(family,arm):
    name,pin=SOURCES[family];source=ROOT/'csrc/kernels'/name;original=source.read_text()
    if sha(original)!=pin:raise ValueError('production denominator changed '+family)
    text=original;proof={}
    if arm=='candidate':text,proof=apply_finite(text)
    edits=[]
    def edit(old,new):
        nonlocal text
        n=text.count(old)
        if not n:raise ValueError('relocation missing '+old)
        text=text.replace(old,new);edits.append((old,new,n))
    # Keep both MDL entrypoints uniquely named; only its q1 enters this study.
    prefix='oscar_attention_cv_hybrid_l1_decode' if family=='q4' else 'oscar_attention_cv_mdl_decode'
    edit(prefix,f'oscar_decode_finite_{family}_{arm}')
    launch='attention_cv_hybrid_l1_decode_launch' if family=='q4' else 'attention_cv_mdl_decode_launch'
    edit(launch,f'decode_finite_{family}_{arm}_launch')
    if family=='q1':edit('attention_cv_mdl_decode_q1_launch',f'decode_finite_{family}_{arm}_q1_launch')
    for inc in re.findall(r'^#include "([^"\n]+)"',text,re.M):
        resolved=(source.parent/inc).resolve()
        # SDK quote-includes (for example lib/matmul_intf.h) resolve via
        # the compiler's SDK include directories, not the source folder.
        if not resolved.is_file():continue
        if not resolved.is_relative_to(ROOT/'csrc'):raise ValueError('unexpected include')
        import os
        edit('#include "'+inc+'"','#include "'+os.path.relpath(resolved,HERE)+'"')
    restored=text
    for old,new,n in reversed(edits):
        if restored.count(new)!=n:raise ValueError('inverse relocation count')
        restored=restored.replace(new,old)
    if arm=='candidate':
        a,b=method(restored,'Softmax');restored=restored[:a]+original[slice(*method(original,'Softmax'))]+restored[b:]
    if restored!=original:raise ValueError('outside Softmax/symbol/include drift')
    names=set(re.findall(r'__aicore__\s+(?:inline\s+)?(?:constexpr\s+)?[\w:<>]+\s+(\w+)\(',original))
    unchanged={}
    for method_name in sorted(n for n in names-{'Softmax'} if not n.startswith('oscar_')):
        before=original[slice(*method(original,method_name))];after=text[slice(*method(text,method_name))]
        if before!=after:raise ValueError('other method changed '+method_name)
        unchanged[method_name]=sha(before)
    flags=lambda t:re.findall(r'CrossCore(?:WaitFlag|SetFlag)[^;]*;',t)
    if flags(text)!=flags(original):raise ValueError('crosscore protocol drift')
    text=HEADER+text
    symbol=f'oscar_decode_finite_{family}_{arm}'+('_q1' if family=='q1' else '')+'_kernel'
    return text,{'family':family,'arm':arm,'source':str(source.relative_to(ROOT)),
       'source_sha256':pin,'generated_sha256':sha(text),'kernel':symbol,'proof':proof,
       'all_other_methods_unchanged':unchanged,'crosscore_flags_unchanged':True,
       'workspace_per_core':1146880,'additional_GM_UB_L1_bytes':0,'production_changed':False,
       'CPU':'not_run','CANN':'not_run','CAModel':'not_run','NPU':'not_run'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--build-binding',type=Path);a=p.parse_args();manifest={}
    for family in SOURCES:
        manifest[family]={}
        for arm in ('baseline','candidate'):
            text,manifest[family][arm]=generate(family,arm);path=HERE/f'{family}_{arm}.cpp'
            if a.check:
                if path.read_text()!=text:raise ValueError('source drift '+str(path))
            else:path.write_text(text)
    body=json.dumps(manifest,indent=2)+'\n';path=HERE/'manifest.json'
    if a.check:
        if path.read_text()!=body:raise ValueError('manifest drift')
    else:path.write_text(body)
    if a.build_binding:
        a.build_binding.write_text(json.dumps({'manifest_sha256':digest(path),'generator_sha256':digest(__file__),
            'sources':{f'{f}_{arm}.cpp':digest(HERE/f'{f}_{arm}.cpp') for f in SOURCES for arm in ('baseline','candidate')}},indent=2)+'\n')
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False}))
if __name__=='__main__':main()
