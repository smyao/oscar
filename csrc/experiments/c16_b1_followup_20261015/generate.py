"""Archive #126/#134/#140/#145/#148/#154; one Rv->finite->anchor-skip chain.

Research preparation only until Rv closes and the new full pair qualifies.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c16_anchor_skip_20261015.generate import apply_skip,method,sha
from csrc.experiments.c16_b1_followup_20261015.finite_transform import apply_finite

SOURCE=ROOT/'csrc/experiments/c16_rv_reuse_20261015/c16_candidate.cpp'
SOURCE_SHA='9603817dddfba0e6357b42756a9956880137348ce8076c61aff58884acb6210b'
HEADER='''// GENERATED C16 B1 FOLLOWUP: FINAL RV + FINITE READBACK + ANCHOR SKIP.
// Archive #126/#134/#140/#145/#148/#154; complete D.4 reread.
// 1. Fused history FIA finite-check synchronization and task metadata scan.
// 2. Old6499.8-6655.1ms full-history restoration/209-215ms stores remain
//    forbidden historical bounds, not current phase shares.
// 3. Fully-visible history tiles alone retain every original FP32 ReduceSum
//    in row order but batch V->S result readback using dead natural UB.
//    Partial/precise/empty rows retain the exact original loop. Successful
//    native-prepared C16 clusters skip covered non-anchor scan rows only;
//    first-pass statuses, every math method and cross-core flag remain.
// 4. No resource increase (GM7487488B/core,UB157760B/AIV,B1=256KiB).
//    Full C320 same-library qualification plus overflow/cancel/nonfinite/
//    masked-row CPU evidence required. Source counts are not speed/NPU,
//    graph,32K/64K or whole-model acceptance.
'''


def generate_arm(arm):
    if arm not in ('baseline','candidate'):raise ValueError('two-arm identity required')
    original=SOURCE.read_text()
    if sha(original)!=SOURCE_SHA:raise ValueError('pending qualified Rv source changed')
    text=original;proof={}
    if arm=='candidate':
        text,proof['finite']=apply_finite(text)
        text,proof['anchor_skip']=apply_skip(text)
    old_symbol='c16_rv_reuse_candidate';new_symbol=f'c16_b1_followup_{arm}'
    if text.count(old_symbol)!=2:raise ValueError('symbol relocation anchor')
    text=text.replace(old_symbol,new_symbol)
    restored=text.replace(new_symbol,old_symbol)
    if arm=='candidate':
        for name in ('Softmax','Process'):
            old=original[slice(*method(original,name))];start,end=method(restored,name)
            restored=restored[:start]+old+restored[end:]
    if restored!=original:raise ValueError('outside two methods/symbols changed')
    names=re.findall(r'__aicore__\s+(?:inline\s+)?(?:constexpr\s+)?[\w:<>]+\s+(\w+)\(',original)
    unchanged={}
    for name in sorted(set(names)-{'Softmax','Process'}):
        before=original[slice(*method(original,name))];after=text[slice(*method(text,name))]
        if before!=after:raise ValueError('unintended method/resource/flag change '+name)
        unchanged[name]=sha(before)
    flags=lambda value:re.findall(r'CrossCore(?:WaitFlag|SetFlag)[^;]*;',value)
    if flags(text)!=flags(original):raise ValueError('cross-core flag changes')
    text=HEADER+text
    return text,{'arm':arm,'symbol':f'oscar_c16_b1_followup_{arm}_kernel','source_sha256':SOURCE_SHA,
        'generated_sha256':sha(text),'composition':['accepted_B1','final_Rv','same_reduction_batched_readback','successful_cluster_anchor_skip'] if arm=='candidate' else ['accepted_B1','final_Rv'],
        'proof':proof,'all_other_methods_unchanged':unchanged,'crosscore_flags_unchanged':True,
        'workspace_per_core':7487488,'explicit_UB_per_AIV':157760,'external_B1_bytes':262144,
        'additional_workspace_UB_L1_bytes':0,'base_Rv_performance':'pending_current_pair',
        'CPU':'not_run','CANN':'not_run','CAModel':'not_run','NPU':'not_run','production_changed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();manifest={}
    for arm in ('baseline','candidate'):
        text,manifest[arm]=generate_arm(arm);path=HERE/f'c16_{arm}.cpp'
        if a.check:
            if path.read_text()!=text:raise ValueError('generated source drift '+arm)
        else:path.write_text(text)
    body=json.dumps(manifest,indent=2)+'\n';path=HERE/'manifest.json'
    if a.check:
        if path.read_text()!=body:raise ValueError('manifest drift')
    else:path.write_text(body)
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False,'VM_execution':'not_started'}))
