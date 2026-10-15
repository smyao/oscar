"""Archive #134/#140/#148/#154: host predicate implication, not kernel proof.

Check the fast predicate against the original source0 visible interval for
every live row, including GQA row/group boundaries and partial final blocks.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path


def check_case(context,qbegin,request_begin,qcount,group,row_base,start,kvend,sink,recent):
    active=max(0,min(32,qcount*group-row_base))
    if not active:return False
    jend=min(256,kvend-start);tile_end=start+jend
    earliest=context+qbegin+row_base//group-request_begin
    cut=max(sink,earliest+1-recent)
    fast=jend==256 and start>=sink and tile_end<=min(context,cut,earliest+1)
    for r in range(active):
        position=context+qbegin+(row_base+r)//group-request_begin
        lower=max(start,sink);upper=min(tile_end,context,max(sink,position+1-recent),position+1)
        if fast and (upper<=lower or lower!=start or upper!=start+256):
            raise AssertionError(('fast path changed original any/mask',locals()))
    return fast


def prove():
    rng=random.Random(16320261015);cases=fast=0
    for _ in range(20000):
        context=rng.choice((0,64,65,255,256,320,337,512,8192,32768,65536))
        group=rng.choice((1,2,3,6,8,16,64));qcount=rng.randint(1,128//group)
        request_begin=rng.randint(0,4096);qbegin=request_begin+rng.randint(0,16384)
        sink=rng.choice((0,1,64));recent=rng.choice((0,1,32,256,1024))
        kvend=rng.randint(0,context);start=rng.randint(0,kvend)
        for row_base in (0,32,64,96):
            fast+=check_case(context,qbegin,request_begin,qcount,group,row_base,start,kvend,sink,recent);cases+=1
    assert check_case(320,0,0,21,6,96,64,320,64,1)
    assert check_case(337,0,0,21,6,0,64,337,64,1)
    assert not check_case(337,0,0,21,6,0,320,337,64,1)
    assert not check_case(65,0,0,21,6,0,64,65,64,1)
    assert not check_case(320,0,0,21,6,0,64,320,64,256)
    return {'status':'host_visible_interval_implication_passed','cases':cases,'fast_cases':fast,
        'proof':'earliest live row has minimal position/cut; both are nondecreasing across all live rows; jEnd256 and tileEnd<=context/cut/position+1 with start>=sink make every original source0 interval exactly[start,start+256)',
        'C320_recent1_full_tile':'eligible','C65_one_column':'fallback','C337_final17_columns':'fallback',
        'CPU_numerics':'not_run','CANN':'not_run','NPU':'not_run','performance':'not_established'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=prove()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
