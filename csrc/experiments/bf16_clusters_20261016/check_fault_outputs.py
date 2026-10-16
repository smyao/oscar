"""Check declared C16 fault output domains independently of status contents.

Archive #145/#148/#157: metadata and QR validation are lane-owned. An error
must poison only its original source0 lane rows, not neighboring members.
No candidate-produced status value defines the expected output domain.
"""
from __future__ import annotations
import argparse, json, math, struct
from pathlib import Path


def check(fixture: Path, result: Path) -> dict:
    n,hq,hk,d,_,_,_,_,_,_,_,_,_,_,splits,_=map(int,(fixture/'shape.txt').read_text().split())
    _,_,bad,qr=map(int,(fixture/'cluster_flags.txt').read_text().split())
    if bad<0 and qr<0:raise ValueError('declared fault required')
    segments=3*splits;group=hq//hk;expected=set();expected_status={}
    tasks=list(struct.iter_unpack('<16q',(result/'tasks.bin').read_bytes()))
    if len(tasks)!=n*hk*segments:raise ValueError('complete prepare table required')
    for task,row in enumerate(tasks):
        qb,count,head,begin,end,_,segment,kind=row[:8]
        if count<=0 or kind!=0:continue
        for lane in range(2):
            error=0
            if begin<=bad<end and ((bad-begin)%32)//16==lane:error=3
            if qb<=qr<qb+count and ((qr-qb)*group)//64==lane:error=2
            if not error:continue
            expected_status[task*2+lane]=error
            for r in range(lane*64,min((lane+1)*64,count*group)):
                expected.add(((qb+r//group)*hq+head*group+r%group)*segments+segment)
    actual_status=struct.unpack('<'+'i'*(len(tasks)*2),(result/'status.bin').read_bytes())
    if {i:x for i,x in enumerate(actual_status) if x}!=expected_status:
        raise ValueError('complete status domain differs from declared fault')
    lse=struct.unpack('<'+'f'*(n*hq*segments),(result/'lse.bin').read_bytes())
    partial=struct.unpack('<'+'f'*(n*hq*segments*d),(result/'partial.bin').read_bytes())
    for row,value in enumerate(lse):
        if row in expected:
            if not math.isnan(value):raise ValueError('fault LSE must be NaN')
        elif not (math.isfinite(value) or value==-math.inf):
            raise ValueError('fault leaked to independent LSE row')
    for row in range(n*hq*segments):
        values=partial[row*d:(row+1)*d]
        if row in expected:
            if not all(math.isnan(x) for x in values):raise ValueError('fault partial must be all NaN')
        elif not all(math.isfinite(x) for x in values):
            raise ValueError('fault leaked to independent partial row')
    return {'status':'passed','declared_metadata_position':bad,'declared_QR_token':qr,
        'exact_status_words':len(actual_status),'nonzero_status_words':len(expected_status),
        'expected_NaN_partial_rows':len(expected),'all_other_partial_rows_finite':True,
        'all_other_LSE_rows_finite_or_empty':True,'expected_domain_uses_actual_status':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=check(a.fixture,a.result)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
