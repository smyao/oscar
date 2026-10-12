"""Archive #13-22/#126/#145/#154/#155; D.4: small complete C4 parity fixture.

Q336/C113/R1/S1/all1008tasks, source0+1 oracle; current source2 remains empty
as in production prefill. Invalid zero-scale positions80/96 witness AIV1/0.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import probe_history_reuse as reuse
from tools.striped_fixture import raw_to_striped
from csrc.experiments.window_batch_decode.fixture import _bytes
from csrc.experiments.c4_unpack32.fixture import sha


def export(path,bad=-1):
    import torch
    from oscar_ascend.ops.reference import encode_kv,decode_kv,attention
    if path.exists() or bad not in (-1,80,96):raise ValueError('new path and declared bad-position required')
    torch.set_num_threads(min(4,torch.get_num_threads()));logical={}
    def observe(r,b,q,k,v,ck,cv):logical[r]=(k.clone(),v.clone())
    spec=reuse.Shape('unpack32_cv_c113',(336,),(113,),256,1,1,True,padded_tokens=336,recent_tokens=256,
                     corrupt_metadata=bad>=0,corrupt_position=bad if bad>=0 else 100)
    f=reuse.make_fixture(torch,spec,on_request=observe);t=f['cpu'];t['raw']=raw_to_striped(f)
    k,v=logical[0];rk=reuse._hadamard(torch,256)
    dk,dv=decode_kv(encode_kv(k.float()@rk,v.float()@t['rv']),256);dk,dv=dk@rk.T,dv@t['rv'].T
    expected=torch.empty((336,6,256));lse=torch.empty((336,6))
    for q in range(336):
        cut=min(113,max(64,113+q+1-256));kk,vv=k.float().clone(),v.float().clone()
        kk[64:cut],vv[64:cut]=dk[64:cut],dv[64:cut]
        result=attention(t['q'][q:q+1],kk,vv,scale=.0625,causal=False)
        expected[q],lse[q]=result.output[0],result.lse[0]
    t.update(expected_output=expected,expected_lse=lse,expected_positions=torch.arange(113,449,dtype=torch.int64))
    attrs=[336,6,1,256,1,t['table'].shape[1],1008,reuse.BLOCK_TOKENS,f['blocks'],reuse.PREFIX,f['stride'],323*256,323,64,256,3,1]
    path.mkdir(parents=True);hashes={}
    for name,value in t.items():data=_bytes(value);(path/(name+'.bin')).write_bytes(data);hashes[name]=sha(data)
    (path/'shape.txt').write_text(' '.join(map(str,attrs))+'\n');(path/'bad_position.txt').write_text(str(bad)+'\n')
    (path/'tolerance.txt').write_text('0.005 0.005\n')
    meta={'scope':'all Q336/C113 source0+1; source2 suppressed, all1008 tasks kept','attrs':attrs,'bad_position':bad,
        'fault_expected_lane':((bad-64)%32)//16 if bad>=0 else None,'input_sha256':hashes,
        'valid_numeric_oracle':bad<0,'invalid_case_rule':'status3 at consuming source0 leader in original KV-owning lane; other statuses zero',
        'NPU':'not_run','full_model':'not_run'}
    (path/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--bad-position',type=int,choices=(-1,80,96),default=-1);a=p.parse_args()
    m=export(a.output,a.bad_position);print(json.dumps({'output':str(a.output),'bad_position':m['bad_position'],'execution':'host_oracle_only'}))
