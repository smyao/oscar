"""Archive #126/#145/#154/#155, D.4: complete C4 source0+1 only.

Declared Q336/C113 (one round) and C337 (full256+tail17). All1008 real
prepare tasks remain; production prefill suppression only empties source2.
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
from csrc.experiments.c4_member_pipeline.generate import sha


def export(path,context=113,bad=-1):
    import torch
    from oscar_ascend.ops.reference import encode_kv,decode_kv,attention
    if path.exists() or context not in (113,337) or bad not in (-1,80,96,336) or bad>=context:
        raise ValueError('new output; declared context/fault domain required')
    torch.set_num_threads(min(4,torch.get_num_threads()));logical={}
    def observe(r,b,q,k,v,ck,cv):logical[r]=(k.clone(),v.clone())
    spec=reuse.Shape('c4_member_pipeline',(336,),(context,),256,1,1,True,padded_tokens=336,
        recent_tokens=256,corrupt_metadata=bad>=0,corrupt_position=bad if bad>=0 else 100)
    f=reuse.make_fixture(torch,spec,on_request=observe);t=f['cpu'];t['raw']=raw_to_striped(f)
    k,v=logical[0];rk=reuse._hadamard(torch,256)
    dk,dv=decode_kv(encode_kv(k.float()@rk,v.float()@t['rv']),256);dk,dv=dk@rk.T,dv@t['rv'].T
    expected=torch.empty((336,6,256));lse=torch.empty((336,6))
    for q in range(336):
        cut=min(context,max(64,context+q+1-256));kk,vv=k.float().clone(),v.float().clone()
        kk[64:cut],vv[64:cut]=dk[64:cut],dv[64:cut]
        result=attention(t['q'][q:q+1],kk,vv,scale=.0625,causal=False)
        expected[q],lse[q]=result.output[0],result.lse[0]
    t.update(expected_output=expected,expected_lse=lse,
        expected_positions=torch.arange(context,context+336,dtype=torch.int64))
    attrs=[336,6,1,256,1,t['table'].shape[1],1008,reuse.BLOCK_TOKENS,f['blocks'],reuse.PREFIX,
        f['stride'],323*256,323,64,256,3,1]
    path.mkdir(parents=True);hashes={}
    for name,value in t.items():
        data=_bytes(value);(path/(name+'.bin')).write_bytes(data);hashes[name]=sha(data)
    (path/'shape.txt').write_text(' '.join(map(str,attrs))+'\n')
    (path/'bad_position.txt').write_text(str(bad)+'\n');(path/'tolerance.txt').write_text('0.005 0.005\n')
    meta={'scope':f'all Q336/C{context} source0+1; source2 suppressed, all1008 tasks kept',
        'attrs':attrs,'context':context,'bad_position':bad,'input_sha256':hashes,
        'valid_numeric_oracle':bad<0,'declared_deadlines':{'CPU_each_arm_s':120,'CAModel_each_arm_s':900},
        'invalid_case_rule':'status3 in original lane16 domain; compare all output bits with baseline',
        'NPU':'not_run','full_model':'not_run'}
    (path/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context',type=int,choices=(113,337),default=113)
    p.add_argument('--bad-position',type=int,choices=(-1,80,96,336),default=-1);a=p.parse_args()
    m=export(a.output,a.context,a.bad_position)
    print(json.dumps({'output':str(a.output),'context':m['context'],'bad_position':m['bad_position'],'execution':'host_oracle_only'}))
