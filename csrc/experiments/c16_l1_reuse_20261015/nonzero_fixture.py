"""Archive #126/#129/#145/#154: legal nonzero-owner C16 CPU fixture.

Two requests keep all2016 rows: context0 current-only CV is empty after the
same production suppression; warm request owns the complete core1 C16 cluster.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import probe_history_reuse as reuse
from tools.striped_fixture import raw_to_striped
from csrc.experiments.window_batch_decode.fixture import _bytes
from csrc.experiments.c16_l1_reuse_20261015.generate import sha


def export(path):
    import torch
    from oscar_ascend.ops.reference import encode_kv,decode_kv,attention
    if path.exists():raise ValueError('fresh fixture required')
    torch.set_num_threads(min(4,torch.get_num_threads()));logical={}
    def observe(r,b,q,k,v,ck,cv):logical[r]=(k.clone(),v.clone())
    spec=reuse.Shape('c16_nonzero_owner',(336,336),(0,337),256,1,1,True,
        padded_tokens=672,recent_tokens=1)
    fixture=reuse.make_fixture(torch,spec,on_request=observe);t=fixture['cpu'];t['raw']=raw_to_striped(fixture)
    k,v=logical[1];rk=reuse._hadamard(torch,256)
    dk,dv=decode_kv(encode_kv(k.float()@rk,v.float()@t['rv']),256)
    dk,dv=dk@rk.T,dv@t['rv'].T
    kk,vv=k.float().clone(),v.float().clone();kk[64:],vv[64:]=dk[64:],dv[64:]
    output=torch.zeros((672,6,256));lse=torch.full((672,6),float('-inf'))
    for query in range(336,672):
        result=attention(t['q'][query:query+1],kk,vv,scale=.0625,causal=False)
        output[query],lse[query]=result.output[0],result.lse[0]
    t.update(expected_output=output,expected_lse=lse,
        expected_positions=torch.cat((torch.arange(336),torch.arange(337,673))).to(torch.int64))
    attrs=[672,6,1,256,2,t['table'].shape[1],2016,reuse.BLOCK_TOKENS,
        fixture['blocks'],reuse.PREFIX,fixture['stride'],68*256,68,64,1,3,1]
    path.mkdir(parents=True);hashes={}
    for name,value in t.items():
        data=_bytes(value);(path/(name+'.bin')).write_bytes(data);hashes[name]=sha(data)
    (path/'shape.txt').write_text(' '.join(map(str,attrs))+'\n')
    (path/'bad_position.txt').write_text('-1\n');(path/'tolerance.txt').write_text('0.005 0.005\n')
    meta={'attrs':attrs,'context':337,'bad_position':-1,'input_sha256':hashes,
        'scope':'two requests Q336 each, contexts0/337 recent1; all2016 tasks; first CV empty, second dense original-space oracle',
        'cluster_owner':1,'cluster_qbegins':list(range(336,672,21)),
        'first_request':'all three CV partials0/LSE-inf after healthy current suppression',
        'CPU_deadline_each_arm_s':120,'NPU':'not_run','whole_service':'not_established'}
    (path/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps({'output':str(a.output),'scope':export(a.output)['scope']}))
