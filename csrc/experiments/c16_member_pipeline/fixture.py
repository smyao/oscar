"""Archive #126/#129/#145/#148/#154-157, D.4: complete Q672 C16 fixture.

Recent256, sink64, GQA6, one mature16-member history cluster and all sources.
Oracle reconstructs independent PR INT2 only in the visible history domain.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import probe_history_reuse as reuse
from tools.striped_fixture import raw_to_striped
from csrc.experiments.window_batch_decode.fixture import _bytes
from csrc.experiments.c16_member_pipeline.generate import sha


def export(path, context=337, bad=-1, small=False):
    import torch
    from oscar_ascend.ops.reference import encode_kv, decode_kv, attention
    if path.exists() or context not in ((65,337) if small else (337,1341)) or bad not in (-1,80,96,336) or bad>=context:
        raise ValueError('fresh path and declared C16 case required')
    tokens,recent=(336,1) if small else (672,256)
    acceptance = json.loads((ROOT / 'configs/acceptance.json').read_text())
    if acceptance['fused_attention'] != {'atol': .005, 'rtol': .005}:
        raise ValueError('frozen tolerance changed')
    torch.set_num_threads(min(4, torch.get_num_threads())); logical = {}
    def observe(r,b,q,k,v,ck,cv): logical[r] = (k.clone(), v.clone())
    spec = reuse.Shape('c16_pipeline', (tokens,), (context,), 256, 1, 1, True,
        padded_tokens=tokens, recent_tokens=recent, corrupt_metadata=bad>=0,
        corrupt_position=bad if bad>=0 else 100)
    fixture = reuse.make_fixture(torch, spec, on_request=observe)
    tensors = fixture['cpu']; tensors['raw'] = raw_to_striped(fixture)
    k,v = logical[0]; rk = reuse._hadamard(torch,256)
    dk,dv = decode_kv(encode_kv(k.float()@rk,v.float()@tensors['rv']),256)
    dk,dv = dk@rk.T,dv@tensors['rv'].T
    output = torch.empty((tokens,6,256)); lse = torch.empty((tokens,6))
    for query in range(tokens):
        cut = min(context,max(64,context+query+1-recent))
        kk,vv = k.float().clone(),v.float().clone()
        kk[64:cut],vv[64:cut] = dk[64:cut],dv[64:cut]
        result = attention(tensors['q'][query:query+1],kk,vv,scale=.0625,causal=False)
        output[query],lse[query] = result.output[0],result.lse[0]
    tensors.update(expected_output=output,expected_lse=lse,
        expected_positions=torch.arange(context,context+tokens,dtype=torch.int64))
    window=64+recent+3
    attrs = [tokens,6,1,256,1,tensors['table'].shape[1],tokens*3,reuse.BLOCK_TOKENS,
        fixture['blocks'],reuse.PREFIX,fixture['stride'],window*256,window,64,recent,3,1]
    path.mkdir(parents=True); hashes = {}
    for name,value in tensors.items():
        data=_bytes(value);(path/(name+'.bin')).write_bytes(data);hashes[name]=sha(data)
    (path/'shape.txt').write_text(' '.join(map(str,attrs))+'\n')
    (path/'bad_position.txt').write_text(str(bad)+'\n')
    (path/'tolerance.txt').write_text('0.005 0.005\n')
    meta={'attrs':attrs,'context':context,'bad_position':bad,'input_sha256':hashes,
        'scope':f'Q{tokens}/C{context}/R1/Hq6/Hkv1/D256/S1/20core recent{recent}, all{tokens*3} tasks, source0+1; source2 legally suppressed',
        'valid_numeric_oracle':bad<0,'expected_mature_cluster_members':16,
        'declared_deadlines':{'CPU_each_arm_s':120,'CAModel_each_arm_s':900},
        'NPU':'not_run','whole_service':'not_established'}
    (path/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context',type=int,default=337);p.add_argument('--bad-position',type=int,default=-1)
    p.add_argument('--small',action='store_true')
    a=p.parse_args();m=export(a.output,a.context,a.bad_position,a.small)
    print(json.dumps({'status':'host_oracle_only','output':str(a.output),'context':m['context']}))
