"""Archive #126/#145/#148/#154/#157; independent BF16 cluster inputs.

Natural INT2 and a reversible striped copy coexist only offline. Actual
prepare tasks, not candidate outputs, will drive the independent BF16 oracle.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import probe_history_reuse as reuse
from tools.striped_fixture import raw_to_striped

CASES=(
    ('c4_one_KV','c4',84,320,1,1,1,False,-1,-1),
    ('c16_one_KV','c16',336,320,1,1,1,False,-1,-1),
    ('c16_two_KV','c16',336,337,1,1,1,False,-1,-1),
    ('c4_C1_hkv2_s3','c4',5,337,256,2,3,False,-1,-1),
    ('c16_bad80','c16',336,337,1,1,1,True,80,-1),
    ('c16_last_QR','c16',336,337,1,1,1,True,-1,315),
)


def export(root):
    import torch
    if root.exists():raise ValueError('fresh declared fixture directory required')
    torch.set_num_threads(1);root.mkdir(parents=True);cases=[]
    for name,family,qlen,context,recent,hk,splits,invalid,bad,qr in CASES:
        n=8 if qlen==5 else qlen
        spec=reuse.Shape(name,(qlen,),(context,),256,hk,splits,qlen>=84,
            recent_tokens=recent,padded_tokens=n,corrupt_metadata=bad>=0,corrupt_position=bad if bad>=0 else 80,
            corrupt_qr=qr>=0,corrupt_qr_token=qr if qr>=0 else 0)
        fixture=reuse.make_fixture(torch,spec);data=fixture['cpu'];data['raw_striped']=raw_to_striped(fixture)
        positions=torch.full((n,),-1,dtype=torch.int64);positions[:qlen]=torch.arange(context,context+qlen)
        data['expected_positions']=positions
        checks=[]
        if bad>=0:
            for qb in range(0,qlen,21):checks.append((qb*hk*3*splits+(hk-1)*3*splits,((bad-64)%32)//16,3))
        if qr>=0:checks.append(((qr//21*21)*hk*3*splits,((qr%21)*6)//64,2))
        data['required_status_checks']=torch.tensor(checks,dtype=torch.int64).reshape(-1,3)
        path=root/name;path.mkdir();hashes={}
        for key,value in data.items():
            tensor=value.contiguous();raw=ctypes.string_at(tensor.data_ptr(),tensor.numel()*tensor.element_size())
            (path/(key+'.bin')).write_bytes(raw);hashes[key]=hashlib.sha256(raw).hexdigest()
        cores=20;suppress=int(qlen>=84);clusters=1 if qlen>=84 else 0
        shape=(n,spec.heads,hk,256,context,64,recent,3,reuse.BLOCK_TOKENS,fixture['blocks'],reuse.PREFIX,fixture['stride'],cores,1,splits,data['table'].shape[1])
        (path/'shape.txt').write_text(' '.join(map(str,shape))+'\n')
        (path/'case_flags.txt').write_text(str(int(invalid))+'\n')
        (path/'cluster_flags.txt').write_text(f'{suppress} {clusters} {bad} {qr}\n')
        (path/'tolerance.txt').write_text('0.005 0.005\n')
        cases.append({'name':name,'family':family,'tokens':n,'context':context,'recent':recent,'heads':spec.heads,
            'splits':splits,'cores':cores,'task_count':n*hk*3*splits,'expected_clusters':clusters,
            'source2_suppressed':bool(suppress),'fault':invalid,'bad_metadata_position':bad,'QR_token':qr,
            'input_sha256':hashes,'scope':'all_tasks_and_statuses; declared native-current suppression only for long prefill',
            'oracle':'not_yet_exported; requires actual prepare tasks','NPU':'not_run'})
        print(json.dumps({'fixture':name,'status':'host_input_only','task_count':n*hk*3*splits}),flush=True)
    (root/'cases.json').write_text(json.dumps({'cases':cases,'contract':'oscar_bf16_attention_v2','candidate_values_used':False},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.output)
