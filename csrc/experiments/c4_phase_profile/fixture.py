"""Archive #126/#132-135/#145/#148/#154/#155: C4 prefill partition evidence.

Offline independent oracle. Prepared tasks must come from prepare_cpu.cpp;
Python applies the existing production current-source suppression only.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
N,HQ,HK,D,CORES,CONTEXT=1344,6,1,256,20,769
CASE='c4_q1344_r1_c769_s1'
DECLARED_TOKENS=(336,1344)


def sha(data): return hashlib.sha256(data).hexdigest()


def declared_tokens(tokens):
    if type(tokens) is not int or tokens not in DECLARED_TOKENS:
        raise ValueError('only explicitly declared Q336 or Q1344 are allowed')
    return tokens


def case_name(tokens):return f'c4_q{declared_tokens(tokens)}_r1_c769_s1'


def plan(tokens=N):
    n=declared_tokens(tokens)
    return {'schema_version':1,'case':case_name(n),'tokens':n,'requests':1,'hq':HQ,'hkv':HK,
        'dim':D,'cores':CORES,'splits':1,'context':CONTEXT,'sink':64,'recent':256,
        'task_slots':n*3,'all_prepared_tasks_retained':True,
        'source2':'healthy end=begin, identical to production eager-prefill current-source suppression',
        'oracle_scope':f'all {n} queries; merged source0/history and source1/exact-window only',
        'measurement_scope':('bounded low-occupancy phase diagnosis; not K32 or Q2048 saturation' if n==336
                             else 'original prefill partition; first normal arm timed out at 900s without completion'),
        'prior_q1344_failure':{'timed_out':True,'deadline_seconds':900,'cleanup_complete':True,
            'child_returncode_during_term_cleanup':-11,'completion_and_numeric_outputs':'absent',
            'ticks_usable':False,'kernel_math_error_established':False,'repeat_allowed':False},
        'not_measured':['native current FIA','whole attention wrapper','K32 whole step','end-to-end'],
        'mature_history_range':[64,CONTEXT],'mature_history_kv256_units':3,
        'work_ledger':'must be read from real prepare kernel output; no synthetic task tensor accepted',
        'predeclared_arm_deadline_seconds':900,'deadline_extension_allowed':False,
        'prepare_cpu_deadline_seconds':120,'max_absolute_instrumentation_perturbation':0.05,
        'paired_requirements':['same input bytes','same library','same simulator config','20 actual cores',
            'frozen oracle all rows','partial/lse/status/cluster stats bitwise','same 900s deadline'],
        'explicit_aiv_ub_bytes':157632,'workspace_bytes_per_core':1839104,
        'workspace_total_bytes':1839104*CORES,'profile_bytes_per_core':3072,
        'cluster_stats_bytes_per_core':64,'profile_schema':3,
        'finite_reduce_field':31,
        'finite_reduce_scope':'only original Softmax ReduceSum(scores row,256), V_S wait and Finite check; original preceding PIPE_V excluded',
        'validation':{key:'not_run' for key in ('fixture_export','prepare_cpu','cann_compile',
            'normal_camodel','profile_camodel','marker_completeness','npu','graph','whole_service')}}


def export(directory,tokens=N):
    import torch
    from tools import probe_history_reuse as reuse
    from tools.striped_fixture import raw_to_striped
    from csrc.experiments.window_batch_decode.fixture import _bytes
    from oscar_ascend.ops.reference import attention,encode_kv,decode_kv
    n=declared_tokens(tokens)
    policy=json.loads((ROOT/'configs/acceptance.json').read_text())
    if not policy.get('frozen_before_measurement') or policy['fused_attention']!={'atol':.005,'rtol':.005}:
        raise ValueError('frozen acceptance policy changed')
    if directory.exists(): raise ValueError('fixture output must be new')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    logical={}
    def observe(request,begin,q,k,v,ck,cv): logical[request]=(k.clone(),v.clone())
    shape=reuse.Shape(case_name(n),(n,),(CONTEXT,),D,HK,1,True,padded_tokens=n,recent_tokens=256)
    fixture=reuse.make_fixture(torch,shape,on_request=observe)
    tensors=fixture['cpu'];tensors['raw']=raw_to_striped(fixture)
    old_k,old_v=logical[0];rk=reuse._hadamard(torch,D)
    dk,dv=decode_kv(encode_kv(old_k.float()@rk,old_v.float()@tensors['rv']),D)
    dk,dv=dk@rk.T,dv@tensors['rv'].T
    expected=torch.empty((n,HQ,D),dtype=torch.float32)
    logs=torch.empty((n,HQ),dtype=torch.float32)
    for local in range(n):
        cut=min(CONTEXT,max(64,CONTEXT+local+1-256))
        keys,values=old_k.float().clone(),old_v.float().clone()
        keys[64:cut],values[64:cut]=dk[64:cut],dv[64:cut]
        result=attention(tensors['q'][local:local+1],keys,values,scale=D**-.5,causal=False)
        expected[local],logs[local]=result.output[0],result.lse[0]
    tensors.update(expected_output=expected,expected_lse=logs,
                   expected_positions=torch.arange(CONTEXT,CONTEXT+n,dtype=torch.int64))
    attrs=[n,HQ,HK,D,1,tensors['table'].shape[1],n*3,reuse.BLOCK_TOKENS,
        fixture['blocks'],reuse.PREFIX,fixture['stride'],323*HK*D,323,64,256,3,1]
    directory.mkdir(parents=True)
    hashes={}
    for key,tensor in tensors.items():
        data=_bytes(tensor);(directory/(key+'.bin')).write_bytes(data);hashes[key]=sha(data)
    (directory/'shape.txt').write_text(' '.join(map(str,attrs))+'\n')
    evidence=plan(n);evidence.update(attrs=attrs,scale=D**-.5,input_sha256=hashes,
        acceptance_sha256=sha((ROOT/'configs/acceptance.json').read_bytes()))
    evidence['validation']['fixture_export']='completed_cpu_only'
    (directory/'fixture.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


def task_ledger(rows):
    """Read actual task fields; preserve nonmature groups and all empty tasks."""
    if len(rows)%3:raise ValueError('all 3Q task rows are required')
    n=declared_tokens(len(rows)//3);query_tiles=n//21
    for index,row in enumerate(rows):
        token,kind=divmod(index,3)
        if (len(row)!=16 or row[0]!=token or row[2]!=0 or row[5]!=0 or
                row[6]!=kind or row[7]!=kind or row[8]!=CONTEXT or row[9]!=0 or row[10]!=0):
            raise ValueError('prepared task geometry/status drift')
        want=21 if token%21==0 else 0
        if row[1]!=want: raise ValueError('query leader/follower drift')
        if kind==2 and row[4]!=row[3]: raise ValueError('current source is not suppressed')
    clusters=[];members=set()
    for anchor in range(0,n,84):
        group=[rows[(anchor+j*21)*3] for j in range(4)]
        first=group[0]
        if all(row[1]==21 and row[3]<row[4] and row[4]==CONTEXT and
               row[3:5]==first[3:5] for row in group):
            clusters.append((anchor,first[3],first[4]));members.update(anchor+j*21 for j in range(4))
    owners=[{'history':0,'window':0,'current':0,'clusters':0,'c1_tasks':0} for _ in range(CORES)]
    for index,row in enumerate(rows):
        if row[1]<=0 or (row[7]==0 and row[0] in members): continue
        owner=(row[7]*query_tiles+row[0]//21)%CORES
        owners[owner]['c1_tasks']+=1
        owners[owner][('history','window','current')[row[7]]]+=(row[4]-row[3]+255)//256
    for anchor,begin,end in clusters:
        owner=(anchor//84)%CORES;owners[owner]['clusters']+=1
        owners[owner]['history']+=(end-begin+255)//256
    return {'task_slots':len(rows),'leaders':sum(row[1]>0 for row in rows),
        'followers':sum(row[1]==0 for row in rows),'cluster_count':len(clusters),
        'cluster_members':len(members),'mature_clusters':clusters,
        'kv256_units':{key:sum(owner[key] for owner in owners) for key in ('history','window','current')},
        'owners':owners,'counter_scope':'same task work for AIC/AIV0/AIV1; never sum actor wall times'}


def finish_preparation(fixture,prepared):
    import torch
    from oscar_ascend.integration.current_attention import _empty_current_source_ranges
    from csrc.experiments.window_batch_decode.fixture import _bytes
    evidence=json.loads((fixture/'fixture.json').read_text());n=declared_tokens(evidence['tokens'])
    receipt=json.loads((prepared/'prepare_complete.json').read_text())
    if receipt!={'status':'real_prepare_kernel_completed','cores':20,'all_positions_checked':True,'guards':True}:
        raise ValueError('real CPU task preparation completion missing')
    raw=(prepared/'tasks_before_suppression.bin').read_bytes()
    words=array('q');words.frombytes(raw)
    if sys.byteorder!='little': words.byteswap()
    if len(words)!=n*3*16: raise ValueError('task byte count changed')
    tensor=torch.tensor(words.tolist(),dtype=torch.int64).reshape(n*3,16)
    _empty_current_source_ranges(tensor,n,HK,1)
    data=_bytes(tensor);rows=tensor.tolist();ledger=task_ledger(rows)
    (prepared/'tasks.bin').write_bytes(data)
    ledger.update(origin='real AscendC prepare CPU-debug then production suppression helper',
        before_sha256=sha(raw),after_sha256=sha(data),fixture_sha256=sha((fixture/'fixture.json').read_bytes()),
        source2='all healthy task end=begin; every task retained',attrs=evidence['attrs'])
    (prepared/'task_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    return ledger


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'plan.json')
    p.add_argument('--export',action='store_true');p.add_argument('--prepared',type=Path)
    p.add_argument('--tokens',type=int,choices=DECLARED_TOKENS,default=N)
    args=p.parse_args()
    if args.prepared: value=finish_preparation(args.output,args.prepared)
    elif args.export: value=export(args.output,args.tokens)
    else:
        value=plan(args.tokens);args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'case':case_name(value['attrs'][0]) if 'attrs' in value else value['case'],
                     'scope':'host fixture/ledger only','device_execution':'not_run'}))
