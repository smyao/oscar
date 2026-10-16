"""Archive #13/#94/#126/#145/#148/#157: independent BF16 qualification.

D.4: bounded tiled CV with BF16 QK/PV/Rv operands and FP32 accumulators/merge,
never online whole-history restore.
The offline export reads real prepare tasks and produces a predeclared BF16
oracle. It is not a CPU substitute for the separate fail-closed NPU command.
Old FP32 goldens remain untouched and are diagnostic only for this branch.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from .phase import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OPERATORS={
 'q4':'attention_cv_bf16_decode_out',
 'q1':'attention_cv_bf16_decode_q1_out',
 'c4':'attention_cv_bf16_cluster4_out',
 'c16':'attention_cv_bf16_cluster16_out'}
BASELINES={'q4':'attention_cv_hybrid_l1_decode_out','q1':'attention_cv_mdl_decode_q1_out',
 'c4':'attention_cv_member_pipeline_cluster4_out','c16':'attention_cv_l1_reuse_cluster16_out'}
WORKSPACE={'q4':1146880,'q1':1146880,'c4':2494464,'c16':7487488}
CLUSTER={'q4':1,'q1':1,'c4':4,'c16':16}
SOURCE_PATHS=(
 'csrc/kernels/attention_cv_bf16_decode.cpp','csrc/kernels/attention_cv_bf16_cluster4.cpp',
 'csrc/kernels/attention_cv_bf16_cluster16.cpp','csrc/include/oscar_bf16_decode_launch.h',
 'csrc/include/oscar_bf16_cluster_launch.h','csrc/include/oscar_attention_launch.h',
 'csrc/include/oscar_cv_schedule.h','csrc/kernels/oscar_common.h',
 'csrc/kernels/attention_striped_unpack_padded.h','csrc/kernels/attention_striped_unpack32.h',
 'csrc/kernels/attention_striped_unpack.h','csrc/kernels/attention_fast_unpack.h',
 'csrc/kernels/attention_range_mask.h','csrc/kernels/attention_tasks.cpp','csrc/kernels/merge_lse.cpp',
 'tools/bf16_reference.py','configs/acceptance_bf16.json','oscar_ascend/runtime.py',
 'oscar_ascend/integration/impl.py','oscar_ascend/ops/cv_dispatch.py',
 'tools/serving_variants.py','tools/probe_bf16_attention.py')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding(root=ROOT):return {name:digest(root/name) for name in SOURCE_PATHS}

def read_tensor(torch,path,dtype,shape):
    data=path.read_bytes();expected=math.prod(shape)*torch.empty((),dtype=dtype).element_size()
    if len(data)!=expected:raise ValueError('fixture size mismatch '+str(path))
    if expected==0:return torch.empty(shape,dtype=dtype)
    return torch.frombuffer(bytearray(data),dtype=dtype).clone().reshape(shape)

def read_fixture(path):
    """Original full checker binary ABI; no candidate values are inputs."""
    import torch
    from .probe_history_reuse import Shape
    values=list(map(int,(path/'shape.txt').read_text().split()))
    if len(values)!=16:raise ValueError('complete original16-field shape required')
    n,hq,hk,d,context,sink,recent,speculative,block,blocks,prefix,stride,cores,requests,splits,columns=values
    if hq!=6*hk or sink!=64 or speculative!=3 or block%128 or d not in (64,128,256):
        raise ValueError('fixture outside declared Qwen BF16 domain')
    starts=read_tensor(torch,path/'starts.bin',torch.int32,(requests+1,))
    lens=read_tensor(torch,path/'lens.bin',torch.int32,(requests,))
    qlens=tuple(map(int,(starts[1:]-starts[:-1]).tolist()))
    contexts=tuple(int(x-y) for x,y in zip(lens.tolist(),qlens))
    mode='q1' if all(x==1 for x in qlens) else 'q4_or_long'
    shape=Shape(path.name,qlens,contexts,d,hk,splits,False,recent_tokens=recent,
                padded_tokens=n,slot_context=mode=='q1')
    window=sink+recent+speculative
    fields={'q':(torch.bfloat16,(n,hq,d)),'qr':(torch.float32,(n,hq,d)),
      'ck':(torch.bfloat16,(n,hk,d)),'cv':(torch.bfloat16,(n,hk,d)),
      'rv':(torch.float32,(d,d)),'raw':(torch.uint8,(prefix+blocks*stride,)),
      'table':(torch.int32,(requests,columns)),'wk':(torch.bfloat16,(blocks,window,hk,d)),
      'wv':(torch.bfloat16,(blocks,window,hk,d)),'tags':(torch.int64,(blocks,window)),
      'slots':(torch.int64,(n,))}
    cpu={key:read_tensor(torch,path/(key+'.bin'),dtype,dims) for key,(dtype,dims) in fields.items()}
    cpu.update(starts=starts,lens=lens)
    shape.validate()
    return {'spec':shape,'cpu':cpu,'tokens':n,'actual_tokens':sum(qlens),'blocks':blocks,
            'stride':stride,'scale':d**-.5,'cores':cores}

def export_reference(fixture_dir,tasks_path,output_dir):
    import torch,ctypes
    from tools.bf16_reference import fixture_tasks_reference,ACCEPTANCE,CONTRACT
    torch.set_num_threads(min(4,torch.get_num_threads()))
    fixture=read_fixture(fixture_dir);spec=fixture['spec']
    count=fixture['tokens']*spec.kv_heads*3*spec.splits
    tasks=read_tensor(torch,tasks_path,torch.int64,(count,16))
    output_dir.mkdir(parents=True,exist_ok=False)
    report={'contract':CONTRACT,'acceptance_sha256':digest(ACCEPTANCE),
        'oracle_sha256':digest(ROOT/'tools/bf16_reference.py'),
        'actual_tasks_sha256':digest(tasks_path),'all_task_slots':count,
        'fixture_inputs_sha256':{p.name:digest(p) for p in sorted(fixture_dir.iterdir()) if p.is_file() and not p.name.startswith('bf16_')},
        'old_FP32_expected_files_unchanged':True,'device_execution':'not_run','NPU':'not_run'}
    rotation=fixture['cpu']['rv'].to(torch.bfloat16).contiguous()
    rotation_bytes=ctypes.string_at(rotation.data_ptr(),rotation.numel()*rotation.element_size())
    rotation_path=output_dir/'rv_bf16.bin';rotation_path.write_bytes(rotation_bytes)
    report['public_rotation']={'path':'rv_bf16.bin','dtype':'bfloat16','layout':'natural_ND',
                               'sha256':digest(rotation_path),'conversion':'initialization RNE, not candidate output'}
    invalid=(fixture_dir/'case_flags.txt').read_text().strip()=='1'
    if invalid:
        report.update(status='expected_error_domain_no_numeric_golden',
            required_status_checks_sha256=digest(fixture_dir/'required_status_checks.bin'))
    else:
        result=fixture_tasks_reference(fixture,tasks);files={}
        for field in ('partial','lse','output','output_lse'):
            tensor=result[field].contiguous();data=ctypes.string_at(tensor.data_ptr(),tensor.numel()*tensor.element_size())
            path=output_dir/('bf16_expected_'+field+'.bin');path.write_bytes(data)
            files[field]={'path':path.name,'shape':list(tensor.shape),'dtype':'float32','sha256':digest(path)}
        report.update(status='independent_BF16_golden_exported',outputs=files,
            original_FP32_comparison_role='diagnostic, not BF16 acceptance')
    atomic_json(output_dir/'bf16_reference.json',report)
    print(json.dumps({'status':report['status'],'all_task_slots':count,'device_execution':'not_run'}),flush=True)
    return report

def require_real_npu():
    import torch
    try:import torch_npu  # noqa:F401
    except ImportError as error:raise RuntimeError('BF16 gate requires real NPU; no CPU substitution') from error
    if not hasattr(torch,'npu') or not torch.npu.is_available():
        raise RuntimeError('BF16 gate requires real NPU; no CPU substitution')
    return torch

def launch(ops,family,tensors,fixture,buffers,cores,*,baseline=False,merge=True):
    """One public op includes BF16 QK/PV/Rv; then FP32 merge and BF16 copy."""
    from .probe_striped_cache import _launch
    name=(BASELINES if baseline else OPERATORS)[family]
    inputs=tensors
    if baseline and family=='q4':inputs=dict(tensors,rv=tensors['rv_nz'])
    elif not baseline:inputs=dict(tensors,rv=tensors['rv_bf16'])
    _launch(ops,name,inputs,fixture,buffers,cores,CLUSTER[family])
    if merge:
        n,h,d,s=fixture['tokens'],fixture['spec'].heads,fixture['spec'].dim,3*fixture['spec'].splits
        ops.merge_lse_out(buffers['partial'].view(n*h,s,d),buffers['lse'].view(n*h,s),
                          buffers['output'],buffers['output_lse'],buffers['merge_status'])
        buffers['bf16'].copy_(buffers['output'])

def case_plan():
    from dataclasses import replace
    from .probe_history_reuse import Shape
    shapes={
      'q4':Shape('bf16_q4_multikv',(4,4),(641,701),256,1,1,False,padded_tokens=12),
      'q1':Shape('bf16_q1_multikv',(1,1),(641,701),256,1,1,False,padded_tokens=8,slot_context=True),
      'c4':Shape('bf16_c4_mature',(84,),(641,),256,1,1,True,recent_tokens=1,padded_tokens=96),
      'c16':Shape('bf16_c16_mature',(336,),(641,),256,1,1,True,recent_tokens=1,padded_tokens=352)}
    rows=[]
    for family,shape in shapes.items():
        rows.append({'family':family,'shape':shape,'graph':True,'timed':False})
        rows.append({'family':family,'shape':replace(shape,name=shape.name+'_bad_metadata',corrupt_metadata=True,corrupt_position=80),'graph':False,'timed':False})
        rows.append({'family':family,'shape':replace(shape,name=shape.name+'_bad_qr',corrupt_qr=True,corrupt_qr_token=1),'graph':False,'timed':False})
    rows.append({'family':'c4','shape':Shape('bf16_generic_C1_fallback',(5,),(641,),256,1,1,False,padded_tokens=8),'graph':True,'timed':False})
    rows.append({'family':'q4','shape':Shape('bf16_q4_hkv2_s3',(4,4),(641,701),256,2,3,False,padded_tokens=12),'graph':False,'timed':False})
    rows.append({'family':'q1','shape':Shape('bf16_q1_s3',(1,1),(1344,1376),256,1,3,False,padded_tokens=8,slot_context=True),'graph':False,'timed':False})
    return tuple(rows)

def allocate(torch,fixture,device,cores,family):
    from . import probe_history_reuse as reuse
    buffers=reuse._allocate(torch,fixture,device,cores,candidate=CLUSTER[family]>1)
    buffers['workspace']=torch.empty(cores*WORKSPACE[family],dtype=torch.uint8,device=device)
    n,h,d=fixture['tokens'],fixture['spec'].heads,fixture['spec'].dim
    buffers.update(output=torch.empty(n*h,d,dtype=torch.float32,device=device),
      output_lse=torch.empty(n*h,dtype=torch.float32,device=device),
      merge_status=torch.empty(n*h,dtype=torch.int32,device=device),
      bf16=torch.empty(n*h,d,dtype=torch.bfloat16,device=device))
    return buffers

def make_device_inputs(torch,fixture,device):
    from .striped_fixture import raw_to_striped
    cpu=fixture['cpu'];values=dict(cpu,raw=raw_to_striped(fixture))
    # Both legacy FP32 baselines and the new family receive their declared
    # rotation layout. These conversions complete during fixture initialization.
    values['rv_nz']=cpu['rv'].reshape(256,32,8).permute(1,0,2).contiguous().view(256,256)
    values['rv_bf16']=cpu['rv'].to(torch.bfloat16).contiguous()
    return {name:value.to(device) for name,value in values.items()}

def check_outputs(torch,fixture,buffers,golden,tasks_before,baseline=None):
    from . import probe_history_reuse as reuse
    from tools.bf16_reference import error_metrics,compare_results
    from oscar_ascend.ops.reference import AttentionResult
    invalid=fixture['spec'].corrupt_metadata or fixture['spec'].corrupt_qr
    if not torch.equal(buffers['tasks'].cpu(),tasks_before):raise RuntimeError('BF16 kernel modified prepared tasks')
    if invalid:
        if baseline is None or not torch.equal(buffers['status'],baseline['status']):
            raise RuntimeError('BF16 exact error status/lane domain changed')
        if not bool(buffers['status'].ne(0).any()):raise RuntimeError('expected error not reported')
        for name in ('partial','lse'):
            if not torch.equal(torch.isnan(buffers[name]),torch.isnan(baseline[name])):
                raise RuntimeError('invalid-output classification changed '+name)
        return {'status':'passed','precision':'exact_original_error_codes_and_classification'}
    for name in ('status','merge_status'):
        if bool(buffers[name].ne(0).any()):raise RuntimeError('nonzero valid '+name)
    details={}
    n,h,d=fixture['tokens'],fixture['spec'].heads,fixture['spec'].dim
    for name,want in (('partial',golden['partial']),('lse',golden['lse']),
                      ('output',golden['output'].view(n*h,d)),('output_lse',golden['output_lse'].view(n*h))):
        details[name]=error_metrics(buffers[name].cpu(),want)
        if not details[name]['passed']:raise RuntimeError('new BF16 oracle mismatch '+name+': '+str(details[name]))
    # Copy must equal the actual FP32 result cast, not a second BF16 oracle
    # tolerance which could conceal a wrong output conversion.
    if not torch.equal(buffers['bf16'],buffers['output'].to(torch.bfloat16)):
        raise RuntimeError('final BF16 copy differs from actual output conversion')
    dead=(fixture['cpu']['slots']<0)
    if bool(dead.any()):
        if bool(buffers['partial'].cpu()[dead].view(torch.int32).ne(0).any()) or not bool(torch.isneginf(buffers['lse'].cpu()[dead]).all()):
            raise RuntimeError('padding output/LSE not fully initialized')
    if baseline is not None:
        details['old_FP32_diagnostic_only']={name:error_metrics(buffers[name].cpu(),baseline[name].cpu()) for name in ('output','output_lse')}
    return {'status':'passed','precision':'independent_BF16_oracle_all_outputs','detail':details}

def run_case(torch,ops,case,device,cores):
    from dataclasses import replace
    from . import probe_history_reuse as reuse
    from . import probe_mdl_decode as mdl
    from .probe_striped_cache import _store_bits_equal
    from tools.bf16_reference import fixture_tasks_reference
    family=case['family'];fixture=reuse.make_fixture(torch,case['shape'],rotation_mode='nontrivial')
    tensors=make_device_inputs(torch,fixture,device)
    old,new=(allocate(torch,fixture,device,cores,family) for _ in range(2))
    reuse._prepare(torch,ops,tensors,fixture,old);new['tasks']=old['tasks'];new['positions']=old['positions']
    tasks=old['tasks'].cpu();invalid=case['shape'].corrupt_metadata or case['shape'].corrupt_qr
    golden=None if invalid else fixture_tasks_reference(fixture,tasks)
    inputs={name:value.clone() for name,value in tensors.items()}
    def execute(buffers,baseline=False):launch(ops,family,tensors,fixture,buffers,cores,baseline=baseline,merge=not invalid)
    for b,is_old in ((old,True),(new,False)):mdl.poison(torch,b);execute(b,is_old)
    torch.npu.synchronize()
    check=check_outputs(torch,fixture,new,golden,tasks,old)
    if not all(_store_bits_equal(torch,inputs[name],value) for name,value in tensors.items()):raise RuntimeError('readonly input mutated')
    if not invalid and CLUSTER[family]>1:
        eligible=int(new['cluster_stats'][:,0].sum().cpu())
        if case['shape'].expect_clusters and eligible<=0:raise RuntimeError('mature BF16 cluster not exercised')
        if not case['shape'].expect_clusters and eligible:raise RuntimeError('generic case unexpectedly formed cluster')
    timing=None  # User2026-10-16: correctness/system validation only; no Events.
    graph_report=None
    if case['graph']:
        addresses={name:value.data_ptr() for name,value in tensors.items()}
        addresses.update({name:value.data_ptr() for name,value in new.items() if value is not None})
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph,capture_error_mode='thread_local',auto_dispatch_capture=True):execute(new)
        states=[]
        for state in ('original','changed_query','changed_context'):
            if state=='changed_query':
                fixture['cpu']['q'][0].neg_();fixture['cpu']['qr'][0].neg_()
                tensors['q'].copy_(fixture['cpu']['q']);tensors['qr'].copy_(fixture['cpu']['qr'])
            elif state=='changed_context':
                changed=replace(case['shape'],contexts=tuple(x+32 for x in case['shape'].contexts))
                fixture=reuse.make_fixture(torch,changed,rotation_mode='nontrivial')
                fresh=make_device_inputs(torch,fixture,device)
                for name,value in fresh.items():
                    if value.shape!=tensors[name].shape:raise RuntimeError('graph fixture allocation changed '+name)
                    tensors[name].copy_(value)
                reuse._prepare(torch,ops,tensors,fixture,new);tasks=new['tasks'].cpu()
            golden=fixture_tasks_reference(fixture,tasks)
            mdl.poison(torch,new);graph.replay();torch.npu.synchronize()
            check_outputs(torch,fixture,new,golden,tasks)
            captured=mdl.snapshot(new);mdl.poison(torch,new);execute(new);torch.npu.synchronize();mdl.same_bits(torch,captured,new)
            states.append({'state':state,'new_BF16_oracle':'passed','eager_replay_bitwise':'passed'})
        current={name:value.data_ptr() for name,value in tensors.items()};current.update({name:value.data_ptr() for name,value in new.items() if value is not None})
        if current!=addresses:raise RuntimeError('graph addresses changed')
        graph_report={'capture':'passed','replay':'passed','same_addresses':True,'states':states}
    return {'case':case['shape'].name,'family':family,'operator':OPERATORS[family],
            'status':'passed',
            'all_tasks':tasks.shape[0],'check':check,'timing':timing,'graph':graph_report}

def validate_evidence(report):
    required={'status':'passed','contract':'oscar_bf16_attention_v2','device_completion':'passed',
              'precision':'passed','error_domain':'passed','graph_capture':'passed','graph_replay':'passed','performance':'not_run_user_requested'}
    bad=[k for k,v in required.items() if report.get(k)!=v]
    if report.get('acceptance_sha256')!=digest(ROOT/'configs/acceptance_bf16.json'):bad.append('acceptance_sha256')
    if report.get('oracle_sha256')!=digest(ROOT/'tools/bf16_reference.py'):bad.append('oracle_sha256')
    if report.get('probe_sha256')!=digest(__file__):bad.append('probe_sha256')
    if report.get('operator_families')!=OPERATORS:bad.append('operator_families')
    if report.get('source_binding')!=source_binding():bad.append('source_binding')
    rows=report.get('cases',[]);plan=case_plan()
    if len(rows)!=len(plan):return bad+['complete_case_plan']
    for row,case in zip(rows,plan):
        if row.get('case')!=case['shape'].name or row.get('family')!=case['family'] or row.get('status')!='passed':bad.append(case['shape'].name)
        expected_tasks=(case['shape'].padded_tokens or sum(case['shape'].qlens))*case['shape'].kv_heads*3*case['shape'].splits
        if row.get('all_tasks')!=expected_tasks or row.get('operator')!=OPERATORS[case['family']] or row.get('check',{}).get('status')!='passed':bad.append(case['shape'].name+'.task_operator')
        if case['graph']:
            graph=row.get('graph') or {}
            if (graph.get('capture')!='passed' or graph.get('replay')!='passed' or graph.get('same_addresses') is not True or
                [x.get('state') for x in graph.get('states',[])]!=['original','changed_query','changed_context'] or
                any(x.get('new_BF16_oracle')!='passed' or x.get('eager_replay_bitwise')!='passed' for x in graph.get('states',[]))):bad.append(case['shape'].name+'.graph')
        if row.get('timing') is not None:bad.append(case['shape'].name+'.unexpected_timing')
    return bad

def run_npu(config_path=ROOT/'configs/target.json'):
    import gc
    from . import probe_history_reuse as reuse
    from tools.bf16_reference import load_acceptance,ACCEPTANCE,CONTRACT
    load_acceptance();config_path=Path(config_path).resolve();config=json.loads(config_path.read_text())
    if config.get('experimental_bf16_attention') is not True:
        raise RuntimeError('BF16 gate requires experimental_bf16_attention=True in the effective serving config')
    reuse._active_device(config)
    torch=require_real_npu();torch.npu.set_device(0);torch.set_num_threads(min(4,torch.get_num_threads()))
    from .build_ops import normalize_soc
    if normalize_soc(torch.npu.get_device_name(0))!=config['soc_version']:
        raise RuntimeError('BF16 probe target SoC differs from compiled contract')
    from oscar_ascend.ops.loader import validate_build_artifacts,require_capabilities
    manifest=validate_build_artifacts(ROOT/'build/ascendc/build_manifest.json')
    require_capabilities(set(OPERATORS.values())|set(BASELINES.values())|{'prepare_attention_tasks_out','merge_lse_out'})
    bindings=source_binding();config_hash=digest(config_path)
    cores=reuse._core_count(torch,config);rows=[]
    for case in case_plan():
        row=run_case(torch,torch.ops.oscar_ascend_ops,case,torch.device('npu:0'),cores);rows.append(row)
        print('BF16_PROBE '+json.dumps({'case':row['case'],'status':row['status'],'operator':row['operator']}),flush=True)
        gc.collect();torch.npu.empty_cache()
        if row['status']!='passed':break
    if source_binding()!=bindings or digest(config_path)!=config_hash:
        raise RuntimeError('BF16 source/config changed during the device gate')
    report={'status':'passed' if len(rows)==len(case_plan()) and all(x['status']=='passed' for x in rows) else 'failed',
      'contract':CONTRACT,'acceptance_sha256':digest(ACCEPTANCE),'oracle_sha256':digest(ROOT/'tools/bf16_reference.py'),
      'probe_sha256':digest(__file__),
      'source_binding':bindings,'effective_config_sha256':config_hash,
      'artifact_signature':manifest['signature'],'artifact_sha256':manifest['sha256'],'operator_families':OPERATORS,
      'cases':rows,'device_completion':'passed','precision':'passed','error_domain':'passed','graph_capture':'passed',
      'graph_replay':'passed','performance':'not_run_user_requested',
      'old_FP32_outputs':'diagnostic_only; exact metadata/error protocol retained','model_quality':'not_established','native_E2E_gt30pct':'not_established'}
    bad=validate_evidence(report)
    if bad:report.update(status='failed',evidence_mismatches=bad)
    return report

def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export-reference')
    for name in ('fixture-dir','tasks','output-dir'):e.add_argument('--'+name,type=Path,required=True)
    n=sub.add_parser('npu');n.add_argument('--output',type=Path,required=True)
    n.add_argument('--config',type=Path,default=ROOT/'configs/target.json')
    a=p.parse_args(argv)
    if a.command=='export-reference':export_reference(a.fixture_dir,a.tasks,a.output_dir);return 0
    try:
        report=run_npu(a.config)
        atomic_json(a.output,report);return 0 if report['status']=='passed' else 2
    except Exception as error:
        import traceback
        traceback.print_exc()
        atomic_json(a.output,{'status':'failed','error':str(error),'NPU':'not_passed','contract':'oscar_bf16_attention_v2'})
        return 2
if __name__=='__main__':raise SystemExit(main())
