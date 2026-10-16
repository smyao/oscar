"""Archive #55-69/#126/#145/#148/#153/#154/#157: full attention research gate.

Mirror integration/impl.py's actual bounded physical-page path, including
external native-current FIA on prefill, read-before-write cache publication,
and the unchanged frozen guards. No RuntimeProvider or serving-policy bypass.
All device timing is called only after the probe's global numerical barrier.
Cache restoration/poisoning is diagnostic setup outside each event interval.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import statistics

from oscar_ascend.integration import current_attention as current
from .experiments.bf16_direct_decode import CONTRACT
from oscar_ascend.integration.guard_diag import guard_diag_enabled, run_status_guard

SCOPE = 'complete_attention_eager_prepare_rotate_cv_native_current_merge_copy_store_guard_device_events'
STORAGE_KEYS = ('raw','wk','wv','tags')


def config_sha256(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def execution_contract(config,cores):
    from oscar_ascend.runtime import WorkspaceGeometry
    # The independent process has no engine-provided native token budget.
    # Never substitute a guessed capacity for that runtime value.
    if config.get('use_native_token_budget',False) is not False:
        raise ValueError('full attention requires an explicit engine token budget')
    budget=config.get('max_num_batched_tokens')
    captures=config.get('compilation_config',{}).get('cudagraph_capture_sizes',[])
    if type(budget) is not int or budget<=0 or not isinstance(captures,list) or any(type(v) is not int or v<=0 for v in captures):
        raise ValueError('full attention requires valid explicit budget/capture sizes')
    geometry=WorkspaceGeometry(max(budget,max(captures,default=0)),6,1,256,
        config.get('attention_splits',1),cores,16,
        history_token_capacity=config.get('history_token_capacity'),c4_member_pipeline=True,bf16_history=True)
    if config.get('rotation_method','hadamard')!='hadamard' or config.get('sink_tokens',64)!=64 or config.get('recent_tokens',256)!=256:
        raise ValueError('fixed full attention fixtures require Hadamard/sink64/recent256')
    settings={'rotate_operator':'rotate_vector_init_out' if config.get('experimental_quantization_bundle',False) else 'rotate_out',
              'hadamard':True,'k_clip_ratio':float(config.get('k_clip_ratio',0.0)),
              'v_clip_ratio':float(config.get('v_clip_ratio',0.0)),'guard_host_sync':False}
    return {'operand_contract':CONTRACT,'effective_config_sha256':config_sha256(config),'cube_cores':cores,
            'engine_budget':budget,'capture_sizes':captures,'workspace_geometry':asdict(geometry),
            'cv_arena_bytes':{'baseline':replace(geometry,bf16_history=False).cv_bytes,'candidate':geometry.cv_bytes},
            'settings':settings,'synthetic_physical_pages':True,'physical_block_tokens':512,
            'raw_ssm_offset_bytes':64,'window_stride_tokens':323,
            'partial_buffers':'production_active_view_shapes',
            'full_graph_performance':'not_established','whole_model_performance':'not_established'}


def validate_execution(contract):
    from oscar_ascend.runtime import WorkspaceGeometry
    try:
        geometry=WorkspaceGeometry(**contract['workspace_geometry'])
        budget=contract['engine_budget'];captures=contract['capture_sizes'];settings=contract['settings']
        if type(budget) is not int or budget<=0 or not isinstance(captures,list) or any(type(v) is not int or v<=0 for v in captures):return False
        if (geometry.tokens!=max(budget,max(captures,default=0)) or
            (geometry.query_heads,geometry.kv_heads,geometry.head_dim,geometry.history_cluster_size)!=(6,1,256,16) or
            not geometry.c4_member_pipeline or not geometry.bf16_history or geometry.cube_cores!=contract['cube_cores']):return False
        if contract['cv_arena_bytes']!={'baseline':replace(geometry,bf16_history=False).cv_bytes,'candidate':geometry.cv_bytes}:return False
        if (settings['guard_host_sync'] is not False or settings['hadamard'] is not True or
            settings['rotate_operator'] not in {'rotate_out','rotate_vector_init_out'} or
            any(type(settings[key]) not in (float,int) or not math.isfinite(settings[key]) or not 0<=settings[key]<1 for key in ('k_clip_ratio','v_clip_ratio'))):return False
        return (contract.get('operand_contract')==CONTRACT and isinstance(contract['effective_config_sha256'],str) and len(contract['effective_config_sha256'])==64 and
                contract['synthetic_physical_pages'] is True and contract['physical_block_tokens']==512 and
                contract['raw_ssm_offset_bytes']==64 and contract['window_stride_tokens']==323 and
                contract['partial_buffers']=='production_active_view_shapes' and
                contract['full_graph_performance']=='not_established' and contract['whole_model_performance']=='not_established')
    except (KeyError,TypeError,ValueError):return False


def production_case(case,execution):
    from oscar_ascend.runtime import WorkspaceGeometry
    if not validate_execution(execution):raise ValueError('invalid full attention execution contract')
    geometry=WorkspaceGeometry(**execution['workspace_geometry'])
    # Preserve all original operator/graph fixtures; use distinct full-path
    # target-head shapes and actual runtime split/selector geometry.
    shape=replace(case.shape,kv_heads=1)
    if case.route=='c16':shape=replace(shape,qlens=(8192,),padded_tokens=8208)
    elif case.route=='balanced':shape=replace(shape,padded_tokens=512)
    tokens=shape.padded_tokens or sum(shape.qlens)
    splits=geometry.splits_for_tokens(tokens)
    if tokens>geometry.history_tokens or tokens*splits>geometry.history_tokens*geometry.splits:
        raise ValueError('full attention active partials exceed production history arena capacity')
    shape=replace(shape,splits=splits)
    return replace(case,shape=shape)


def bounded_dense_oracle(torch,fixture,*,block_rows=32):
    """All queries/heads via unchanged oracle, never an Nq x Nq allocation."""
    from oscar_ascend.ops.reference import attention
    if type(block_rows) is not int or block_rows<=0:raise ValueError('invalid oracle query block')
    fixture['expected']={}
    for begin,q,k,v,ck,cv,dk,dv in fixture['logical'].values():
        local=0;context=len(k);recent=fixture['spec'].recent_tokens
        while local<len(q):
            cut=min(context,max(64,context+local+1-recent))
            # Exact-window frontier changes per query; after it reaches the
            # cached end, several query rows share the same restored history.
            stop=min(len(q),local+(block_rows if cut==context else 1))
            kk,vv=k.clone(),v.clone();kk[64:cut]=dk[64:cut];vv[64:cut]=dv[64:cut]
            result=attention(q[local:stop],torch.cat((kk,ck[:stop])),torch.cat((vv,cv[:stop])),
                             scale=fixture['scale'],causal=True)
            for offset in range(stop-local):
                fixture['expected'][begin+local+offset]=(result.output[offset],result.lse[offset])
            local=stop


def case_policy(case):
    """These are bounded, unsplit step fixtures, not whole-model simulation."""
    if case.route not in {'mdl','mdl_q1','c4','c16','general','balanced'} or not case.timed:
        raise ValueError('full attention timing requires one of the six fixed timed cases')
    if any(context <= 0 for context in case.shape.contexts):
        raise ValueError('full attention pair requires existing compressed history')
    native = case.route in {'c4','c16','general','balanced'}
    if bool(case.shape.slot_context) != (case.route == 'mdl_q1'):
        raise ValueError('later MTP full pipeline must preserve slot-context preparation')
    from oscar_ascend.ops.cv_dispatch import select_cv_op
    from .probe_bf16_history import ROUTES
    selected=select_cv_op(16,case.shape.heads,case.shape.kv_heads,
                         case.shape.padded_tokens or sum(case.shape.qlens),max(case.shape.qlens),
                         q1_draft=case.shape.slot_context,fast_unpack=True,mixed_cv=True,
                         striped_cache=True,decode_bundle=True,c4_member_pipeline=True,mdl_decode=True)
    if selected!=ROUTES[case.route][0]:
        raise ValueError('full attention fixture is not selected by the actual production eligibility')
    # All six fixed cases avoid the mixed splitter: long suffix 17 is below
    # four complete query groups (84 at GQA6), other cases are uniform/single.
    if case.route=='balanced' and max(case.shape.qlens)>=4*(128//6):
        raise ValueError('mixed fixture requires the actual production partition path')
    return {'native_current':native,
            'stage':'prefill' if native else 'later_mtp' if case.shape.slot_context else 'target_decode',
            'current_source':'native_fia' if native else 'ascendc_cv',
            'capture':False,'dummy':False,'mixed_partition':False,'current_only':False}


def launch_pipeline(ops,fixture,tensors,buffers,cores,*,name,cluster,policy,settings):
    """One real complete step. No sync/readback/reset occurs inside this chain."""
    from .probe_bf16_history import launch
    from .probe_history_reuse import BLOCK_TOKENS,PREFIX,SINK,SPECULATIVE
    spec=fixture['spec'];n=fixture['tokens'];h=spec.heads;hk=spec.kv_heads;s=spec.splits
    q,k,v=(tensors[key] for key in ('q','ck','cv'))
    # Fixture projections are contiguous, the production projection fast path.
    # Slot tensors are already int64, so no int32->int64 copy is applicable.
    args=(tensors['starts'],tensors['lens'],tensors['slots'],buffers['tasks'],buffers['positions'],
          h,hk,SINK,spec.recent_tokens,s)
    if spec.slot_context:ops.prepare_attention_tasks_out(*args,tensors['table'],True)
    else:ops.prepare_attention_tasks_out(*args,None,False)
    if policy['native_current']:
        current.suppress_current_source_tasks(buffers['tasks'],n,hk,s)
    getattr(ops,settings['rotate_operator'])(q,tensors['rk_transpose'],tensors['qr'],
                                           buffers['rotate_status'],settings['hadamard'],tensors['slots'])
    launch(ops,name,tensors,fixture,buffers,cores,cluster,merge=False)
    if policy['native_current']:
        active=fixture['actual_tokens']
        current.guard_current_slots(ops,buffers['merge_status'].view(n,h),tensors['slots'],active)
        out,lse=current.native_current_partial(q[:active],k[:active],v[:active],fixture['current_cumulative'],
                                               heads=h,kv_heads=hk,scale=fixture['scale'])
        current.write_current_partial(buffers['partial'],buffers['lse'],out,lse,active,s)
    ops.merge_lse_out(buffers['partial'].view(n*h,3*s,256),buffers['lse'].view(n*h,3*s),
                      buffers['output'],buffers['output_lse'],buffers['merge_status'])
    buffers['bf16'].copy_(buffers['output'])
    # The same raw/window/tag allocations read by CV are written only now.
    ops.rotate_clip_store_striped_out(k,v,tensors['rk_transpose'],tensors['rv_transpose'],
        tensors['slots'],buffers['positions'],tensors['raw'],tensors['wk'],tensors['wv'],tensors['tags'],
        buffers['store_status'],BLOCK_TOKENS,fixture['blocks'],PREFIX,fixture['stride'],
        SINK,spec.recent_tokens+SPECULATIVE,settings['k_clip_ratio'],settings['v_clip_ratio'],settings['hadamard'])
    run_status_guard(ops,'bf16_history.full_attention',
        [('attention',buffers['status']),('rotate',buffers['rotate_status']),
         ('merge',buffers['merge_status']),('store',buffers['store_status'])],capture=False)


def build_pair(torch,ops,case,device,cores,config,first_receipt):
    from . import probe_bf16_history as probe
    if getattr(device,'type',None)!='npu':raise RuntimeError('full attention gate requires real NPU')
    fixture=probe.make_fixture(torch,case,dense_oracle=False)
    bounded_dense_oracle(torch,fixture)
    previous=first_receipt.get('full_attention')
    if previous and fixture['input_sha256']!=previous['input_sha256']:
        raise RuntimeError('full attention fixture differs from its numerical pass')
    if case.rotation!='hadamard':raise RuntimeError('fixed full attention cases require Hadamard rotations')
    rk=probe.reuse._hadamard(torch,256)
    total=0;cumulative=[]
    for length in case.shape.qlens:total+=length;cumulative.append(total)
    fixture['current_cumulative']=tuple(cumulative)
    execution=execution_contract(config,cores)
    settings=dict(execution['settings']);settings['guard_host_sync']=guard_diag_enabled()
    fixture['execution']=execution
    pair=[]
    for candidate,name in enumerate(probe.ROUTES[case.route][:2]):
        tensors={key:value.to(device) for key,value in fixture['cpu'].items()}
        tensors['rk_transpose']=rk.T.contiguous().to(device)
        tensors['rv_transpose']=fixture['cpu']['rv'].T.contiguous().to(device)
        buffers=probe.allocate(torch,fixture,device,cores,case.route,candidate=bool(candidate))
        # Runtime reserves the maximal C16 arena once, shared by every route.
        buffers['workspace']=torch.empty(execution['cv_arena_bytes']['candidate' if candidate else 'baseline'],dtype=torch.uint8,device=device)
        buffers['rotate_status']=torch.empty((fixture['tokens'],case.shape.heads),dtype=torch.int32,device=device)
        buffers['store_status']=torch.empty((fixture['tokens'],case.shape.kv_heads),dtype=torch.int32,device=device)
        prepared=probe.reuse._prepare(torch,ops,tensors,fixture,buffers)
        if candidate==0:fixture['initial_task_sha256']=prepared['task_sha256']
        if prepared['task_sha256']!=fixture['initial_task_sha256'] or (previous and prepared['task_sha256']!=previous['initial_task_sha256']):
            raise RuntimeError('full attention initial tasks differ between arms or its numerical pass')
        pair.append({'name':name,'tensors':tensors,'buffers':buffers,
                     'initial_storage':{key:tensors[key].clone() for key in STORAGE_KEYS}})
    return fixture,pair,case_policy(case),settings


def reset_arm(torch,arm):
    from . import probe_mdl_decode as mdl
    for key in STORAGE_KEYS:arm['tensors'][key].copy_(arm['initial_storage'][key])
    mdl.poison(torch,arm['buffers'])
    arm['buffers']['rotate_status'].fill_(-99);arm['buffers']['store_status'].fill_(-99)
    arm['tensors']['qr'].fill_(float('nan'))


def snapshot_arm(arm):
    from .probe_bf16_history import snapshot
    return {**snapshot(arm['buffers']),'rotate_status':arm['buffers']['rotate_status'].clone(),
            'store_status':arm['buffers']['store_status'].clone(),
            **{key:arm['tensors'][key].clone() for key in (*STORAGE_KEYS,'qr')},
            'tasks':arm['buffers']['tasks'].clone(),'positions':arm['buffers']['positions'].clone()}


def same_arm(torch,reference,arm):
    from .probe_striped_cache import _store_bits_equal
    values={**arm['buffers'],**arm['tensors']}
    for key,value in reference.items():
        if not _store_bits_equal(torch,value,values[key]):raise RuntimeError('full attention repeat changed '+key+' bits')


def full_oracle(torch,fixture,tasks,policy):
    from .probe_bf16_history import partial_oracle
    from oscar_ascend.ops.reference import attention
    gold=partial_oracle(torch,fixture,tasks)
    if policy['native_current']:
        # Source2 split0 now holds the entire native current causal partial;
        # every other current split remains the CV-written zero/-inf.
        source=2*fixture['spec'].splits
        for begin,q,k,v,ck,cv,dk,dv in fixture['logical'].values():
            for first in range(0,len(q),32):
                stop=min(len(q),first+32)
                result=attention(q[first:stop],ck[:stop],cv[:stop],scale=fixture['scale'],causal=True)
                gold['partial'][begin+first:begin+stop,:,source]=result.output
                gold['lse'][begin+first:begin+stop,:,source]=result.lse
    return gold


def check_pair(torch,case,fixture,pair,policy,acceptance):
    from . import probe_bf16_history as probe
    old,new=(arm['buffers'] for arm in pair)
    old_tasks=old['tasks'].cpu();task_hash=probe.reuse._fingerprint({'tasks':old_tasks})
    if not torch.equal(old_tasks,new['tasks'].cpu()):raise RuntimeError('full attention changed task bytes between arms')
    detail=probe.compare_pair(torch,case,fixture,old,new,task_hash,acceptance,
                              oracle=full_oracle(torch,fixture,old_tasks,policy))
    if policy['native_current']:
        part=2*fixture['spec'].splits
        for key in ('partial','lse'):
            if not probe.shared._store_bits_equal(torch,old[key][:,:,part],new[key][:,:,part]):
                raise RuntimeError('unchanged native current partial changed '+key+' bits')
    for key in (*STORAGE_KEYS,'qr'):
        if not probe.shared._store_bits_equal(torch,pair[0]['tensors'][key],pair[1]['tensors'][key]):
            raise RuntimeError('full attention changed writer/rotation '+key+' bytes')
    for arm in pair:
        for key in ('rotate_status','store_status'):
            if not bool((arm['buffers'][key]==0).all()):raise RuntimeError('full attention '+key+' failed')
        for key in ('q','ck','cv','rv','table','starts','lens','slots'):
            if not probe.shared._store_bits_equal(torch,fixture['cpu'][key],arm['tensors'][key].cpu()):
                raise RuntimeError('full attention changed immutable input '+key)
        if probe.shared._store_bits_equal(torch,arm['initial_storage']['raw'],arm['tensors']['raw']):
            raise RuntimeError('full attention did not publish current INT2 rows')
    return {'passed':detail['passed'],'precision':detail,'task_sha256':task_hash,
            'writer_bytes_equal':True,'rotation_bytes_equal':True,'immutable_inputs':True,
            'all_status_zero':True,'same_arm_repeat_bitwise':True,
            'native_current_bitwise':True if policy['native_current'] else None}


def numerical_case(torch,ops,case,device,cores,config,acceptance,first_receipt):
    from .probe_bf16_history import ROUTES
    case=production_case(case,execution_contract(config,cores))
    fixture,pair,policy,settings=build_pair(torch,ops,case,device,cores,config,first_receipt)
    for arm in pair:
        reset_arm(torch,arm)
        launch_pipeline(ops,fixture,arm['tensors'],arm['buffers'],cores,name=arm['name'],
                        cluster=ROUTES[case.route][2],policy=policy,settings=settings)
    torch.npu.synchronize()
    detail=check_pair(torch,case,fixture,pair,policy,acceptance)
    for arm in pair:
        reference=snapshot_arm(arm);reset_arm(torch,arm)
        launch_pipeline(ops,fixture,arm['tensors'],arm['buffers'],cores,name=arm['name'],
                        cluster=ROUTES[case.route][2],policy=policy,settings=settings)
        torch.npu.synchronize();same_arm(torch,reference,arm)
    return {'scope':SCOPE,'policy':policy,'settings':settings,'numerical':detail,'timing':None,
            'input_sha256':fixture['input_sha256'],'initial_task_sha256':fixture['initial_task_sha256'],
            'operator_input_sha256':first_receipt['input_sha256'],'operator_task_sha256':first_receipt['task_sha256'],
            'shape':json.loads(json.dumps(asdict(case.shape))), 'execution':fixture['execution'],
            'physical_page_stride_bytes':fixture['stride'],
            'reset_scope':'same_address_raw_window_tags_and_poison_outside_events',
            'whole_model_performance':'not_established','model_quality':'not_established'}


def timing_case(torch,ops,case,device,cores,config,acceptance,first_receipt):
    from .probe_bf16_history import ROUTES,MIN_SPEEDUP
    case=production_case(case,execution_contract(config,cores))
    fixture,pair,policy,settings=build_pair(torch,ops,case,device,cores,config,first_receipt)
    previous=first_receipt['full_attention']
    if settings['guard_host_sync']:
        raise RuntimeError('full attention timing forbids OSCAR_GUARD_HOST_SYNC diagnostics')
    if policy!=previous['policy'] or settings!=previous['settings'] or fixture['execution']!=previous['execution']:
        raise RuntimeError('full attention policy/settings changed after numerical qualification')
    for arm in pair:
        reset_arm(torch,arm)
        launch_pipeline(ops,fixture,arm['tensors'],arm['buffers'],cores,name=arm['name'],
                        cluster=ROUTES[case.route][2],policy=policy,settings=settings)
    torch.npu.synchronize()
    checked=check_pair(torch,case,fixture,pair,policy,acceptance)
    if not checked['passed'] or checked['task_sha256']!=previous['numerical']['task_sha256']:
        raise RuntimeError('full attention numerics/tasks changed before timing')
    references=[snapshot_arm(arm) for arm in pair];samples={arm['name']:[] for arm in pair}
    for repeat in range(7):
        for index in ((0,1) if repeat<2 or repeat%2==0 else (1,0)):
            arm=pair[index]
            # Restore exact initial cache at the same addresses, on the same
            # stream before start.record. Each interval contains one full step.
            reset_arm(torch,arm)
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            start.record()
            launch_pipeline(ops,fixture,arm['tensors'],arm['buffers'],cores,name=arm['name'],
                            cluster=ROUTES[case.route][2],policy=policy,settings=settings)
            end.record();end.synchronize()
            elapsed=float(start.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid full attention device event')
            same_arm(torch,references[index],arm)
            if repeat>=2:samples[arm['name']].append(elapsed)
    a,b=(statistics.median(samples[arm['name']]) for arm in pair)
    return {'scope':SCOPE,'samples_ms':samples,'baseline_ms':a,'candidate_ms':b,'speedup':a/b,
            'minimum_speedup':MIN_SPEEDUP,'warmup':2,'repeats':5,
            'gate':'passed' if a/b>=MIN_SPEEDUP else 'failed'}


def validate_receipt(case,row,*,timed,execution):
    from .probe_bf16_history import ROUTES,MIN_SPEEDUP
    try:case=production_case(case,execution)
    except (ValueError,TypeError,KeyError):return [case.shape.name+'.full_attention.execution']
    receipt=row.get('full_attention') or {};numeric=receipt.get('numerical') or {};bad=[]
    if (receipt.get('scope')!=SCOPE or receipt.get('policy')!=case_policy(case) or
            receipt.get('operator_input_sha256')!=row.get('input_sha256') or
            receipt.get('operator_task_sha256')!=row.get('task_sha256') or
            receipt.get('shape')!=json.loads(json.dumps(asdict(case.shape))) or
            receipt.get('execution')!=execution or receipt.get('settings')!=execution['settings'] or
            receipt.get('physical_page_stride_bytes')!=512*(256//2+8) or
            receipt.get('model_quality')!='not_established' or receipt.get('whole_model_performance')!='not_established' or
            not isinstance(receipt.get('input_sha256'),str) or not receipt['input_sha256'] or
            not isinstance(receipt.get('initial_task_sha256'),str) or not receipt['initial_task_sha256'] or
            receipt.get('reset_scope')!='same_address_raw_window_tags_and_poison_outside_events'):
        bad.append(case.shape.name+'.full_attention.contract')
    for key in ('passed','writer_bytes_equal','rotation_bytes_equal','immutable_inputs','all_status_zero','same_arm_repeat_bitwise'):
        if numeric.get(key) is not True:bad.append(case.shape.name+'.full_attention.'+key)
    if numeric.get('native_current_bitwise')!=(True if case_policy(case)['native_current'] else None):
        bad.append(case.shape.name+'.full_attention.native_current_bits')
    precision=numeric.get('precision') or {}
    if any(precision.get(key)!=value for key,value in {'passed':True,'queries':sum(case.shape.qlens),
        'oracle':'all_actual_queries_raw_space','status_lane_bits':True,'empty_partial_bits':True,'operand_contract':CONTRACT}.items()):
        bad.append(case.shape.name+'.full_attention.oracle')
    comparisons=precision.get('comparisons') or {}
    if set(comparisons)!={'baseline','candidate'}:bad.append(case.shape.name+'.full_attention.oracle_arms')
    for arm in ('baseline','candidate'):
        checks=comparisons.get(arm) or {}
        if set(checks)!={'partial','lse','output','output_lse','bf16','relative_l2'} or any(v.get('passed') is not True for v in checks.values()):
            bad.append(case.shape.name+'.full_attention.'+arm+'.oracle')
    if timed:
        timing=receipt.get('timing') or {};samples=timing.get('samples_ms') or {};names=ROUTES[case.route][:2]
        valid=set(samples)==set(names) and all(isinstance(samples[name],list) and len(samples[name])==5 and
            all(type(value) in (int,float) and math.isfinite(value) and value>0 for value in samples[name]) for name in names)
        if valid:
            a,b=(statistics.median(samples[name]) for name in names)
            valid=a/b>=MIN_SPEEDUP and all(timing.get(key)==value for key,value in {
                'scope':SCOPE,'baseline_ms':a,'candidate_ms':b,'speedup':a/b,'minimum_speedup':MIN_SPEEDUP,
                'warmup':2,'repeats':5,'gate':'passed'}.items())
        if not valid:bad.append(case.shape.name+'.full_attention.device_latency')
    return bad
