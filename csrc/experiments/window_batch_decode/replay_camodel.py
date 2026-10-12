"""Archive #126/#129/#140/#145/#148/#155/D.4: one full current-SIMD A/B arm.

Runs every prepared source/task at the original fixture core budget. Save all
output bytes immediately after completion, then input/oracle/profile checks.
CAModel ticks are neither target NPU milliseconds nor model quality evidence.
"""
from __future__ import annotations
import argparse
from array import array
import ctypes as C
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import re
import struct
import stat
import sys
import threading
import time
import tomllib


INSTRUCTION_PATTERN=re.compile(r'^core(\d+)\.(cubecore\d+|veccore\d+)\.(instr_log|instr_popped_log)\.dump$')
RESEARCH_TRACE_FREE_BYTES=9*1024**3//2


def trace_policy(budget,cores,profile_counters,completion_only,output):
    """Archive #132-135/#150: isolated full completion logs, unchanged defaults."""
    if not 0<=budget<=64*1024*1024:
        raise ValueError('bounded instruction budget must be 0..64MiB/file')
    result={'research_completion_only':completion_only,'per_file_budget_bytes':budget}
    if completion_only:
        if not profile_counters or cores!=20 or budget<=0:
            raise ValueError('research completion-only requires profile counters, 20 cores and a positive explicit cap')
        if output.stat().st_dev!=Path('/dev/shm').stat().st_dev:
            raise ValueError('research completion-only output must be on the owned /dev/shm tmpfs')
        usage=os.statvfs(output);available=usage.f_bavail*usage.f_frsize
        if available<RESEARCH_TRACE_FREE_BYTES:
            raise ValueError('research completion-only requires at least 4.5GiB free tmpfs before execution')
        result.update(free_tmpfs_bytes_before=available,minimum_free_tmpfs_bytes=RESEARCH_TRACE_FREE_BYTES,
                      raw_cleanup='per_actor_after_gzip_and_marker_save',popped_logs='redirected_to_dev_null')
    elif budget>2*1024*1024 and cores>2:
        raise ValueError('full marker traces require at most two cores; high-core runs retain the 2MiB cap')
    return result


def retain_instruction(name,cores,completion_only):
    match=INSTRUCTION_PATTERN.match(name)
    return bool(match and int(match[1])<cores and (not completion_only or
        (match[3]=='instr_log' and match[2] in ('cubecore0','veccore0','veccore1'))))


def collect_completion_markers(output,log,sample_dir,budget,limited,expected,interval_type):
    """Stream one completed regular file at a time; never retain raw+gzip sets."""
    sample_dir.mkdir();marker_files=[];samples=[];failures=[]
    for name,counts in expected.items():
        path=log/name;errors=[];events=[];intervals=interval_type()
        digest=hashlib.sha256();captured=0;head=b'';tail=b''
        size=0;identity=None;complete=False;raw_removed=False;stable=False;parse_enabled=True
        event_limit=2*sum(counts.values())
        compressed=sample_dir/(name+'.gz')
        try:
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as source:
                info=os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode):raise ValueError('completion trace is not a regular file')
                identity=(info.st_dev,info.st_ino);size=info.st_size
                with gzip.open(compressed,'wb',compresslevel=1) as saved:
                    while captured<budget:
                        line=source.readline(min(65537,budget-captured))
                        if not line:break
                        captured+=len(line);digest.update(line);saved.write(line)
                        head=(head+line)[:8192];tail=(tail+line)[-8192:]
                        if len(line)>65536 or not line.endswith(b'\n'):
                            errors.append('partial or oversized instruction line');continue
                        if parse_enabled:
                            try:
                                for event in marker_events([line.decode(errors='replace')]):
                                    if len(events)>=event_limit:raise ValueError('marker count exceeds real task/profile bound')
                                    intervals.accept(event);events.append(event)
                            except ValueError as error:
                                errors.append(str(error));parse_enabled=False
                after=os.fstat(source.fileno())
                if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)!=(
                        info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns):
                    raise ValueError('completion trace changed after device finalization')
                stable=True
            if name in limited or size>=budget or captured!=size:
                errors.append('completion trace cap reached or prefix incomplete')
            intervals.finish()
            if {key:row['count'] for key,row in intervals.groups.items()}!=dict(counts):
                errors.append('phase event counts differ from real tasks/profile counters')
            complete=not errors
        except (OSError,ValueError) as error:
            errors.append(str(error))
        marker_path=sample_dir/(name+'.markers.json')
        save_json(marker_path,{'source':name,'source_sha256':digest.hexdigest(),
            'complete_for_actor':complete,'captured_bytes':captured,'source_bytes':size,
            'events':events,'errors':errors,'method':'bounded_completion_only_regular_file'})
        preview=sample_dir/(name+'.txt')
        preview.write_bytes(head+b'\n[bounded head/tail]\n'+tail)
        # Preserve original raw on any unstable inode/read/gzip failure.
        # Cap/nesting failure still has a bounded saved gzip+marker witness.
        if stable and identity is not None and compressed.is_file() and captured==min(size,budget):
            try:
                current=path.lstat()
                if (not stat.S_ISREG(current.st_mode) or (current.st_dev,current.st_ino)!=identity or
                        current.st_size!=size or current.st_mtime_ns!=info.st_mtime_ns):
                    raise ValueError('completion trace identity changed before owned raw cleanup')
                path.unlink();raw_removed=True
            except (OSError,ValueError) as error:
                errors.append(str(error));complete=False
                evidence=json.loads(marker_path.read_text())
                evidence.update(complete_for_actor=False,errors=errors);save_json(marker_path,evidence)
        marker_files.append({'path':str(marker_path.relative_to(output)),'event_count':len(events),
                             'complete_for_actor':complete,'source':name})
        samples.append({'source':name,'source_bytes':size,'captured_bytes':captured,
                        'prefix_capped':size>=budget or name in limited,'complete_for_actor':complete,
                        'raw_removed':raw_removed,'sample':str(preview.relative_to(output))})
        failures.extend(name+': '+error for error in errors)
    save_json(output/'instruction_samples.json',{'method':'research_completion_only; exact closure and task/profile counts',
        'per_file_budget_bytes':budget,'files':samples,'failures':failures})
    return samples,marker_files,failures


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def floats(data):
    result=array('f');result.frombytes(data);return result


def merged_oracle(partial_bytes,lse_bytes,expected_bytes,expected_lse_bytes,n,hq,d,segments):
    partial,lse=floats(partial_bytes),floats(lse_bytes)
    expected,expected_lse=floats(expected_bytes),floats(expected_lse_bytes)
    if len(expected)!=n*hq*d or len(expected_lse)!=n*hq:
        raise ValueError('independent merged oracle size mismatch')
    max_output=max_lse=0.;bad_output=bad_lse=0
    for row in range(n*hq):
        logs=lse[row*segments:(row+1)*segments]
        if any(math.isnan(x) or x==math.inf for x in logs):
            bad_lse+=1;bad_output+=d;continue
        maximum=max(logs)
        if maximum==-math.inf:
            values=[0.]*d;merged_lse=-math.inf
        else:
            weights=[math.exp(x-maximum) if x!=-math.inf else 0. for x in logs]
            denominator=math.fsum(weights)
            merged_lse=maximum+math.log(denominator)
            values=[math.fsum(weights[s]*partial[(row*segments+s)*d+c]
                for s in range(segments) if weights[s])/denominator for c in range(d)]
        want_lse=expected_lse[row]
        if math.isinf(want_lse):
            bad_lse+=merged_lse!=want_lse
        else:
            error=abs(merged_lse-want_lse)
            bad_lse+=not math.isfinite(merged_lse) or error>.005+.005*abs(want_lse)
            if math.isfinite(error):max_lse=max(max_lse,error)
        for col,value in enumerate(values):
            want=expected[row*d+col];error=abs(value-want)
            bad_output+=not math.isfinite(value) or error>.005+.005*abs(want)
            if math.isfinite(error):max_output=max(max_output,error)
    return {'bad_output':bad_output,'bad_lse':bad_lse,'max_output_abs':max_output,
            'max_lse_abs':max_lse,'atol':.005,'rtol':.005,
            'method':'offline stable LSE merge of actual CAModel partials versus independent PR golden'}


def entries(node):
    if isinstance(node,dict):
        if {'start_cycle','end_cycle','duration'}<=node.keys():yield node
        else:
            for child in node.values():yield from entries(child)


PHASE_FIELDS=('tasks','kv_units','kv_rows_observed','task_span','query_stage',
    'packed_dma','unpack','precise_load','kv_publish','aiv_qk_wait','softmax_p',
    'aiv_pv_wait','pv_acc','aic_kv_wait','aic_qk','aic_p_wait','aic_pv','aic_acc_wait',
    'aic_rotation','aiv_finalize','task_metadata','final_writeback','rotation_wait','compute_span')


def marker_events(lines):
    """Explicit diagnostic MOVs; never infer whole phases from sampled ops."""
    pattern=re.compile(r'\[info\]\s+\[(\d+)\].*?MOV_XD_XN\s+.*?XD:X\d+=(0x[0-9a-fA-F]+)')
    events=[]
    for line in lines:
        match=pattern.search(line)
        if not match:continue
        tag=int(match[2],16)
        if tag>>32!=0x5a77c01d:continue
        if tag&0xfffffc00:raise ValueError('unknown phase marker bits')
        field=(tag&255)>>1;source=(tag>>8)&3
        if field not in range(3,24):raise ValueError('unknown phase marker field')
        events.append({'cycle':int(match[1]),'source':source,'field':field,'end':bool(tag&1)})
    return events


def phase_profile(data,cores):
    """Schema2 agreed with the isolated profiler owner; no wall-time summation."""
    if len(data)!=cores*2304:raise ValueError('profile [cores,3,4,24] int64 byte count differs')
    values=struct.unpack('<'+'q'*(len(data)//8),data)
    if any(x<0 for x in values):raise ValueError('profile contains unwritten poison or negative counter')
    actors=[]
    for core in range(cores):
        for engine,name in enumerate(('AIC','AIV0','AIV1')):
            sources={}
            for source,label in enumerate(('history','window','current','total')):
                start=((core*3+engine)*4+source)*24
                sources[label]={field:values[start+i] for i,field in enumerate(PHASE_FIELDS) if values[start+i]}
            actors.append({'cube_core':core,'engine':name,'sources':sources})
    timed_actors=[a['sources']['total'].get('compute_span',0) for a in actors]
    clock_available=all(value>0 for value in timed_actors)
    return {'schema':2,'dtype':'int64','shape':[cores,3,4,24],'bytes_per_core':2304,
        'fields':PHASE_FIELDS,'omitted_fields_equal_zero':True,'counter':'GetSystemCycle',
        'timing_status':'observed' if clock_available else 'unavailable_zero_SYS_CNT',
        'timing_usable':clock_available,
        'actors':actors,'interpretation':'per-actor recorded spans; do not sum actors or overlapping fields as wall',
        'existing_wait_fields':[9,11,13,15,17,22],
        'stage_semantics':'other stage spans include Scalar issue and existing internal synchronization; no added completion fence',
        'compute_span_semantics':'field23 actor Init+Process excludes profile flush'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('baseline','candidate'),required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--fixture-root',type=Path,default=Path('artifacts/window_batch_decode'))
    parser.add_argument('--export-root',type=Path,default=Path('artifacts/window_batch_decode_cpu_export'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--candidate-q4',default='oscar_attention_cv_window_batch_decode_kernel')
    parser.add_argument('--candidate-q1',default='oscar_attention_cv_window_batch_q1_kernel')
    parser.add_argument('--candidate-workspace-bytes-per-core',type=int,default=917504)
    parser.add_argument('--baseline-workspace-bytes-per-core',type=int,default=917504)
    parser.add_argument('--instruction-budget-bytes',type=int,default=0)
    parser.add_argument('--profile-counters',action='store_true')
    parser.add_argument('--require-phase-markers',action='store_true')
    parser.add_argument('--research-completion-only',action='store_true',
                        help='20-core profile research only: retain capped completion logs, discard popped logs, require complete markers')
    parser.add_argument('--baseline-q4',default='oscar_attention_cv_striped_decode_simd_kernel')
    parser.add_argument('--baseline-q1',default='oscar_attention_cv_striped_q1_simd_kernel')
    parser.add_argument('--cpu-reference',choices=('required','optional'),default='required')
    parser.add_argument('--simulator-config-dir',type=Path)
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    config_dir=(args.simulator_config_dir.resolve() if args.simulator_config_dir else
                Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib'))
    simulator_config=json.loads((config_dir/'config_stars.json').read_text())
    simulator_detail=json.loads((config_dir/'config.json').read_text())
    reference_detail=json.loads(Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/config.json').read_text())
    if simulator_detail.get('L2CACHE')!=reference_detail.get('L2CACHE'):
        raise ValueError('simulator memory/cache model changed; not a logging/host-thread-only trial')
    model_top=simulator_config.get('model_top',{})
    if model_top!={'sim_type':0,'num_aic':24,'num_aiv':48}:
        raise ValueError('simulator hardware/model changed; comparison requires the same cycle-model topology')
    if args.simulator_config_dir and os.environ.get('CAMODEL_CONFIG_PATH')!=str(config_dir):
        raise ValueError('explicit simulator config was not passed before runtime library initialization')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'owner.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'output':str(output),
                                'script':str(Path(__file__).resolve())})
    fixture_root=args.fixture_root if args.fixture_root.is_absolute() else root/args.fixture_root
    export_root=args.export_root if args.export_root.is_absolute() else root/args.export_root
    fixture=fixture_root/args.case
    exported=export_root/args.case
    manifest=json.loads((fixture.parent/'cases.json').read_text())
    cases=[c for c in manifest['cases'] if c['name']==args.case]
    if len(cases)!=1 or cases[0]['expected_error']:
        raise ValueError('performance arm requires one valid registered full-source fixture')
    mode=cases[0]['mode']
    shape=list(map(int,(fixture/'shape.txt').read_text().split()))
    if len(shape)!=16:raise ValueError('fixture shape must have 16 fields')
    n,hq,hk,d,context,sink,recent,spec,block,blocks,prefix,stride,cores,requests,splits,columns=shape
    attrs_text=(exported/'attrs.txt').read_text().split()
    attrs=list(map(int,attrs_text[:17]));scale=float(attrs_text[17]);export_cores=int(attrs_text[18])
    expected_attrs=[n,hq,hk,d,requests,columns,n*hk*3*splits,block,blocks,prefix,stride,
                    (sink+recent+spec)*hk*d,sink+recent+spec,sink,recent,spec,splits]
    if len(attrs_text)!=19 or attrs!=expected_attrs or export_cores!=cores or cores!=cases[0]['cores']:
        raise ValueError('CPU-export attrs differ from complete original fixture/core budget')
    if d!=256 or scale!=.0625 or not 1<=cores<=32:
        raise ValueError('unsupported D256/core/scale')
    if tuple(map(float,(fixture/'tolerance.txt').read_text().split()))!=(.005,.005):
        raise ValueError('frozen fused oracle tolerance changed')
    inputs={name:(fixture/path).read_bytes() for name,path in {
        'q':'q.bin','qr':'qr.bin','ck':'ck.bin','cv':'cv.bin','rv':'rv.bin','raw':'raw_striped.bin',
        'table':'table.bin','wk':'wk.bin','wv':'wv.bin','tags':'tags.bin'}.items()}
    inputs['tasks']=(exported/'tasks.bin').read_bytes()
    expected_sizes={'q':n*hq*d*2,'qr':n*hq*d*4,'ck':n*hk*d*2,'cv':n*hk*d*2,'rv':d*d*4,
        'raw':prefix+blocks*stride,'table':requests*columns*4,'wk':blocks*(sink+recent+spec)*hk*d*2,
        'wv':blocks*(sink+recent+spec)*hk*d*2,'tags':blocks*(sink+recent+spec)*8,
        'tasks':attrs[6]*16*8}
    if any(len(inputs[k])!=size for k,size in expected_sizes.items()):raise ValueError('input size mismatch')
    workspace_per_core=(args.baseline_workspace_bytes_per_core if args.variant=='baseline'
                        else args.candidate_workspace_bytes_per_core)
    if workspace_per_core<917504 or workspace_per_core%32:
        raise ValueError('invalid explicit candidate workspace capacity')
    output_sizes={'partial':n*hq*3*splits*d*4,'lse':n*hq*3*splits*4,
                  'status':attrs[6]*2*4,'workspace':cores*workspace_per_core}
    if args.profile_counters:output_sizes['profile']=cores*2304
    oracle={name:(fixture/path).read_bytes() for name,path in {
        'output':'expected_output.bin','lse':'expected_lse.bin'}.items()}
    cpu_paths={name:exported/(name+'.bin') for name in ('partial','lse','status')}
    available=[path.is_file() for path in cpu_paths.values()]
    if not all(available) and (args.cpu_reference=='required' or any(available)):
        raise ValueError('CPU reference is missing or partial; optional admits only an explicitly absent complete reference')
    cpu_outputs={name:path.read_bytes() for name,path in cpu_paths.items()} if all(available) else {}
    if any(len(data)!=output_sizes[name] for name,data in cpu_outputs.items()):raise ValueError('CPU reference size mismatch')
    log=output/'log_ca';log.mkdir()
    os.environ['CAMODEL_LOG_PATH']=str(log);os.environ['STARS_LOG_PATH']=str(log)
    stop=threading.Event();redirected=set();limited_instructions=set()
    trace_policy_report=trace_policy(args.instruction_budget_bytes,cores,args.profile_counters,
                                     args.research_completion_only,output)
    instruction_pattern=INSTRUCTION_PATTERN
    def quiet_dumps():
        sink_fd=os.open('/dev/null',os.O_WRONLY)
        try:
            while not stop.is_set():
                for item in os.listdir('/proc/self/fd'):
                    if not item.isdigit() or int(item)==sink_fd:continue
                    try:
                        target=os.readlink('/proc/self/fd/'+item)
                        if target.startswith(str(log)+'/') and target.endswith('.dump'):
                            instruction=instruction_pattern.match(Path(target).name)
                            if args.instruction_budget_bytes and retain_instruction(
                                    Path(target).name,cores,args.research_completion_only):
                                if os.fstat(int(item)).st_size<args.instruction_budget_bytes:continue
                                limited_instructions.add(Path(target).name)
                            os.dup2(sink_fd,int(item));redirected.add(target)
                    except OSError:pass
                time.sleep(.001)
        finally:os.close(sink_fd)
    thread=threading.Thread(target=quiet_dumps,daemon=True);thread.start()
    library=args.library.resolve() if args.library else root/'build/device/lib/liboscar_window_batch_decode.so'
    name=((args.baseline_q4 if mode=='q4' else args.baseline_q1) if args.variant=='baseline'
          else (args.candidate_q4 if mode=='q4' else args.candidate_q1))
    unprofiled_names={'oscar_attention_cv_striped_decode_simd_kernel','oscar_attention_cv_striped_q1_simd_kernel',
                      'oscar_attention_cv_window_batch_decode_kernel','oscar_attention_cv_window_batch_q1_kernel'}
    if args.profile_counters and name in unprofiled_names:
        raise ValueError('extra profile GM requires an explicit instrumented kernel symbol')
    report={'status':'initialized','variant':args.variant,'case':args.case,'mode':mode,'cores':cores,
        'attrs':attrs,'scale':scale,'all_sources_and_tasks':True,'task_filtering':False,
        'library_sha256':sha(library.read_bytes()),'kernel':name,
        'workspace_bytes_per_core':workspace_per_core,
        'profile_counters':args.profile_counters,'phase_profile_schema':2 if args.profile_counters else None,
        'trace_policy':trace_policy_report,
        'cpu_reference':'available_checked' if cpu_outputs else 'not_available',
        'cpu_reference_policy':args.cpu_reference,
        'simulator_configuration':{'directory':str(config_dir),'config':simulator_config,
            'sha256':sha((config_dir/'config_stars.json').read_bytes()),
            'detail':simulator_detail,'detail_sha256':sha((config_dir/'config.json').read_bytes())},
        'input_sha256':{k:sha(v) for k,v in inputs.items()},
        'oracle_sha256':{k:sha(v) for k,v in oracle.items()},
        'cpu_baseline_sha256':{k:sha(v) for k,v in cpu_outputs.items()},
        'backend':'local_CANN9.1_dav2201_CAModel','profile_scope':'complete operator; actor durations include waits',
        'npu':'not_run','graph':'not_run','model_quality':'not_run','whole_service':'not_run'}
    save_json(output/'inputs.json',report)
    acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
    def api(name,argtypes):
        fn=getattr(acl,name);fn.argtypes=argtypes;fn.restype=C.c_int;return fn
    init=api('aclInit',[C.c_char_p]);setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int]);final=api('aclFinalize',[])
    alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
    copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int])
    create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)]);destroy=api('aclrtDestroyStream',[C.c_void_p]);sync=api('aclrtSynchronizeStream',[C.c_void_p])
    def check(label,rc):
        if rc:raise RuntimeError(label+' rc='+str(rc))
    check('init',init(None));check('set_device',setdev(0))
    lib=C.CDLL(str(library),mode=C.RTLD_GLOBAL);launch=getattr(lib,'aclrtlaunch_'+name)
    launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*(16 if args.profile_counters else 15)+[C.c_int64]*17+[C.c_float]
    launch.restype=C.c_uint32
    ptrs={};owners={};hosts=[];guard=b'\xa5'*64
    initial={**inputs,**{k:b'\x85'*v for k,v in output_sizes.items()}}
    for key,data in initial.items():
        allocation=C.c_void_p();check('alloc '+key,alloc(C.byref(allocation),len(data)+128,0));owners[key]=allocation
        ptrs[key]=C.c_void_p(allocation.value+64)
        payload=guard+data+guard;host=C.create_string_buffer(payload,len(payload));hosts.append(host)
        check('h2d '+key,copy(allocation,len(payload),host,len(payload),1))
    stream=C.c_void_p();check('stream',create(C.byref(stream)))
    print('WINDOW_CAMODEL_START',args.variant,args.case,'cores',cores,flush=True)
    order=('q','qr','ck','cv','rv','raw','table','wk','wv','tags','tasks','partial','lse','status','workspace')
    gm_args=[ptrs[k] for k in order]
    if args.profile_counters:gm_args.append(ptrs['profile'])
    check('launch',launch(cores,stream,*gm_args,*attrs,scale))
    check('sync',sync(stream));print('WINDOW_CAMODEL_COMPLETE',args.variant,args.case,flush=True)
    outputs={};saved={};failures=[]
    # Save all four outputs before any oracle read/compare or input download.
    for key,size in output_sizes.items():
        host=C.create_string_buffer(size+128);check('d2h '+key,copy(host,size+128,owners[key],size+128,2))
        data=host.raw
        if data[:64]!=guard or data[-64:]!=guard:failures.append(key+'_guard')
        outputs[key]=data[64:-64];path=output/(key+'.bin');path.write_bytes(outputs[key])
        saved[key]={'bytes':size,'sha256':sha(outputs[key]),'path':path.name}
    save_json(output/'completion.json',{**report,'status':'device_complete_outputs_saved','outputs':saved,'failures':failures})
    print('WINDOW_CAMODEL_SAVED',args.variant,args.case,flush=True)
    for key,data in inputs.items():
        size=len(data);host=C.create_string_buffer(size+128)
        check('d2h input '+key,copy(host,size+128,owners[key],size+128,2))
        if host.raw!=guard+data+guard:failures.append(key+'_input_or_guard_changed')
    for key,want in cpu_outputs.items():
        if outputs[key]!=want:failures.append(key+'_cpu_baseline_bitwise')
    if any(struct.unpack('<'+'i'*(len(outputs['status'])//4),outputs['status'])):failures.append('nonzero_status')
    numerical=merged_oracle(outputs['partial'],outputs['lse'],oracle['output'],oracle['lse'],n,hq,d,3*splits)
    if numerical['bad_output'] or numerical['bad_lse']:failures.append('independent_oracle')
    if args.profile_counters:
        try:
            decoded=phase_profile(outputs['profile'],cores)
            save_json(output/'phase_profile.json',decoded)
            report['phase_profile_path']='phase_profile.json'
        except ValueError as error:
            failures.append('phase_profile');report['phase_profile_error']=str(error)
    report.update(outputs=saved,oracle=numerical,failures=failures,
                  status='numeric_passed_pending_profile' if not failures else 'failed')
    save_json(output/'report.json',report)
    for allocation in owners.values():check('free',free(allocation))
    check('destroy',destroy(stream));check('reset',reset(0));check('final',final());stop.set();thread.join(timeout=2)
    samples=[];marker_files=[]
    if args.research_completion_only:
        # Only pure marker/count validators are reused; no FIFO is constructed.
        if str(root) not in sys.path:sys.path.insert(0,str(root))
        from tools.experiments.stream_camodel_markers import Intervals,expected_counts
        try:
            expected=expected_counts(report,inputs['tasks'],outputs['profile'])
            samples,marker_files,marker_failures=collect_completion_markers(
                output,log,output/'instruction_samples',args.instruction_budget_bytes,
                limited_instructions,expected,Intervals)
            if marker_failures:failures.append('completion_marker_counts_or_trace_incomplete')
        except ValueError as error:
            failures.append('completion_marker_collection_failed')
            report['completion_marker_error']=str(error)
    elif args.instruction_budget_bytes:
        sample_dir=output/'instruction_samples';sample_dir.mkdir()
        for path in sorted(log.glob('*.dump')):
            match=instruction_pattern.match(path.name)
            if not match or int(match[1])>=cores:continue
            size=path.stat().st_size
            if not size:continue
            with path.open('rb') as source:raw=source.read(args.instruction_budget_bytes+65536)
            lines=raw.decode(errors='replace').splitlines()
            interesting=[line for line in lines if re.search(r'mmad|wait_flag|set_flag|mte[123]|PIPE_',line,re.I)]
            text=raw[:8192].decode(errors='replace')+'\n[bounded head / selected lines / tail]\n'
            text+='\n'.join(interesting[:96])+'\n'+raw[-8192:].decode(errors='replace')
            destination=sample_dir/(path.name+'.txt');destination.write_text(text)
            if match[3]=='instr_log':
                markers=marker_events(lines)
                if markers:
                    marker_path=sample_dir/(path.name+'.markers.json')
                    complete=path.name not in limited_instructions and size<=len(raw)
                    evidence={'source':path.name,'source_sha256':sha(raw),'complete_for_actor':complete,
                              'captured_bytes':len(raw),'events':markers}
                    save_json(marker_path,evidence)
                    (sample_dir/(path.name+'.gz')).write_bytes(gzip.compress(raw,compresslevel=1))
                    marker_files.append({'path':str(marker_path.relative_to(output)),
                        'event_count':len(markers),'complete_for_actor':complete})
            samples.append({'source':path.name,'source_bytes':size,'captured_bytes':len(raw),
                'prefix_capped':path.name in limited_instructions,'complete_for_actor':path.name not in limited_instructions and size<=len(raw),
                'sample':str(destination.relative_to(output))})
        save_json(output/'instruction_samples.json',{'method':'bounded instruction prefixes; samples do not establish whole-kernel pipe percentages',
            'per_file_budget_bytes':args.instruction_budget_bytes,'files':samples})
    if (args.require_phase_markers or args.research_completion_only) and (len(marker_files)!=cores*3 or
            not all(row['complete_for_actor'] for row in marker_files)):
        failures.append('phase_marker_trace_missing_or_incomplete')
    profiles={}
    for kind,count in (('task',1),('aic',cores),('aiv',2*cores)):
        records=list(entries(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
        if len(records)!=count or any(r['duration']<=0 or r['duration']!=r['end_cycle']-r['start_cycle'] for r in records):
            failures.append(kind+'_profile_incomplete')
        profiles[kind]=records
    if len(profiles['task'])==1:
        task=profiles['task'][0]
        for kind in ('aic','aiv'):
            if any((r['stream_id'],r['task_id'])!=(task['stream_id'],task['task_id']) for r in profiles[kind]):
                failures.append(kind+'_task_identity')
        if {r['core_id'] for r in profiles['aic']}!=set(range(cores)):failures.append('aic_owners')
        if {r['core_id'] for r in profiles['aiv']}!=set(range(2*cores)):failures.append('aiv_owners')
    report.update(status='passed' if not failures else 'failed',failures=failures,profiles=profiles,
                  task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None,
                  dump_files_redirected=len(redirected),instruction_samples=samples,
                  phase_marker_files=marker_files)
    save_json(output/'report.json',report);print('WINDOW_CAMODEL_RESULT',json.dumps(report),flush=True)
    return 2 if failures else 0


if __name__=='__main__':raise SystemExit(main())
