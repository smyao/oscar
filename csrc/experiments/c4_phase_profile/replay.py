"""Archive #94/#95/#126/#132-135/#145/#148/#154/#155: one C4 CAModel arm.

Run under run_camodel_arm.py with its fixed 900s computation deadline. This
child alone loads SDK/preload. No retries, FIFO, source/task filtering, global
decode tool changes, NPU claim or end-to-end estimate.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time
import tomllib
import traceback

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_phase_profile import fixture as case,markers,generate
from csrc.experiments.window_batch_decode.replay_camodel import (
    save_json,sha,entries,merged_oracle,trace_policy,retain_instruction)


def read_inputs(directory,prepared):
    meta=json.loads((directory/'fixture.json').read_text())
    n=case.declared_tokens(meta['tokens'])
    if (meta['case']!=case.case_name(n) or meta['oracle_scope']!=case.plan(n)['oracle_scope'] or
            meta['acceptance_sha256']!=sha((ROOT/'configs/acceptance.json').read_bytes())):
        raise ValueError('wrong C4 fixture/oracle/frozen acceptance')
    attrs=list(map(int,(directory/'shape.txt').read_text().split()))
    if len(attrs)!=17 or attrs!=meta['attrs'] or (attrs[0:5],attrs[6],attrs[13:])!=([n,6,1,256,1],3*n,[64,256,3,1]):
        raise ValueError('not declared Q336/Q1344 R1/C769/S1 attributes')
    names=('q','qr','ck','cv','rv','raw','table','wk','wv','tags')
    inputs={name:(directory/(name+'.bin')).read_bytes() for name in names}
    for name,data in inputs.items():
        if sha(data)!=meta['input_sha256'][name]:raise ValueError('fixture input changed: '+name)
    oracle={name:(directory/('expected_'+name+'.bin')).read_bytes() for name in ('output','lse')}
    for name,data in oracle.items():
        if sha(data)!=meta['input_sha256']['expected_'+name]:raise ValueError('oracle changed: '+name)
    inputs['tasks']=(prepared/'tasks.bin').read_bytes()
    ledger=json.loads((prepared/'task_ledger.json').read_text())
    if (ledger['after_sha256']!=sha(inputs['tasks']) or
            ledger['before_sha256']!=sha((prepared/'tasks_before_suppression.bin').read_bytes()) or
            ledger['fixture_sha256']!=sha((directory/'fixture.json').read_bytes()) or ledger['attrs']!=attrs):
        raise ValueError('real preparation ledger identity failed')
    before=list(struct.iter_unpack('<16q',(prepared/'tasks_before_suppression.bin').read_bytes()))
    expected=[]
    for row in before:
        changed=list(row)
        if row[7]==2 and row[10]==0 and row[4]>=row[3]:changed[4]=row[3]
        expected.append(tuple(changed))
    actual=list(struct.iter_unpack('<16q',inputs['tasks']))
    if actual!=expected:raise ValueError('changed more than healthy source2 end')
    recomputed=case.task_ledger(actual)
    if any(ledger[key]!=json.loads(json.dumps(recomputed[key])) for key in recomputed):
        raise ValueError('C4 work ledger changed')
    n,hq,hk,d,r,cols,tasks,b,blocks,prefix,stride,ws,ts,*_=attrs
    sizes={'q':n*hq*d*2,'qr':n*hq*d*4,'ck':n*hk*d*2,'cv':n*hk*d*2,'rv':d*d*4,
        'raw':prefix+blocks*stride,'table':r*cols*4,'wk':blocks*ws*2,'wv':blocks*ws*2,
        'tags':blocks*ts*8,'tasks':tasks*128}
    if any(len(inputs[k])!=v for k,v in sizes.items()):raise ValueError('input byte size differs')
    return meta,attrs,inputs,oracle,ledger


def run(args):
    normal_source=generate.generate(False)[0];profile_source=generate.generate(True)[0]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    save_json(output/'owner.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'script':str(Path(__file__).resolve()),'output':str(output)})
    meta,attrs,inputs,oracle,ledger=read_inputs(args.fixture.resolve(),args.prepared.resolve())
    n=attrs[0]
    config=args.simulator_config.resolve()
    if os.environ.get('CAMODEL_CONFIG_PATH')!=str(config):raise ValueError('SDK config must precede process startup')
    stars=json.loads((config/'config_stars.json').read_text());detail=json.loads((config/'config.json').read_text())
    stock=json.loads(Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/config.json').read_text())
    if stars.get('model_top')!={'sim_type':0,'num_aic':24,'num_aiv':48} or detail.get('L2CACHE')!=stock.get('L2CACHE'):
        raise ValueError('simulator topology/cache changed')
    profile=args.arm=='profile';library=args.library.resolve()
    report={'status':'initialized','experiment':'c4_phase_profile','profile_schema':3 if profile else None,
        'arm':args.arm,'case':meta['case'],'kernel':'oscar_c4_'+args.arm+'_kernel','cores':20,
        'attrs':attrs,'scale':.0625,'all_prepared_tasks_retained':True,'task_filtering':False,
        'source2_suppressed':True,'oracle_scope':meta['oracle_scope'],
        'source_sha256':generate.PINS[generate.SOURCE],
        'instrumentation_revision':'schema3_finite_reduce_v1',
        'generated_source_sha256':sha((profile_source if profile else normal_source).encode()),
        'dependency_sha256':generate.PINS,'library_sha256':sha(library.read_bytes()),
        'input_sha256':{k:sha(v) for k,v in inputs.items()},'oracle_sha256':{k:sha(v) for k,v in oracle.items()},
        'acceptance_sha256':meta['acceptance_sha256'],'task_ledger':ledger,
        'simulator_configuration':{'stars_sha256':sha((config/'config_stars.json').read_bytes()),
            'detail_sha256':sha((config/'config.json').read_bytes()),'stars':stars,'detail':detail},
        'workspace_bytes_per_core':1839104,'npu':'not_run','graph':'not_run','whole_service':'not_run'}
    log=output/'log_ca';log.mkdir();os.environ['CAMODEL_LOG_PATH']=str(log);os.environ['STARS_LOG_PATH']=str(log)
    if args.completion_markers and not profile:raise ValueError('completion markers require profile arm')
    budget=64*1024*1024 if args.completion_markers else 0
    report['trace_policy']=trace_policy(budget,20,profile,args.completion_markers,output)
    save_json(output/'inputs.json',report)
    (output/'task_witness.bin').write_bytes(inputs['tasks'])
    stopped=threading.Event();limited=set();redirected=set()
    def quiet():
        fd=os.open('/dev/null',os.O_WRONLY)
        try:
            while not stopped.is_set():
                for name in os.listdir('/proc/self/fd'):
                    if not name.isdecimal() or int(name)==fd:continue
                    try:
                        target=os.readlink('/proc/self/fd/'+name)
                        if not target.startswith(str(log)+'/') or not target.endswith('.dump'):continue
                        if budget and retain_instruction(Path(target).name,20,True):
                            if os.fstat(int(name)).st_size<budget:continue
                            limited.add(Path(target).name)
                        os.dup2(fd,int(name));redirected.add(target)
                    except OSError:continue
                time.sleep(.001)
        finally:os.close(fd)
    thread=threading.Thread(target=quiet,daemon=True);thread.start()
    owners={};acl=None;stream=None;initialized=False;device=False;failures=[]
    def check(label,rc):
        if rc:raise RuntimeError(label+' rc='+str(rc))
    try:
        acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
        def api(name,types):
            function=getattr(acl,name);function.argtypes=types;function.restype=C.c_int;return function
        init=api('aclInit',[C.c_char_p]);final=api('aclFinalize',[])
        setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int])
        alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
        copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int])
        create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)]);destroy=api('aclrtDestroyStream',[C.c_void_p])
        sync=api('aclrtSynchronizeStream',[C.c_void_p])
        check('init',init(None));initialized=True;check('device',setdev(0));device=True
        lib=C.CDLL(str(library),mode=C.RTLD_GLOBAL);launch=getattr(lib,'aclrtlaunch_'+report['kernel'])
        launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*(17 if profile else 16)+[C.c_int64]*17+[C.c_float]
        launch.restype=C.c_uint32
        sizes={'partial':n*6*3*256*4,'lse':n*6*3*4,'status':n*3*2*4,
               'workspace':20*1839104,'cluster_stats':20*64}
        if profile:sizes['profile']=20*3072
        ptrs={};hosts=[];guard=b'\xa5'*64
        for key,data in {**inputs,**{k:b'\x85'*size for k,size in sizes.items()}}.items():
            allocation=C.c_void_p();check('alloc '+key,alloc(C.byref(allocation),len(data)+128,0));owners[key]=allocation
            ptrs[key]=C.c_void_p(allocation.value+64);payload=guard+data+guard
            host=C.create_string_buffer(payload,len(payload));hosts.append(host)
            check('upload '+key,copy(allocation,len(payload),host,len(payload),1))
        stream=C.c_void_p();check('stream',create(C.byref(stream)))
        order=('q','qr','ck','cv','rv','raw','table','wk','wv','tags','tasks','partial','lse','status','workspace','cluster_stats')
        pointers=[ptrs[name] for name in order]+([ptrs['profile']] if profile else [])
        print('C4_CAMODEL_START',args.arm,f'Q{n} R1 C769 S1 cores20',flush=True)
        check('launch',launch(20,stream,*pointers,*attrs,.0625));check('sync',sync(stream))
        print('C4_CAMODEL_COMPLETE',args.arm,flush=True)
        outputs={};saved={}
        for name,size in sizes.items():
            host=C.create_string_buffer(size+128);check('download '+name,copy(host,size+128,owners[name],size+128,2))
            data=host.raw
            if data[:64]!=guard or data[-64:]!=guard:failures.append(name+'_guard')
            outputs[name]=data[64:-64];(output/(name+'.bin')).write_bytes(outputs[name])
            saved[name]={'bytes':size,'sha256':sha(outputs[name]),'path':name+'.bin'}
        save_json(output/'completion.json',{**report,'status':'device_complete_outputs_saved','outputs':saved,'failures':failures})
        for name,data in inputs.items():
            host=C.create_string_buffer(len(data)+128);check('input check '+name,copy(host,len(host),owners[name],len(host),2))
            if host.raw!=guard+data+guard:failures.append(name+'_input_or_guard_changed')
        if any(struct.unpack('<'+str(n*6)+'i',outputs['status'])):failures.append('nonzero_status')
        numerical=merged_oracle(outputs['partial'],outputs['lse'],oracle['output'],oracle['lse'],n,6,256,3)
        if numerical['bad_output'] or numerical['bad_lse']:failures.append('independent_oracle')
        counters=[list(row) for row in struct.iter_unpack('<8q',outputs['cluster_stats'])]
        if counters!=markers.expected_cluster_stats(inputs['tasks']):failures.append('cluster_counter_contract')
        report.update(outputs=saved,oracle=numerical,cluster_stats=counters,failures=failures)
        if profile:
            save_json(output/'phase_profile.json',markers.decoded_profile(outputs['profile']))
            expected=markers.expected_counts(report,inputs['tasks'],outputs['profile'])
        save_json(output/'report.json',{**report,'status':'numeric_passed_pending_profile' if not failures else 'failed'})
        for allocation in owners.values():check('free',free(allocation))
        owners.clear();check('destroy',destroy(stream));stream=None
        check('reset',reset(0));device=False;check('final',final());initialized=False
        stopped.set();thread.join(timeout=2)
        marker_files=[]
        if args.completion_markers:
            _samples,marker_files,errors=markers.collect(output,log,output/'instruction_samples',budget,limited,expected)
            failures.extend(errors)
            if len(marker_files)!=60 or not all(row['complete_for_actor'] for row in marker_files):failures.append('incomplete_C4_markers')
        profiles={}
        for kind,count in (('task',1),('aic',20),('aiv',40)):
            rows=list(entries(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
            if len(rows)!=count or any(row['duration']<=0 or row['duration']!=row['end_cycle']-row['start_cycle'] for row in rows):
                failures.append(kind+'_profile_incomplete')
            profiles[kind]=rows
        if len(profiles['task'])==1:
            task=profiles['task'][0]
            for kind,count in (('aic',20),('aiv',40)):
                if ({row['core_id'] for row in profiles[kind]}!=set(range(count)) or
                        any((row['task_id'],row['stream_id'])!=(task['task_id'],task['stream_id']) for row in profiles[kind])):
                    failures.append(kind+'_identity')
        report.update(status='failed' if failures else 'passed',failures=failures,profiles=profiles,
            task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None,
            marker_collection_requested=args.completion_markers,phase_marker_files=marker_files,
            dump_files_redirected=len(redirected))
        save_json(output/'report.json',report)
        print('C4_CAMODEL_RESULT',json.dumps({'status':report['status'],'task_ticks':report['task_ticks'],
            'oracle':numerical,'failures':failures,'report':str(output/'report.json')}),flush=True)
        return 2 if failures else 0
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save_json(output/'report.json',report);raise
    finally:
        stopped.set();thread.join(timeout=2)
        # On device execution errors no further potentially blocking SDK calls
        # are attempted. The bounded supervisor owns termination of this child.


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('normal','profile'),required=True)
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--simulator-config',type=Path,required=True)
    parser.add_argument('--completion-markers',action='store_true')
    raise SystemExit(run(parser.parse_args()))
