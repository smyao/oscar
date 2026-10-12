"""Archive #94/#125/#126/#145/#151/#154-156, D.4: one real query-rotation arm.

Run with the existing durable supervisor, --timeout 180. No retry/CPU fallback,
no whole-model claim. Preserve finite/class oracle, exact cross-arm bytes and actual task profile.
"""
import argparse
import ctypes as C
import json
import math
import struct
import os
from pathlib import Path
import sys
import threading
import time
import tomllib
import traceback

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.hadamard_batch8 import generate
from csrc.experiments.window_batch_decode.replay_camodel import save_json,sha,entries

ARMS={
    'old':('oscar_rotate_kernel','csrc/kernels/rotate_clip_store.cpp'),
    'init':('oscar_rotate_batch8_init_kernel','csrc/experiments/hadamard_batch8/rotate_init.cpp'),
    'batch8':('oscar_rotate_batch8_kernel','csrc/experiments/hadamard_batch8/rotate_batch.cpp'),
}
PLAN={
    'n128_h6_d256_bf16':(768,256,2,6), 'r1_h1_d64_bf16_signed_zero':(1,64,2,1),
    'r7_h1_d64_fp16':(7,64,1,1), 'r8_h1_d128_bf16':(8,128,2,1),
    'r9_h3_d128_fp16_partial':(9,128,1,3), 'r7_h1_d256_bf16_nonfinite':(7,256,2,1),
    'r8_h1_d256_fp16_nonfinite':(8,256,1,1), 'r9_h3_d64_bf16_allpadding':(9,64,2,3),
}
SCOPE='isolated query rotation; not writer, attention or service performance'


def oracle_check(outputs,oracle,tolerance):
    # Host NaN payload is not a device golden. Cross-arm checks below retain
    # every output bit, including NaN payloads and +/-0; never use equal_nan.
    if len(outputs['output'])!=len(oracle['output']) or len(outputs['output'])%4:
        raise ValueError('wrong query output length')
    if len(outputs['status'])!=len(oracle['status']):raise ValueError('wrong status length')
    status_bad=sum(a!=b for a,b in zip(outputs['status'],oracle['status']))
    finite_bad=class_bad=0;first=[]
    for index,(actual,expected) in enumerate(zip(struct.iter_unpack('<f',outputs['output']),
                                                struct.iter_unpack('<f',oracle['output']))):
        got,want=actual[0],expected[0]
        if math.isfinite(want):
            bad=not math.isfinite(got) or abs(got-want)>tolerance['atol']+tolerance['rtol']*abs(want)
            finite_bad+=bad
        else:
            bad=not math.isnan(got) if math.isnan(want) else got!=want
            class_bad+=bad
        if bad and len(first)<8:first.append(index)
    return {'status':'passed' if not(status_bad or finite_bad or class_bad) else 'failed',
            'status_byte_differences':status_bad,'finite_failures':finite_bad,
            'nonfinite_class_failures':class_bad,'first_elements':first}


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    save_json(out/'owner.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'output':str(out),'script':str(Path(__file__).resolve())})
    report={'status':'initializing','arm':args.arm,'phase':'inputs','scope':SCOPE,
            'NPU':'not_run','graph':'not_run','whole_service':'not_run','promotion_allowed':False}
    stopped=threading.Event();thread=None
    try:
        generate.check();binding=generate.source_binding()
        fixture=args.fixture.resolve();meta=json.loads((fixture/'fixture.json').read_text())
        manifest=json.loads((fixture.parent/'cases.json').read_text())
        if [row for row in manifest['cases'] if row['case']==meta['case']]!=[meta]:
            raise ValueError('case differs from registered fixture manifest')
        rows,d,dtype,h=(meta[k] for k in ('rows','dim','dtype','heads'))
        if PLAN.get(meta['case'])!=(rows,d,dtype,h) or meta['hadamard'] is not True:
            raise ValueError('undeclared query fixture geometry')
        grid=min(32,(rows+7)//8);g={'rows':rows,'dim':d,'dtype':dtype,'heads':h,'hadamard':True}
        if meta['grid']!=grid or meta['shape']!=[rows,d,dtype,h,1,grid]:
            raise ValueError('binding grid/shape differs')
        expected_sources={'csrc/kernels/rotate_clip_store.cpp','csrc/kernels/oscar_common.h',
                          'configs/acceptance.json','csrc/experiments/hadamard_batch8/rotate_fixture.py'}
        if set(meta['source_sha256'])!=expected_sources or manifest['source_sha256']!=meta['source_sha256']:
            raise ValueError('fixture source inventory differs')
        for name,wanted in meta['source_sha256'].items():
            if sha((ROOT/name).read_bytes())!=wanted:raise ValueError('fixture source drift: '+name)
        acceptance=json.loads((ROOT/'configs/acceptance.json').read_text())
        tolerance=acceptance['store_dequant']
        if not acceptance['frozen_before_measurement'] or tolerance!=meta['independent_finite_tolerance']:
            raise ValueError('frozen independent oracle tolerance changed')
        config=args.simulator_config.resolve()
        if os.environ.get('CAMODEL_CONFIG_PATH')!=str(config) or 'libruntime_camodel.so' not in os.environ.get('LD_PRELOAD',''):
            raise ValueError('explicit simulator child environment required')
        stars=json.loads((config/'config_stars.json').read_text());detail=json.loads((config/'config.json').read_text())
        stock=json.loads(Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/config.json').read_text())
        if stars.get('model_top')!={'sim_type':0,'num_aic':24,'num_aiv':48} or detail.get('L2CACHE')!=stock.get('L2CACHE'):
            raise ValueError('simulator topology/cache changed')
        input_names=('source','rotation','slots');output_names=('output','status')
        sizes={'source':rows*d*2,'rotation':d*d*4,'slots':rows//h*8,'output':rows*d*4,'status':rows*4}
        inputs={k:(fixture/(k+'.bin')).read_bytes() for k in input_names}
        initial={k:b'\x85'*sizes[k] for k in output_names}
        oracle={k:(fixture/('expected_'+k+'.bin')).read_bytes() for k in output_names}
        for prefix,values in (('',inputs),('expected_',oracle)):
            for name,data in values.items():
                record=meta['files'][prefix+name+'.bin']
                if len(data)!=sizes[name] or len(data)!=record['bytes'] or sha(data)!=record['sha256']:
                    raise ValueError('fixture bytes/hash differ: '+prefix+name)
        shape_bytes=(fixture/'shape.txt').read_bytes();shape_record=meta['files']['shape.txt']
        if len(shape_bytes)!=shape_record['bytes'] or sha(shape_bytes)!=shape_record['sha256'] or list(map(int,shape_bytes.split()))!=meta['shape']:
            raise ValueError('fixture shape file differs')
        for name,data in oracle.items():(out/('expected_'+name+'.bin')).write_bytes(data)
        kernel,source=ARMS[args.arm];library=args.library.resolve()
        build_manifest=library.parent.parent/'frozen_source_binding.json'
        if json.loads(build_manifest.read_text())!=binding:
            raise ValueError('research build does not bind the frozen A/B/C sources')
        report.update(case=meta['case'],geometry=g,grid=grid,kernel=kernel,kernel_sha256=sha((ROOT/source).read_bytes()),
            kernel_source=source,source_binding=binding,build_manifest_sha256=sha(build_manifest.read_bytes()),
            library_sha256=sha(library.read_bytes()),fixture_sha256=sha((fixture/'fixture.json').read_bytes()),
            input_sha256={k:sha(v) for k,v in inputs.items()},initial_sha256={k:sha(v) for k,v in initial.items()},
            oracle_sha256={k:sha(v) for k,v in oracle.items()},oracle_tolerance=tolerance,
            fixture_sources=meta['source_sha256'],
            helpers={p:sha((ROOT/p).read_bytes()) for p in ('csrc/experiments/striped_store_simd_pack/pack.h',
                                                         'csrc/experiments/hadamard_indices/vector_indices.h')},
            simulator_configuration={'stars':stars,'detail':detail})
        save_json(out/'inputs.json',report);save_json(out/'report.json',report)
        log=out/'log_ca';log.mkdir();os.environ.update(CAMODEL_LOG_PATH=str(log),STARS_LOG_PATH=str(log))
        def quiet():
            sink=os.open('/dev/null',os.O_WRONLY)
            try:
                while not stopped.is_set():
                    for name in os.listdir('/proc/self/fd'):
                        if not name.isdecimal() or int(name)==sink:continue
                        try:
                            path=os.readlink('/proc/self/fd/'+name)
                            if path.startswith(str(log)+'/') and path.endswith('.dump'):os.dup2(sink,int(name))
                        except OSError:continue
                    time.sleep(.001)
            finally:os.close(sink)
        thread=threading.Thread(target=quiet,daemon=True);thread.start()
        acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
        def api(name,types):
            fn=getattr(acl,name);fn.argtypes=types;fn.restype=C.c_int;return fn
        def check(name,rc):
            if rc:
                report.update(failed_sdk_call=name,sdk_returncode=rc)
                raise RuntimeError(name+' rc='+str(rc))
        init=api('aclInit',[C.c_char_p]);final=api('aclFinalize',[]);setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int])
        alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
        copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int])
        create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)]);destroy=api('aclrtDestroyStream',[C.c_void_p]);sync=api('aclrtSynchronizeStream',[C.c_void_p])
        check('init',init(None));check('device',setdev(0))
        lib=C.CDLL(str(library),mode=C.RTLD_GLOBAL);launch=getattr(lib,'aclrtlaunch_'+kernel)
        launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*4+[C.c_int64]*2+[C.c_int32,C.c_bool,C.c_void_p,C.c_int64]
        launch.restype=C.c_uint32;ptrs={};owned={};hosts=[];guard=b'\xa5'*64
        all_initial={**inputs,**initial}
        for name,data in all_initial.items():
            pointer=C.c_void_p();check('alloc '+name,alloc(C.byref(pointer),len(data)+128,0));owned[name]=pointer
            ptrs[name]=C.c_void_p(pointer.value+64);payload=guard+data+guard;buf=C.create_string_buffer(payload,len(payload));hosts.append(buf)
            check('upload '+name,copy(pointer,len(payload),buf,len(payload),1))
        stream=C.c_void_p();check('stream',create(C.byref(stream)))
        report['phase']='kernel_running';save_json(out/'report.json',report)
        print('QUERY_ROTATE_START',args.arm,meta['case'],'grid',grid,flush=True)
        check('launch',launch(grid,stream,ptrs['source'],ptrs['rotation'],ptrs['output'],ptrs['status'],
                              rows,d,dtype,True,ptrs['slots'],h));check('sync',sync(stream))
        saved={};outputs={};failures=[];differences={}
        for name in (*output_names,*input_names):
            size=len(all_initial[name]);buf=C.create_string_buffer(size+128)
            check('download '+name,copy(buf,size+128,owned[name],size+128,2));data=buf.raw
            label=name if name in output_names else name+'_after'
            (out/(label+'.guarded.bin')).write_bytes(data);(out/(label+'.bin')).write_bytes(data[64:-64])
            saved[label]={'bytes':size,'sha256':sha(data[64:-64]),'guarded_sha256':sha(data),'path':label+'.bin'}
            if data[:64]!=guard or data[-64:]!=guard:failures.append(name+'_guard')
            if name in inputs and data[64:-64]!=inputs[name]:failures.append(name+'_input_changed')
            if name in output_names:outputs[name]=data[64:-64]
        save_json(out/'completion.json',{**report,'status':'device_complete_outputs_saved','outputs':saved,'failures':failures})
        oracle_result=oracle_check(outputs,oracle,tolerance)
        if oracle_result['status']!='passed':failures.append('independent_oracle')
        report.update(outputs=saved,failures=failures,oracle_result=oracle_result,
                      status='numeric_checked_pending_profile',phase='finalize')
        save_json(out/'report.json',report)
        for pointer in owned.values():check('free',free(pointer))
        check('destroy',destroy(stream));check('reset',reset(0));check('finalize',final());stopped.set();thread.join(timeout=2)
        profiles={}
        for kind,count in (('task',1),('aiv',grid)):
            records=list(entries(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
            if len(records)!=count or any(r['duration']<=0 or r['duration']!=r['end_cycle']-r['start_cycle'] for r in records):failures.append(kind+'_profile')
            profiles[kind]=records
        if len(profiles['task'])==1:
            task=profiles['task'][0]
            if ({r['core_id'] for r in profiles['aiv']}!=set(range(grid)) or
                any((r['task_id'],r['stream_id'])!=(task['task_id'],task['stream_id']) for r in profiles['aiv'])):failures.append('actor_identity')
        report.update(status='failed' if failures else 'passed',phase='complete',failures=failures,profiles=profiles,
                      task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None)
        save_json(out/'report.json',report)
        print('QUERY_ROTATE_RESULT',json.dumps({'arm':args.arm,'status':report['status'],'ticks':report['task_ticks'],'failures':failures,'report':str(out/'report.json')}),flush=True)
        return 2 if failures else 0
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc());save_json(out/'report.json',report)
        traceback.print_exc();return 2
    finally:
        stopped.set()
        if thread is not None:thread.join(timeout=2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=ARMS,required=True)
    for name in ('fixture','output','library','simulator-config'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
