"""Archive #94/#125/#126/#145/#151/#154-156, D.4: one real full-writer arm.

Run with the existing durable supervisor, --timeout 180. No retry/CPU fallback,
no whole-model claim. Preserve independent oracle and actual task profile.
"""
import argparse
import ctypes as C
import json
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

ARMS={'old':('oscar_rotate_clip_store_striped_kernel','csrc/experiments/striped_store_simd_pack/writer_baseline.cpp'),
      'current':('oscar_rotate_clip_store_striped_simd_pack_init_kernel','csrc/experiments/striped_store_simd_pack/writer_pack_init.cpp'),
      'batch8':('oscar_rotate_clip_store_striped_batch8_kernel','csrc/experiments/hadamard_batch8/writer_batch.cpp')}


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    save_json(out/'owner.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'output':str(out),'script':str(Path(__file__).resolve())})
    report={'status':'initializing','arm':args.arm,'phase':'inputs','scope':'one complete new-KV striped writer',
            'NPU':'not_run','graph':'not_run','whole_service':'not_run','promotion_allowed':False}
    stopped=threading.Event();thread=None
    try:
        generate.check();binding=generate.source_binding()
        fixture=args.fixture.resolve();meta=json.loads((fixture/'fixture.json').read_text());g=meta['shape_fields']
        if 'writer_source_binding' in meta and meta['writer_source_binding']!=binding:
            raise ValueError('fixture writer source binding differs')
        n,h,d,dtype=(g[k] for k in ('tokens','heads','dim','dtype'))
        if n<=0 or h<=0 or d not in (64,128,256) or dtype not in (1,2):raise ValueError('invalid fixture geometry')
        grid=min(32,(n*h+7)//8)
        if meta['production_launch_grid']!=grid:raise ValueError('binding grid differs')
        config=args.simulator_config.resolve()
        if os.environ.get('CAMODEL_CONFIG_PATH')!=str(config) or 'libruntime_camodel.so' not in os.environ.get('LD_PRELOAD',''):
            raise ValueError('explicit simulator child environment required')
        stars=json.loads((config/'config_stars.json').read_text());detail=json.loads((config/'config.json').read_text())
        stock=json.loads(Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/config.json').read_text())
        if stars.get('model_top')!={'sim_type':0,'num_aic':24,'num_aiv':48} or detail.get('L2CACHE')!=stock.get('L2CACHE'):
            raise ValueError('simulator topology/cache changed')
        input_names=('key','value','rk','rv','slots','positions');output_names=('packed','raw_key','raw_value','tags','status')
        inputs={k:(fixture/(k+'.bin')).read_bytes() for k in input_names}
        initial={k:(fixture/('initial_'+k+'.bin')).read_bytes() for k in output_names}
        oracle={k:(fixture/('expected_'+k+'.bin')).read_bytes() for k in output_names}
        for prefix,values in (('',inputs),('initial_',initial),('expected_',oracle)):
            for name,data in values.items():
                if len(data)!=meta['files'][prefix+name+'.bin']['bytes'] or sha(data)!=meta['files'][prefix+name+'.bin']['sha256']:
                    raise ValueError('fixture bytes/hash differ: '+prefix+name)
        kernel,source=ARMS[args.arm];library=args.library.resolve()
        build_manifest=library.parent.parent/'frozen_source_binding.json'
        if json.loads(build_manifest.read_text())!=binding:
            raise ValueError('research build does not bind the frozen A/B/C sources')
        report.update(case=meta['name'],geometry=g,grid=grid,kernel=kernel,kernel_sha256=sha((ROOT/source).read_bytes()),
            kernel_source=source,source_binding=binding,build_manifest_sha256=sha(build_manifest.read_bytes()),
            library_sha256=sha(library.read_bytes()),fixture_sha256=sha((fixture/'fixture.json').read_bytes()),
            input_sha256={k:sha(v) for k,v in inputs.items()},initial_sha256={k:sha(v) for k,v in initial.items()},
            oracle_sha256={k:sha(v) for k,v in oracle.items()},
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
        launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*11+[C.c_int64]*3+[C.c_int32]+[C.c_int64]*9+[C.c_float]*2+[C.c_bool]
        launch.restype=C.c_uint32;ptrs={};owned={};hosts=[];guard=b'\xa5'*64
        all_initial={**inputs,**initial}
        for name,data in all_initial.items():
            pointer=C.c_void_p();check('alloc '+name,alloc(C.byref(pointer),len(data)+128,0));owned[name]=pointer
            ptrs[name]=C.c_void_p(pointer.value+64);payload=guard+data+guard;buf=C.create_string_buffer(payload,len(payload));hosts.append(buf)
            check('upload '+name,copy(pointer,len(payload),buf,len(payload),1))
        stream=C.c_void_p();check('stream',create(C.byref(stream)))
        attrs=[n,h,d,dtype]+[g[k] for k in ('block_tokens','blocks','offset','page_stride','key_stride','value_stride','tag_stride','sink','recent')]+[g['key_clip'],g['value_clip'],bool(g['hadamard'])]
        report['phase']='kernel_running';save_json(out/'report.json',report)
        print('PACK_WRITER_START',args.arm,meta['name'],'grid',grid,flush=True)
        check('launch',launch(grid,stream,*[ptrs[k] for k in (*input_names,*output_names)],*attrs));check('sync',sync(stream))
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
        for name,data in outputs.items():
            bad=[i for i,(a,b) in enumerate(zip(data,oracle[name])) if a!=b]
            if len(data)!=len(oracle[name]) or bad:
                failures.append(name+'_oracle');differences[name]={'count':len(bad),'first':bad[:8]}
        report.update(outputs=saved,failures=failures,differences=differences,status='numeric_checked_pending_profile',phase='finalize')
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
        print('PACK_WRITER_RESULT',json.dumps({'arm':args.arm,'status':report['status'],'ticks':report['task_ticks'],'failures':failures,'report':str(out/'report.json')}),flush=True)
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
