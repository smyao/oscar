"""Archive #94/#126/#140/#145/#154 and D.4: exact-input FP32/BF16 same-shape Matmul ratio.

One real Matmul arm. Save all bytes before gates; retain complete instruction
logs with a fixed cap. Unmarked task ticks are local only, never whole-decode speed.
Use the existing VM durable supervisor and a new output for every arm.
"""
import argparse
import ctypes as C
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import time
import tomllib
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.fp32_bf16_mmad_ratio import fixture as cases
def sha(data):return hashlib.sha256(data).hexdigest()
def save(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def records(node):
    if isinstance(node,dict):
        if {'start_cycle','end_cycle','duration'}<=node.keys():yield node
        else:
            for child in node.values():yield from records(child)


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'owner.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'scope':'ratio research'})
    log=out/'log_ca';log.mkdir()
    os.environ['CAMODEL_LOG_PATH']=str(log);os.environ['STARS_LOG_PATH']=str(log)
    config=Path(os.environ['CAMODEL_CONFIG_PATH'])
    if 'libruntime_camodel.so' not in os.environ.get('LD_PRELOAD',''):raise ValueError('explicit CAModel environment required')
    configuration={n:json.loads((config/n).read_text()) for n in ('config.json','config_stars.json')}
    stock=json.loads(Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/config.json').read_text())
    if configuration['config.json']['L2CACHE']!=stock['L2CACHE'] or configuration['config_stars.json'].get('model_top')!={'sim_type':0,'num_aic':24,'num_aiv':48}:
        raise ValueError('cache model/topology changed')
    meta=json.loads((a.fixture/'fixture.json').read_text());binding=json.loads(a.binding.read_text())
    for key,file in [('kernel_sha256',Path(__file__).parent/'kernel.cpp'),('common_sha256',ROOT/'csrc/kernels/oscar_common.h'),('baseline_sha256',ROOT/'csrc/kernels/attention_cv_decode_bundle.cpp')]:
        if sha(file.read_bytes())!=binding[key]:raise ValueError('build source differs '+key)
    if (binding['basic']!=[64,64,128] or binding['single']!=[128,256,256] or
            binding['C_dtype']!='float32' or binding['hf32'] is not False):
        raise ValueError('fixed Matmul specification changed')
    manifest=json.loads((a.fixture.parent/'cases.json').read_text())
    if meta['source']!=cases.check_source() or manifest['source']!=meta['source'] or [r for r in manifest['cases'] if r['case']==meta['case']]!=[meta]:
        raise ValueError('fixture source/inventory differs')
    live,padded,layout=meta['live_rows'],meta['padded_rows'],meta['layout']
    if (meta['case'],live,padded,layout) not in cases.PLAN or meta['n']!=256 or meta['k']!=256:
        raise ValueError('declared geometry only')
    files={name:(a.fixture/name).read_bytes() for name in meta['files']}
    for name,data in files.items():
        record=meta['files'][name]
        if len(data)!=record['bytes'] or sha(data)!=record['sha256']:raise ValueError('fixture changed '+name)
    if list(map(int,files['shape.txt'].split()))!=[live,padded,layout]:raise ValueError('shape file differs')
    mothers={n:list(struct.unpack('<'+str(len(files[n+'_int16.bin'])//2)+'h',files[n+'_int16.bin'])) for n in ('a','b')}
    for n,values in mothers.items():
        f32,bf16=cases.exact_operands(values)
        if f32!=files[n+'_fp32.bin'] or bf16!=files[n+'_bf16.bin']:raise ValueError('not identical exact operand encodings')
    if any(mothers['a'][live*256:]):raise ValueError('input padding not zero')
    golden,_=cases.integer_dot(mothers['a'],mothers['b'],live,padded,layout)
    if golden!=files['expected_output.bin']:raise ValueError('independent integer golden differs')
    inputs={n:files[n+'_'+a.arm+'.bin'] for n in ('a','b')}
    arm=0 if a.arm=='fp32' else 1
    if a.library.name!='liboscar_fp32_bf16_mmad_ratio.so':raise ValueError('explicit ratio library required')
    (out/'expected_output.bin').write_bytes(golden)
    report={'arm':a.arm,'case':meta['case'],'live_rows':live,'padded_rows':padded,'layout':layout,'status':'running',
        'scope':'same-value exactly representable FP32/BF16 operand Matmul; FP32 accumulator; not production','library_sha256':sha(a.library.read_bytes()),
        'binding':binding,'binding_sha256':sha(a.binding.read_bytes()),'fixture_sha256':sha((a.fixture/'fixture.json').read_bytes()),
        'input_sha256':{n:sha(data) for n,data in inputs.items()},
        'mother_sha256':{n:sha(files[n+'_int16.bin']) for n in ('a','b')},
        'oracle_sha256':sha(golden),'math_proof':meta['math_proof'],'simulator_config':configuration,
        'NPU':'not_run','graph':'not_run','trace_cap_per_file_bytes':16*1024*1024}
    save(out/'inputs.json',report)
    # The cap fails the diagnostic rather than dropping a suffix and claiming
    # complete MMAD/barrier evidence. The outer supervisor owns cleanup.
    stopped=threading.Event()
    def watch():
        while not stopped.wait(.02):
            for p in log.glob('*.dump'):
                if p.stat().st_size>=16*1024*1024:
                    save(out/'trace-cap.json',{'file':p.name,'bytes':p.stat().st_size,'status':'failed'})
                    os._exit(4)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
    def api(name,types):
        f=getattr(acl,name);f.argtypes=types;f.restype=C.c_int;return f
    def check(name,rc):
        if rc:raise RuntimeError(name+' rc='+str(rc))
    init=api('aclInit',[C.c_char_p]);setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int]);final=api('aclFinalize',[])
    alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
    copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int]);create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)])
    destroy=api('aclrtDestroyStream',[C.c_void_p]);sync=api('aclrtSynchronizeStream',[C.c_void_p])
    check('init',init(None));check('device',setdev(0))
    library=C.CDLL(str(a.library.resolve()),mode=C.RTLD_GLOBAL)
    launch=getattr(library,'aclrtlaunch_oscar_mmad_ratio_'+a.arm+'_kernel')
    launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*4+[C.c_int64]*2;launch.restype=C.c_uint32
    initial={'a':inputs['a'],'b':inputs['b'],'c':b'\x85'*(padded*256*4),'status':b'\x85'*64};owners={};ptrs={};guard=b'\xa5'*64
    for name,data in initial.items():
        p=C.c_void_p();check('malloc',alloc(C.byref(p),len(data)+128,0));owners[name]=p;ptrs[name]=C.c_void_p(p.value+64)
        payload=guard+data+guard;h=C.create_string_buffer(payload,len(payload));check('h2d',copy(p,len(payload),h,len(payload),1))
    stream=C.c_void_p();check('stream',create(C.byref(stream)))
    check('launch',launch(1,stream,*[ptrs[n] for n in initial],live,layout));check('sync',sync(stream))
    actual={};failures=[];output_records={}
    for name,data in initial.items():
        h=C.create_string_buffer(len(data)+128);check('d2h',copy(h,len(data)+128,owners[name],len(data)+128,2));raw=h.raw
        (out/(name+'.guarded.bin')).write_bytes(raw);actual[name]=raw[64:-64];(out/(name+'.bin')).write_bytes(actual[name])
        output_records[name]={'bytes':len(data),'sha256':sha(actual[name]),'guarded_sha256':sha(raw)}
        if raw[:64]!=guard or raw[-64:]!=guard:failures.append(name+'_guard')
        if name in ('a','b') and actual[name]!=data:failures.append(name+'_input_changed')
    status=list(struct.unpack('<16i',actual['status']))
    wanted=[1,arm,live,layout,4 if arm==0 else 2,64,64,128,128,256,256]
    if status[:11]!=wanted or not 0<status[11]<=512*1024 or status[15]!=0:failures.append('completion')
    bad=sum(x!=y for x,y in zip(actual['c'],golden))
    if len(actual['c'])!=len(golden) or bad:failures.append('exact_integer_oracle')
    report.update(status='numeric_passed_pending_profile' if not failures else 'failed',outputs=output_records,failures=failures,oracle_byte_differences=bad,tiling_fields=status)
    save(out/'completion.json',report)
    for p in owners.values():check('free',free(p))
    check('destroy',destroy(stream));check('reset',reset(0));check('finalize',final());stopped.set();watcher.join(timeout=2)
    profiles={}
    for kind in ('task','aic'):
        rows=list(records(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
        if len(rows)!=1 or any(r['duration']<=0 or r['duration']!=r['end_cycle']-r['start_cycle'] for r in rows):failures.append(kind+'_profile')
        profiles[kind]=rows
    if len(profiles['task'])==len(profiles['aic'])==1:
        t,c=profiles['task'][0],profiles['aic'][0]
        if c['core_id']!=0 or any(t[k]!=c[k] for k in ('task_id','stream_id')):failures.append('actor_identity')
    traces={}
    for p in log.glob('*.dump'):
        if p.stat().st_size>=16*1024*1024:failures.append('trace_cap')
        data=p.read_bytes();traces[p.name]={'bytes':len(data),'sha256':sha(data)}
        with gzip.open(str(p)+'.gz','wb') as f:f.write(data)
        p.unlink()
    for name in ('core0.cubecore0.instr_log.dump','core0.cubecore0.instr_popped_log.dump'):
        if name not in traces or traces[name]['bytes']==0:failures.append('missing_active_instruction_trace')
    report.update(status='passed' if not failures else 'failed',profiles=profiles,traces=traces,
                  task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None,
                  instruction_analysis='complete dumps saved; dtype/MMAD inspection separate',grid=1,N=256,K=256,
                  R_scope='complete Matmul task including internal movement/control, not peak arithmetic throughput')
    save(out/'report.json',report)
    print(json.dumps({'arm':a.arm,'status':report['status'],'task_ticks':report['task_ticks'],'failures':failures}))
    return 2 if failures else 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=('fp32','bf16'),required=True)
    for n in ('fixture','library','binding','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    try:rc=run(args)
    except Exception as error:
        owner=args.output/'owner.json'
        if owner.is_file() and json.loads(owner.read_text()).get('pid')==os.getpid():
            path=args.output/'report.json';report=json.loads(path.read_text()) if path.exists() else {}
            report.update(status='failed',error=str(error),traceback=traceback.format_exc(),NPU='not_run')
            save(path,report)
        traceback.print_exc();rc=2
    raise SystemExit(rc)
