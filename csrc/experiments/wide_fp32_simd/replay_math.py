"""Archive #126/#129/#140/#145/#148/#154/#155, D.4: one bounded CAModel math task.

No producer, task/page/source scheduling, graph, real NPU or full-service evidence.
Save D2H results before cleanup/profile parsing so a later failure loses no completion.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
import tomllib

p=argparse.ArgumentParser()
p.add_argument('--variant',choices=('kv256','wide'),required=True)
p.add_argument('--kv',type=int,choices=(512,1024),required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve()
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'owner.json').write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),
    'cwd':str(Path.cwd()),'script':str(Path(__file__).resolve()),'output':str(a.output)})+'\n')
log=a.output/'log_ca';log.mkdir()
os.environ['CAMODEL_LOG_PATH']=str(log);os.environ['STARS_LOG_PATH']=str(log)
stop=threading.Event();redirected=set()
def quiet_dumps():
    sink=os.open('/dev/null',os.O_WRONLY)
    try:
        while not stop.is_set():
            for item in os.listdir('/proc/self/fd'):
                if not item.isdigit() or int(item)==sink:continue
                try:
                    target=os.readlink('/proc/self/fd/'+item)
                    if target.startswith(str(log)+'/') and target.endswith('.dump'):
                        os.dup2(sink,int(item));redirected.add(target)
                except OSError:pass
            time.sleep(.001)
    finally:os.close(sink)
thread=threading.Thread(target=quiet_dumps,daemon=True);thread.start()
fixture=a.root/f'artifacts/wide_fp32_cpu_cases_20261006/b{a.kv}_normal'
rows,kv,dim,scale=map(float,(fixture/'shape.txt').read_text().split())
assert (rows,kv,dim,scale)==(126,a.kv,256,.0625)
rows,kv,dim=int(rows),int(kv),int(dim)
assert tuple(map(float,(fixture/'tolerance.txt').read_text().split()))==(.005,.005)
libname='oscar_kv256_fp32_experimental' if a.variant=='kv256' else 'oscar_wide_fp32_simd_experimental'
library=a.root/f'build/device-ab/lib/lib{libname}.so'
kernel=f'oscar_kv256_fp32_b{kv}_kernel' if a.variant=='kv256' else f'oscar_wide_fp32_simd_b{kv}_kernel'
source=a.root/'csrc/experiments/wide_fp32_simd'/('kv256_baseline.cpp' if a.variant=='kv256' else 'wide_fp32_simd.cpp')
def digest(data):return hashlib.sha256(data).hexdigest()
inputs={name:(fixture/path).read_bytes() for name,path in {
    'q':'qrot.bin','k':'krot.bin','v':'vrot.bin','rotation':'rv.bin','visible':'visible.bin'}.items()}
score_cols=256 if a.variant=='kv256' else kv
sizes={'workspace':rows*(score_cols+dim)*4,'output':rows*dim*4,'lse':rows*4,'status':rows*4}
report={'variant':a.variant,'M':rows,'D':dim,'B':kv,'grid':1,'actors':'AIC1+AIV2','hf32':False,
 'scope':'decoded FP32 math QK/V2/PV/normalize/RvT only; producer/task/source excluded',
 'input_sha256':{k:digest(v) for k,v in inputs.items()},'library_sha256':digest(library.read_bytes()),
 'kernel_sha256':digest(source.read_bytes()),'kernel':kernel,'workspace_bytes':sizes['workspace'],
 'ub_explicit_bytes':135872 if a.variant=='kv256' else 85696,
 'npu':'not_run','graph':'not_run','model_quality':'not_run','whole_service':'not_run'}
(a.output/'inputs.json').write_text(json.dumps(report,indent=2)+'\n')
acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
def api(name,args):
    f=getattr(acl,name);f.argtypes=args;f.restype=C.c_int;return f
init=api('aclInit',[C.c_char_p]);setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int]);final=api('aclFinalize',[])
alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int])
create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)]);destroy=api('aclrtDestroyStream',[C.c_void_p]);sync=api('aclrtSynchronizeStream',[C.c_void_p])
def check(label,rc):
    if rc:raise RuntimeError(f'{label} rc={rc}')
check('init',init(None));check('set_device',setdev(0))
lib=C.CDLL(str(library),mode=C.RTLD_GLOBAL);launch=getattr(lib,'aclrtlaunch_'+kernel)
launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*9+[C.c_float];launch.restype=C.c_uint32
ptrs={};hosts=[];initial={};guards=b'\xa5'*64
for name,data in {**inputs,**{name:struct.pack('<I',0x7fc12345)*(size//4) for name,size in sizes.items()}}.items():
    data+=guards;initial[name]=data;ptr=C.c_void_p();check('alloc '+name,alloc(C.byref(ptr),len(data),0));ptrs[name]=ptr
    buf=C.create_string_buffer(data,len(data));hosts.append(buf);check('h2d '+name,copy(ptr,len(data),buf,len(data),1))
stream=C.c_void_p();check('create_stream',create(C.byref(stream)))
print('MATH_START',a.variant,'B',kv,'grid1',flush=True)
t0=time.monotonic()
check('launch',launch(1,stream,*[ptrs[k] for k in ('q','k','v','rotation','visible','workspace','output','lse','status')],scale))
check('sync',sync(stream))
print('MATH_DEVICE_COMPLETE',a.variant,'B',kv,flush=True)
outputs={};manifest={};failures=[]
for name,ptr in ptrs.items():
    size=len(initial[name]);buf=C.create_string_buffer(size);check('d2h '+name,copy(buf,size,ptr,size,2));data=buf.raw
    if data[-64:]!=guards:failures.append(name+'_guard')
    if name in inputs and data!=initial[name]:failures.append(name+'_input_changed')
    if name in sizes:
        outputs[name]=data[:-64];path=a.output/(name+'.bin');path.write_bytes(outputs[name])
        manifest[name]={'bytes':len(outputs[name]),'sha256':digest(outputs[name]),'path':path.name}
completion={**report,'status':'device_complete_outputs_saved','outputs':manifest,'guard_failures':failures,
            'guest_host_seconds_to_saved':time.monotonic()-t0}
(a.output/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
print('MATH_SAVED',a.variant,json.dumps(manifest,sort_keys=True),flush=True)
expected=(fixture/'expected_output.bin').read_bytes();expected_lse=(fixture/'expected_lse.bin').read_bytes()
statuses=struct.unpack('<126i',outputs['status']);visible=struct.unpack('<126i',inputs['visible'])
if any(statuses):failures.append('status_nonzero')
maxout=maxlse=0.;badout=badlse=0
for got,want in zip(struct.unpack('<'+str(rows*dim)+'f',outputs['output']),struct.unpack('<'+str(rows*dim)+'f',expected)):
    badout+=not math.isfinite(got) or abs(got-want)>.005+.005*abs(want)
    if math.isfinite(got):maxout=max(maxout,abs(got-want))
for got,want in zip(struct.unpack('<126f',outputs['lse']),struct.unpack('<126f',expected_lse)):
    if math.isinf(want):badlse+=got!=want
    else:
        badlse+=not math.isfinite(got) or abs(got-want)>.005+.005*abs(want)
        if math.isfinite(got):maxlse=max(maxlse,abs(got-want))
if badout or badlse:failures.append('oracle')
pbad=0
for r in range(rows):
    for col in range(score_cols):
        value=struct.unpack_from('<f',outputs['workspace'],(r*score_cols+col)*4)[0]
        global_col=kv-score_cols+col
        pbad+=not math.isfinite(value) or value<0 or value>1 or (global_col>=visible[r] and value!=0)
if pbad:failures.append('P_domain')
report['oracle']={'output_max_abs':maxout,'lse_max_abs':maxlse,'bad_output':badout,'bad_lse':badlse,'bad_P':pbad}
report['outputs']=manifest;report['status']='numeric_passed_pending_profile' if not failures else 'failed'
report['failures']=failures;(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for ptr in ptrs.values():check('free',free(ptr))
check('destroy',destroy(stream));check('reset',reset(0));check('finalize',final())
stop.set();thread.join(timeout=2)
def entries(node):
    if isinstance(node,dict):
        if {'start_cycle','end_cycle','duration'}<=node.keys():yield node
        else:
            for child in node.values():yield from entries(child)
profiles={}
for kind,want in (('task',1),('aic',1),('aiv',2)):
    path=log/f'profile_{kind}_log0.toml'
    records=list(entries(tomllib.loads(path.read_text())))
    if len(records)!=want or any(r['duration']<=0 or r['duration']!=r['end_cycle']-r['start_cycle'] for r in records):
        failures.append(kind+'_profile_incomplete')
    profiles[kind]=records
if len(profiles['task'])==1:
    task=profiles['task'][0]
    if any((r['stream_id'],r['task_id'])!=(task['stream_id'],task['task_id']) for kind in ('aic','aiv') for r in profiles[kind]):
        failures.append('actor_task_mismatch')
    if {r['core_id'] for r in profiles['aic']}!={0} or {r['core_id'] for r in profiles['aiv']}!={0,1}:
        failures.append('unexpected_actor_owner')
report.update(status='passed' if not failures else 'failed',failures=failures,profiles=profiles,
              task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None,
              dump_files_redirected=len(redirected),profile_scope='actor durations include waits; no component breakdown')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('MATH_RESULT',json.dumps(report,sort_keys=True),flush=True)
raise SystemExit(2 if failures else 0)
