"""Archive #94/#126/#154: bounded metadata-only CAModel arm, independent oracle.

Run through run_camodel_arm.py; timings include output copies and setup.
Never interpreted as whole-store, NPU, graph, or service performance.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import tomllib

p=argparse.ArgumentParser()
p.add_argument('--variant',choices=('old','vector'),required=True)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--dim',type=int,choices=(64,128,256),default=256)
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
log=a.output/'log_ca';log.mkdir()
os.environ['CAMODEL_LOG_PATH']=str(log);os.environ['STARS_LOG_PATH']=str(log)
report={'variant':a.variant,'dim':a.dim,'scope':'metadata_initialization_only',
        'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
        'source_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                         for n in ('kernel.cpp','vector_indices.h')},
        'NPU':'not_run','graph':'not_run','full_store':'not_run'}
(a.output/'inputs.json').write_text(json.dumps(report,indent=2)+'\n')
acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
def api(name,types):
    f=getattr(acl,name);f.argtypes=types;f.restype=C.c_int;return f
def check(name,rc):
    if rc:raise RuntimeError(f'{name} rc={rc}')
init=api('aclInit',[C.c_char_p]);setdev=api('aclrtSetDevice',[C.c_int])
alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int])
copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int])
create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)])
sync=api('aclrtSynchronizeStream',[C.c_void_p]);free=api('aclrtFree',[C.c_void_p])
destroy=api('aclrtDestroyStream',[C.c_void_p]);reset=api('aclrtResetDevice',[C.c_int]);finalize=api('aclFinalize',[])
check('init',init(None));check('device',setdev(0))
lib=C.CDLL(str(a.library.resolve()),mode=C.RTLD_GLOBAL)
launch=getattr(lib,'aclrtlaunch_oscar_hadamard_indices_'+a.variant)
launch.argtypes=[C.c_uint32,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int32];launch.restype=C.c_uint32
size=8192;guard=b'\xa5'*64;initial=guard+b'\xa5'*size+guard
pointers=[]
for _ in range(2):
    ptr=C.c_void_p();check('malloc',alloc(C.byref(ptr),len(initial),0))
    host=C.create_string_buffer(initial,len(initial));check('h2d',copy(ptr,len(initial),host,len(initial),1));pointers.append(ptr)
stream=C.c_void_p();check('stream',create(C.byref(stream)))
check('launch',launch(1,stream,*[C.c_void_p(x.value+64) for x in pointers],a.dim));check('sync',sync(stream))
expected_i=[0xa5a5a5a5]*2048;expected_s=[0xa5a5a5a5]*2048
dist=1;stage=0
while dist<a.dim:
    for i in range(a.dim):
        expected_i[stage*a.dim+i]=(i^dist)*4
        expected_s[stage*a.dim+i]=0xbf800000 if i&dist else 0x3f800000
    dist*=2;stage+=1
failures=[]
for ptr,name,want in zip(pointers,('indices','signs'),(expected_i,expected_s)):
    buffer=C.create_string_buffer(len(initial));check('d2h',copy(buffer,len(initial),ptr,len(initial),2));data=buffer.raw
    (a.output/(name+'.guarded.bin')).write_bytes(data)
    if data[:64]!=guard or data[-64:]!=guard:failures.append(name+'_guard')
    if data[64:-64]!=struct.pack('<2048I',*want):failures.append(name+'_formula')
report.update(status='numeric_passed_pending_profile' if not failures else 'failed',failures=failures)
(a.output/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
for ptr in pointers:check('free',free(ptr))
check('destroy',destroy(stream));check('reset',reset(0));check('finalize',finalize())
def records(node):
    if isinstance(node,dict):
        if {'start_cycle','end_cycle','duration'}<=node.keys():yield node
        else:
            for child in node.values():yield from records(child)
profiles={}
for kind in ('task','aiv'):
    rows=list(records(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
    if len(rows)!=1 or any(r['duration']<=0 or r['end_cycle']-r['start_cycle']!=r['duration'] for r in rows):failures.append(kind+'_profile')
    profiles[kind]=rows
if len(profiles['task'])==len(profiles['aiv'])==1:
    t,v=profiles['task'][0],profiles['aiv'][0]
    if any(t[k]!=v[k] for k in ('task_id','stream_id')) or v['core_id']!=0:failures.append('actor_identity')
report.update(status='passed' if not failures else 'failed',profiles=profiles,
              task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'variant':a.variant,'status':report['status'],'task_ticks':report['task_ticks'],'failures':failures}))
raise SystemExit(2 if failures else 0)
