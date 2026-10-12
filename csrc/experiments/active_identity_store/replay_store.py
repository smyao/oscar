"""Archive #86/#97/#111/#126/#145/#155/D.4: one matched CAModel store task.

Independent byte oracle, exact input dtype, explicit identity, no model/graph claim.
"""
import argparse,ctypes as C,hashlib,json,os,threading,time,tomllib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--variant',choices=('old','active','identity'),required=True)
p.add_argument('--case',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--grid',type=int,choices=(4,16),default=16)
a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
(a.output/'owner.json').write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),'cwd':str(Path.cwd()),'output':str(a.output)})+'\n')
fixture=a.root/'artifacts/active_identity_store_20261006'/a.case
meta=json.loads((fixture/'fixture.json').read_text());g=meta['shape_fields']
n,h,d,dtype=g['tokens'],g['heads'],g['dim'],g['dtype']
assert meta['name']==a.case and dtype in (1,2)
assert not (a.variant=='identity' and g['hadamard']), 'explicit-I entry cannot consume Hadamard'
if meta.get('production_binding_geometry'):assert a.grid==meta['production_launch_grid']
log=a.output/'log_ca';log.mkdir();os.environ['CAMODEL_LOG_PATH']=str(log);os.environ['STARS_LOG_PATH']=str(log)
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
def sha(data):return hashlib.sha256(data).hexdigest()
input_names=('key','value','rk','rv','slots','positions')
output_names=('packed','raw_key','raw_value','tags','status')
inputs={name:(fixture/(name+'.bin')).read_bytes() for name in input_names}
initial={name:(fixture/('initial_'+name+'.bin')).read_bytes() for name in output_names}
for name,data in inputs.items():assert sha(data)==meta['files'][name+'.bin']['sha256']
for name,data in initial.items():assert sha(data)==meta['files']['initial_'+name+'.bin']['sha256']
library=a.root/'build/device-final/lib/liboscar_active_identity_store.so'
kernel={'old':'oscar_rotate_clip_store_striped_kernel','active':'oscar_store_active_striped_kernel',
        'identity':'oscar_store_identity_active_striped_kernel'}[a.variant]
source=a.root/('csrc/kernels/rotate_clip_store_striped.cpp' if a.variant=='old' else
              'csrc/experiments/active_identity_store/active_identity_store.cpp')
report={'variant':a.variant,'case':a.case,'geometry':g,'grid':a.grid,'actors':str(a.grid)+' AIV',
 'scope':'one fused striped new-KV store; MTP/service/graph not included',
 'input_sha256':{k:sha(v) for k,v in inputs.items()},'initial_sha256':{k:sha(v) for k,v in initial.items()},
 'oracle_sha256':{k:meta['files']['expected_'+k+'.bin']['sha256'] for k in output_names},
 'library_sha256':sha(library.read_bytes()),'kernel_sha256':sha(source.read_bytes()),'kernel':kernel,
 'NPU':'not_run','graph':'not_run','model_quality':'not_run','whole_service':'not_run'}
(a.output/'inputs.json').write_text(json.dumps(report,indent=2)+'\n')
acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
def api(name,args):
 f=getattr(acl,name);f.argtypes=args;f.restype=C.c_int;return f
init=api('aclInit',[C.c_char_p]);setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int]);final=api('aclFinalize',[])
alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int]);create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)])
destroy=api('aclrtDestroyStream',[C.c_void_p]);sync=api('aclrtSynchronizeStream',[C.c_void_p])
def check(name,rc):
 if rc:raise RuntimeError(name+' rc='+str(rc))
check('init',init(None));check('setdev',setdev(0))
lib=C.CDLL(str(library),mode=C.RTLD_GLOBAL);launch=getattr(lib,'aclrtlaunch_'+kernel)
launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*11+[C.c_int64]*3+[C.c_int32]+[C.c_int64]*9+[C.c_float]*2+[C.c_bool]
launch.restype=C.c_uint32
ptrs={};hosts=[];all_initial={**inputs,**initial};guard=b'\xa5'*64
for name,data in all_initial.items():
 data+=guard;ptr=C.c_void_p();check('alloc '+name,alloc(C.byref(ptr),len(data),0));ptrs[name]=ptr
 host=C.create_string_buffer(data,len(data));hosts.append(host);check('h2d '+name,copy(ptr,len(data),host,len(data),1))
stream=C.c_void_p();check('stream',create(C.byref(stream)))
attrs=[n,h,d,dtype]+[g[k] for k in ('block_tokens','blocks','offset','page_stride','key_stride','value_stride','tag_stride','sink','recent')]+[g['key_clip'],g['value_clip'],bool(g['hadamard'])]
print('STORE_CAMODEL_START',a.variant,a.case,'grid',a.grid,flush=True)
check('launch',launch(a.grid,stream,*[ptrs[k] for k in (*input_names,*output_names)],*attrs));check('sync',sync(stream))
print('STORE_CAMODEL_COMPLETE',a.variant,a.case,flush=True)
manifest={};failures=[];mismatches={}
for name,data in all_initial.items():
 count=len(data)+64;buf=C.create_string_buffer(count);check('d2h '+name,copy(buf,count,ptrs[name],count,2));got=buf.raw
 if got[-64:]!=guard:failures.append(name+'_guard')
 if name in inputs:
  if got[:-64]!=data:failures.append(name+'_input_changed')
 else:
  path=a.output/(name+'.bin');path.write_bytes(got[:-64]);manifest[name]={'bytes':len(data),'sha256':sha(got[:-64]),'path':path.name}
  want=(fixture/('expected_'+name+'.bin')).read_bytes()
  assert sha(want)==report['oracle_sha256'][name], 'oracle hash drift: '+name
  bad=[i for i,(x,y) in enumerate(zip(got[:-64],want)) if x!=y]
  if len(want)!=len(data) or bad:failures.append(name+'_oracle');mismatches[name]={'count':len(bad),'first':bad[:8]}
completion={**report,'status':'device_complete_outputs_saved','outputs':manifest,'failures':failures,'mismatches':mismatches}
(a.output/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
print('STORE_CAMODEL_SAVED',a.variant,'oracle_failures',failures,flush=True)
report.update(outputs=manifest,failures=failures,mismatches=mismatches,status='numeric_passed_pending_profile' if not failures else 'failed')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for ptr in ptrs.values():check('free',free(ptr))
check('destroy',destroy(stream));check('reset',reset(0));check('finalize',final());stop.set();thread.join(timeout=2)
def entries(node):
 if isinstance(node,dict):
  if {'start_cycle','end_cycle','duration'}<=node.keys():yield node
  else:
   for child in node.values():yield from entries(child)
profiles={}
for kind,count in (('task',1),('aiv',a.grid)):
 records=list(entries(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
 if len(records)!=count or any(x['duration']<=0 or x['duration']!=x['end_cycle']-x['start_cycle'] for x in records):failures.append(kind+'_profile')
 profiles[kind]=records
if len(profiles['task'])==1:
 task=profiles['task'][0]
 if any((x['stream_id'],x['task_id'])!=(task['stream_id'],task['task_id']) for x in profiles['aiv']):failures.append('task_owner_mismatch')
 if {x['core_id'] for x in profiles['aiv']}!=set(range(a.grid)):failures.append('core_budget')
report.update(status='passed' if not failures else 'failed',failures=failures,profiles=profiles,
              task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None,
              dump_files_redirected=len(redirected),profile_scope='actor durations include waits; no subphase breakdown')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('STORE_CAMODEL_RESULT',json.dumps(report),flush=True)
raise SystemExit(2 if failures else 0)
