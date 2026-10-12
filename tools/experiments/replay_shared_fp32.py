"""Archive #126/#145/#148-155/D.4: local CAModel source0 comparison only.

Uses direct ACL for the same CANN library/source as official CPU-debug.
The performance denominator defaults to the current striped C4 route.
Task cycles are model ticks, never NPU milliseconds or whole-run throughput.
"""
import argparse,ctypes,hashlib,json,math,os,struct,threading,time
from pathlib import Path
from ctypes import c_void_p,c_size_t,c_int,c_int64,c_uint32,c_float,byref
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--fixture',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--baseline',choices=('fe0','striped_cluster4'),default='striped_cluster4')
p.add_argument('--library',type=Path)
p.add_argument('--query-starts',help='explicit source0 task subset; empty means all original groups')
p.add_argument('--cv-grid',type=int,choices=(1,20),default=20)
p.add_argument('--producer-grid',type=int,choices=(1,40),default=40)
p.add_argument('--baseline-results',type=Path,
               help='candidate-only run; compare with previously saved baseline outputs')
p.add_argument('--baseline-task-profile',type=Path,
               help='previously completed matched-grid baseline task profile; first task only')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
log_root=a.output/'log_ca';log_root.mkdir(exist_ok=True)
os.environ['CAMODEL_LOG_PATH']=str(log_root);os.environ['STARS_LOG_PATH']=str(log_root)
stop=threading.Event();redirected=set()
def quiet_own_dumps():
    sink=os.open('/dev/null',os.O_WRONLY)
    try:
        while not stop.is_set():
            for item in os.listdir('/proc/self/fd'):
                if not item.isdigit():continue
                fd=int(item)
                if fd==sink:continue
                try:
                    target=os.readlink(f'/proc/self/fd/{fd}')
                    if target.startswith(str(log_root)+'/') and target.endswith('.dump'):
                        os.dup2(sink,fd);redirected.add(target)
                except OSError:pass
            time.sleep(.001)
    finally:os.close(sink)
thread=threading.Thread(target=quiet_own_dumps,daemon=True);thread.start()
acl=ctypes.CDLL('libascendcl.so',mode=ctypes.RTLD_GLOBAL)
def api(name,args):
    fn=getattr(acl,name);fn.argtypes=args;fn.restype=c_int;return fn
init=api('aclInit',[ctypes.c_char_p]);setdev=api('aclrtSetDevice',[c_int]);reset=api('aclrtResetDevice',[c_int]);finalize=api('aclFinalize',[])
alloc=api('aclrtMalloc',[ctypes.POINTER(c_void_p),c_size_t,c_int]);free=api('aclrtFree',[c_void_p])
memcpy=api('aclrtMemcpy',[c_void_p,c_size_t,c_void_p,c_size_t,c_int]);create=api('aclrtCreateStream',[ctypes.POINTER(c_void_p)])
destroy=api('aclrtDestroyStream',[c_void_p]);sync=api('aclrtSynchronizeStream',[c_void_p])
def check(label,rc):
    if rc:raise RuntimeError(f'{label}: rc={rc}')
check('init',init(None));check('setdev',setdev(0))
library=a.library or a.root/'build-device-final/lib/liboscar_shared_fp32_history_experimental.so'
lib=ctypes.CDLL(str(library),mode=ctypes.RTLD_GLOBAL)
baseargs=[c_uint32,c_void_p]+[c_void_p]*15+[c_int64]*17+[c_float]
old=lib.aclrtlaunch_oscar_attention_cv_kernel;old.argtypes=baseargs;old.restype=c_uint32
cluster=lib.aclrtlaunch_oscar_attention_cv_striped_cluster4_kernel
cluster.argtypes=[c_uint32,c_void_p]+[c_void_p]*16+[c_int64]*17+[c_float];cluster.restype=c_uint32
consumer=lib.aclrtlaunch_oscar_shared_fp32_history_consumer_kernel
consumer.argtypes=baseargs+[c_void_p]*4+[c_int64]*2;consumer.restype=c_uint32
producer=lib.aclrtlaunch_oscar_shared_fp32_history_producer_kernel
producer.argtypes=[c_uint32,c_void_p]+[c_void_p]*5+[c_int64]*8;producer.restype=c_uint32
fixture=a.root/'oscar-shared-fp32-fixtures'/a.fixture
shape=list(map(int,(fixture/'shape.txt').read_text().split()))
n,hq,hk,d,context,sink,recent,spec,b,blocks,prefix,stride,cores,requests,splits,columns=shape
assert (hq,hk,d,requests,splits)==(6,1,256,1,1)
cores=a.cv_grid;count=n*3;window=sink+recent+spec
attrs=[n,hq,hk,d,requests,columns,count,b,blocks,prefix,stride,window*d,window,sink,recent,spec,splits]
ptrs={};host_buffers=[];input_hashes={}
taskbytes=(fixture/'tasks_source0.bin').read_bytes()
taskrows=[list(row) for row in struct.iter_unpack('<16q',taskbytes)]
selected=None if a.query_starts is None else set(map(int,a.query_starts.split(',')))
if selected is not None:
    found=set()
    for row in taskrows:
        if row[1]>0 and row[7]==0:
            if row[0] in selected:found.add(row[0])
            else:row[1]=0
    if found!=selected:raise ValueError('query-starts includes a nonleader task')
taskbytes=b''.join(struct.pack('<16q',*row) for row in taskrows)
(a.output/'tasks_source0.bin').write_bytes(taskbytes)
active_tokens=sorted({token for row in taskrows if row[1]>0 and row[7]==0
                      for token in range(row[0],row[0]+row[1])})
def upload(name,contents):
    ptr=c_void_p();check('alloc '+name,alloc(byref(ptr),len(contents),0));ptrs[name]=ptr
    buf=ctypes.create_string_buffer(contents,len(contents));host_buffers.append(buf)
    check('h2d '+name,memcpy(ptr,len(contents),buf,len(contents),1))
for name,path in {'q':'q','qr':'qr','ck':'ck','cv':'cv','rv':'rv','raw':'raw','striped':'raw_striped',
                  'table':'table','wk':'wk','wv':'wv','tags':'tags','tasks':'tasks_source0'}.items():
    contents=taskbytes if name=='tasks' else (fixture/(path+'.bin')).read_bytes()
    upload(name,contents)
    input_hashes[path]=hashlib.sha256(contents).hexdigest()
max_end=max(row[4] for row in taskrows if row[1]>0 and row[7]==0)
output_sizes={'partial':n*hq*3*d*4,'lse':n*hq*3*4,'status':count*2*4}
cached_baseline=None
if a.baseline_results is not None:
    cached_baseline=json.loads((a.baseline_results/'report.json').read_text())
    if cached_baseline['status']!='passed' or cached_baseline['baseline']!=a.baseline or \
            cached_baseline['input_sha256']!=input_hashes or \
            cached_baseline['library_sha256']!=hashlib.sha256(library.read_bytes()).hexdigest():
        raise ValueError('saved baseline does not bind identical inputs and CANN library')
    if (a.baseline_results/'tasks_source0.bin').read_bytes()!=taskbytes:
        raise ValueError('saved baseline task bytes differ')
for tag in ('old','new'):
    for name,size in output_sizes.items():upload(tag+name,bytes([0 if name=='status' else 0x85])*size)
for name,size in {'workspace':cores*1839104,'clusterStats':cores*8*8,'sharedK':8192*d*4,'sharedV':8192*d*4,
                  'errors':8192*4,'state':((n+20)//21)*2*(64*d+128+8)*4}.items():
    upload(name,bytes([0x85])*size)
stream=c_void_p();check('create',create(byref(stream)))
def cv_ptrs(tag):
    return [ptrs[x] for x in ('q','qr','ck','cv','rv','raw','table','wk','wv','tags','tasks',
                             tag+'partial',tag+'lse',tag+'status','workspace')]
launches=[]
stage_outputs={}
def save_stage(stage,names):
    manifest={}
    for name,size in names.items():
        buf=ctypes.create_string_buffer(size)
        check('d2h '+stage+' '+name,memcpy(buf,size,ptrs[name],size,2))
        contents=buf.raw;path=a.output/(stage+'_'+name+'.bin');path.write_bytes(contents)
        manifest[name]={'path':str(path),'bytes':size,'sha256':hashlib.sha256(contents).hexdigest()}
    stage_outputs[stage]=manifest
    (a.output/(stage+'_completion.json')).write_text(json.dumps({
        'status':'completed_and_saved','stage':stage,'outputs':manifest,
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'input_sha256':input_hashes},indent=2)+'\n')
    print('SAVED',stage,'outputs',list(manifest),flush=True)
if cached_baseline is None:
    print('START',a.baseline,'grid',cores,flush=True)
    if a.baseline=='fe0':
        check('old launch',old(cores,stream,*cv_ptrs('old'),*attrs,1/16.0))
    else:
        baseline_ptrs=cv_ptrs('old');baseline_ptrs[5]=ptrs['striped']
        check('C4 launch',cluster(cores,stream,*baseline_ptrs,ptrs['clusterStats'],*attrs,1/16.0))
    check('old sync',sync(stream))
    launches.append(a.baseline);print('COMPLETED',a.baseline,a.fixture,flush=True)
    save_stage('baseline',{('old'+name):size for name,size in output_sizes.items()})
else:
    print('CACHED_BASELINE',str(a.baseline_results),'numeric_grid',cached_baseline['launch_AIC_grid'],
          'candidate_grid',cores,flush=True)
for begin in range(sink,max_end,8192):
    end=min(begin+8192,context)
    print('START producer',begin,end,'grid',a.producer_grid,flush=True)
    check('producer launch',producer(a.producer_grid,stream,*[ptrs[x] for x in ('striped','table','sharedK','sharedV','errors')],
                                    context,begin,end,columns,b,blocks,prefix,stride))
    check('producer sync',sync(stream));launches.append(f'producer_{begin}')
    print('COMPLETED producer',begin,end,flush=True)
    save_stage(f'producer_{begin}',{'sharedK':8192*d*4,'sharedV':8192*d*4,'errors':8192*4})
    print('START consumer',begin,end,'grid',cores,flush=True)
    check('consumer launch',consumer(cores,stream,*cv_ptrs('new'),*attrs,1/16.0,
                                    *[ptrs[x] for x in ('sharedK','sharedV','errors','state')],begin,end))
    check('consumer sync',sync(stream));launches.append(f'consumer_{begin}')
    print('COMPLETED consumer',begin,end,flush=True)
    save_stage(f'consumer_{begin}',{('new'+name):size for name,size in output_sizes.items()})
outputs={};failures=[]
for tag in ('old','new'):
    for name,size in output_sizes.items():
        if tag=='old' and cached_baseline is not None:
            contents=(a.baseline_results/('old_'+name+'.bin')).read_bytes()
            if len(contents)!=size:raise ValueError('cached baseline output byte count differs')
        else:
            buf=ctypes.create_string_buffer(size);check('d2h',memcpy(buf,size,ptrs[tag+name],size,2))
            contents=buf.raw
        outputs[tag+name]=contents;(a.output/(tag+'_'+name+'.bin')).write_bytes(contents)
for name in output_sizes:
    if outputs['old'+name]!=outputs['new'+name]:failures.append(name+'_bitwise')
    print('BITWISE',name,outputs['old'+name]==outputs['new'+name],flush=True)
if any(struct.unpack('<'+str(count*2)+'i',outputs['newstatus'])):failures.append('status_nonzero')
expected=(fixture/'source0_output.bin').read_bytes();expected_lse=(fixture/'source0_lse.bin').read_bytes()
maxout=maxlse=0.0;badout=badlse=0
for row in [token*hq+head for token in active_tokens for head in range(hq)]:
    x=struct.unpack_from('<f',outputs['newlse'],row*3*4)[0];y=struct.unpack_from('<f',expected_lse,row*4)[0]
    maxlse=max(maxlse,abs(x-y));badlse+=not math.isfinite(x) or abs(x-y)>.005+.005*abs(y)
    for col in range(d):
        x=struct.unpack_from('<f',outputs['newpartial'],(row*3*d+col)*4)[0]
        y=struct.unpack_from('<f',expected,(row*d+col)*4)[0]
        maxout=max(maxout,abs(x-y));badout+=not math.isfinite(x) or abs(x-y)>.005+.005*abs(y)
if badout or badlse:failures.append('frozen_oracle')
clusters=None
if cached_baseline is not None:
    clusters=cached_baseline['baseline_clusters']
elif a.baseline=='striped_cluster4':
    stats=ctypes.create_string_buffer(cores*8*8)
    check('stats d2h',memcpy(stats,len(stats),ptrs['clusterStats'],len(stats),2))
    clusters=sum(struct.unpack_from('<q',stats.raw,core*8*8)[0] for core in range(cores))
    if clusters<=0:failures.append('no_cluster_reuse')
for ptr in ptrs.values():check('free',free(ptr))
check('destroy',destroy(stream));check('reset',reset(0));check('finalize',finalize())
stop.set();thread.join(timeout=2)
import re
profile=(log_root/'profile_task_log0.toml').read_text()
ticks=[int(x) for x in re.findall(r'^\s*duration\s*=\s*(\d+)',profile,re.M)]
starts=[int(x) for x in re.findall(r'^\s*start_cycle\s*=\s*(\d+)',profile,re.M)]
ends=[int(x) for x in re.findall(r'^\s*end_cycle\s*=\s*(\d+)',profile,re.M)]
if len(ticks)!=len(launches):failures.append('profile_launch_count')
candidate_start=0 if cached_baseline is not None else 1
baseline_ticks=None
if a.baseline_task_profile is not None:
    previous=a.baseline_task_profile.read_text()
    previous_ticks=[int(x) for x in re.findall(r'^\s*duration\s*=\s*(\d+)',previous,re.M)]
    if not previous_ticks:raise ValueError('previous baseline profile is empty')
    baseline_ticks=previous_ticks[0]
report={'status':'passed' if not failures else 'failed','backend':'local_camodel_dav2201',
        'fixture':a.fixture,'failures':failures,'old_new_bitwise':not any('bitwise' in f for f in failures),
        'oracle':{'max_output_abs':maxout,'max_lse_abs':maxlse,'bad_output':badout,'bad_lse':badlse},
        'task_ticks':dict(zip(launches,ticks)),'candidate_sum_ticks':sum(ticks[candidate_start:]),
        'candidate_stream_span_ticks':ends[-1]-starts[candidate_start] if len(starts)==len(ticks) and len(ticks)>candidate_start else None,
        'profile_start_cycles':starts,'profile_end_cycles':ends,
        'baseline':a.baseline,'baseline_clusters':clusters,
        'active_query_tokens':len(active_tokens),'query_start_subset':sorted(selected) if selected is not None else None,
        'scope':'source0 task subset only' if selected is not None else 'all source0 tasks only',
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'input_sha256':input_hashes,
        'model_AIC_count':24,'launch_AIC_grid':cores,'producer_AIV_grid':a.producer_grid,
        'stage_outputs_saved':stage_outputs,'diagnostic_d2h_between_stages':True,
        'candidate_only':cached_baseline is not None,
        'numeric_baseline_grid':cached_baseline['launch_AIC_grid'] if cached_baseline else cores,
        'matched_grid1_baseline_outputs_available':False if cached_baseline is not None else None,
        'separate_process_baseline_ticks':baseline_ticks,
        'baseline_task_profile':str(a.baseline_task_profile) if a.baseline_task_profile else None,
        'target_npu_verified':False,'full_service_verified':False,'promotion_allowed':False,
        'dump_files_redirected':len(redirected)}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if failures:raise SystemExit(2)
