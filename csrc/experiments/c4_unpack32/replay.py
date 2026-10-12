"""Archive #13-22/#94/#126/#145/#154/#155; D.4: one complete helper CAModel arm.

Run under the durable bounded supervisor, same library/input/config for A/B.
Includes helper setup/copies; never a full C4/NPU/graph performance claim.
"""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
import tomllib

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_unpack32 import generate
from csrc.experiments.window_batch_decode.replay_camodel import sha,save_json,entries


def run(a):
    if (Path(__file__).parent/'unpack32.h').read_text()!=generate.helper():raise ValueError('helper source drift')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);log=out/'log_ca';log.mkdir()
    os.environ['CAMODEL_LOG_PATH']=str(log);os.environ['STARS_LOG_PATH']=str(log)
    meta=json.loads((a.fixture/'fixture.json').read_text());n=int((a.fixture/'shape.txt').read_text())
    if not 1<=n<=256 or n!=meta['case_count']:raise ValueError('case bound')
    inputs={name:(a.fixture/(name+'.bin')).read_bytes() for name in ('packed','live')}
    wanted={name:(a.fixture/(name+'.bin')).read_bytes() for name in ('expected','expected_status')}
    for name,data in {**inputs,**wanted}.items():
        if sha(data)!=meta['sha256'][name]:raise ValueError('fixture changed: '+name)
    if tuple(map(len,inputs.values()))!=(n*5120,n*4) or tuple(map(len,wanted.values()))!=(n*65536,n*4):raise ValueError('ABI byte size')
    config=Path(os.environ['CAMODEL_CONFIG_PATH']);config_data={name:json.loads((config/name).read_text()) for name in ('config.json','config_stars.json')}
    report={'status':'initialized','variant':a.variant,'cases':n,'scope':'32row helper, setup/copies included, not full-C4',
        'library_sha256':sha(a.library.read_bytes()),'fixture_sha256':meta['sha256'],'source_pins':generate.PINS,
        'helper_sha256':sha((Path(__file__).parent/'unpack32.h').read_bytes()),'simulator_config':config_data,
        'NPU':'not_run','graph':'not_run','production':'not_enabled'}
    save_json(out/'inputs.json',report);stop=threading.Event()
    def quiet():
        fd=os.open('/dev/null',os.O_WRONLY)
        try:
            while not stop.is_set():
                for name in os.listdir('/proc/self/fd'):
                    if not name.isdecimal() or int(name)==fd:continue
                    try:
                        target=os.readlink('/proc/self/fd/'+name)
                        if target.startswith(str(log)+'/') and target.endswith('.dump'):os.dup2(fd,int(name))
                    except OSError:continue
                time.sleep(.002)
        finally:os.close(fd)
    thread=threading.Thread(target=quiet,daemon=True);thread.start()
    acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
    def api(name,types):
        f=getattr(acl,name);f.argtypes=types;f.restype=C.c_int;return f
    def check(name,rc):
        if rc:raise RuntimeError(name+' rc='+str(rc))
    init=api('aclInit',[C.c_char_p]);setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int]);final=api('aclFinalize',[])
    alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
    copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int])
    create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)]);sync=api('aclrtSynchronizeStream',[C.c_void_p]);destroy=api('aclrtDestroyStream',[C.c_void_p])
    check('init',init(None));check('device',setdev(0))
    lib=C.CDLL(str(a.library.resolve()),mode=C.RTLD_GLOBAL);launch=getattr(lib,'aclrtlaunch_oscar_unpack'+a.variant+'_helper')
    launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*4+[C.c_int32];launch.restype=C.c_uint32
    initial={**inputs,'output':b'\x85'*(n*65536),'status':b'\x85'*(n*4)};owners={};ptrs={};hosts=[];guard=b'\xa5'*64
    for name,data in initial.items():
        ptr=C.c_void_p();check('allocate',alloc(C.byref(ptr),len(data)+128,0));owners[name]=ptr;ptrs[name]=C.c_void_p(ptr.value+64)
        payload=guard+data+guard;host=C.create_string_buffer(payload,len(payload));hosts.append(host)
        check('upload',copy(ptr,len(payload),host,len(payload),1))
    stream=C.c_void_p();check('stream',create(C.byref(stream)))
    check('launch',launch(1,stream,ptrs['packed'],ptrs['live'],ptrs['output'],ptrs['status'],n));check('sync',sync(stream))
    outputs={};failures=[]
    for name,data in initial.items():
        host=C.create_string_buffer(len(data)+128);check('download',copy(host,len(host),owners[name],len(host),2))
        raw=host.raw
        if raw[:64]!=guard or raw[-64:]!=guard:failures.append(name+'_guard')
        if name in inputs:
            if raw[64:-64]!=data:failures.append(name+'_input_changed')
        else:
            outputs[name]=raw[64:-64];(out/(name+'.bin')).write_bytes(outputs[name])
    if outputs['status']!=wanted['expected_status']:failures.append('status_oracle')
    for offset in range(0,len(outputs['output']),4):
        actual=outputs['output'][offset:offset+4];expected=wanted['expected'][offset:offset+4]
        if math.isnan(struct.unpack('<f',expected)[0]):
            if not math.isnan(struct.unpack('<f',actual)[0]):failures.append('nan_class');break
        elif actual!=expected:failures.append('FP32_oracle_byte_'+str(offset));break
    report.update(status='numeric_passed_pending_profile' if not failures else 'failed',failures=failures,
        outputs={name:{'bytes':len(data),'sha256':sha(data),'path':name+'.bin'} for name,data in outputs.items()})
    save_json(out/'completion.json',report)
    for ptr in owners.values():check('free',free(ptr))
    check('destroy',destroy(stream));check('reset',reset(0));check('finalize',final());stop.set();thread.join(timeout=2)
    profiles={}
    for kind in ('task','aiv'):
        rows=list(entries(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
        if len(rows)!=1 or any(r['duration']<=0 or r['duration']!=r['end_cycle']-r['start_cycle'] for r in rows):failures.append(kind+'_profile')
        profiles[kind]=rows
    if len(profiles['task'])==len(profiles['aiv'])==1:
        t,v=profiles['task'][0],profiles['aiv'][0]
        if v['core_id']!=0 or any(t[k]!=v[k] for k in ('task_id','stream_id')):failures.append('actor_identity')
    report.update(status='failed' if failures else 'passed',profiles=profiles,task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None)
    save_json(out/'report.json',report);print(json.dumps({'variant':a.variant,'status':report['status'],'task_ticks':report['task_ticks'],'failures':failures}),flush=True)
    return 2 if failures else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('16','32'),required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raise SystemExit(run(a))
