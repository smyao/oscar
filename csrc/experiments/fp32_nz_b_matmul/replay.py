"""Archive #94/#126/#140/#145/#154 and D.4: FP32 GM ND/NZ B same-shape upper-bound experiment.

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
from csrc.experiments.fp32_nz_b_matmul import fixture as cases
def sha(data):return hashlib.sha256(data).hexdigest()
def save(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def records(node):
    if isinstance(node,dict):
        if {'start_cycle','end_cycle','duration'}<=node.keys():yield node
        else:
            for child in node.values():yield from records(child)


def instruction_spec(log,live):
    # Complete/popped records must independently contain the same actual
    # F32F32 K128 init/accumulate sequence; integer inputs alone cannot prove it.
    sequences=[]
    for suffix in ('instr_popped_log','instr_log'):
        path=log/('core0.cubecore0.'+suffix+'.dump.gz')
        rows=[]
        for line in gzip.decompress(path.read_bytes()).decode().splitlines():
            if not re.search(r'\bMMAD\s',line):continue
            m=re.search(r'MMAD\s+dtype:(\S+) XD:(0x[0-9a-f]+) XN:(0x[0-9a-f]+) XM:(0x[0-9a-f]+) XT:(0x[0-9a-f]+)',line)
            if not m:return {'status':'failed','reason':'unparsed MMAD'}
            xt=int(m[5],16)
            rows.append({'dtype':m[1],'m':xt&4095,'k':(xt>>12)&4095,'n':(xt>>24)&4095,
                         'control':hex(xt&~((1<<36)-1)),'c_l0':m[2],'a_l0':m[3],'b_l0':m[4]})
        sequences.append(rows)
    rows=sequences[0];wanted=8 if live==24 else 16
    good=sequences[0]==sequences[1] and len(rows)==wanted
    expected_m=[24]*8 if live==24 else [64]*8+[62]*8
    good=good and [r['m'] for r in rows]==expected_m
    for i,row in enumerate(rows):
        expected_control='0x8100000000000000' if i%2==0 else '0x180000000000000'
        good=good and row['dtype']=='F32F32' and row['k']==128 and row['n']==64 and row['control']==expected_control
        if i%2:good=good and row['c_l0']==rows[i-1]['c_l0']
    return {'status':'passed' if good else 'failed','mmad_count':len(rows),'sequence':rows,
            'scope':'full popped/complete F32F32 K128 init-then-accumulate sequence; no duration sums'}


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
            binding['C_dtype']!='float32' or binding['hf32'] is not False or
            binding['B_formats']!=['ND','NZ'] or binding['C0']!=8):
        raise ValueError('fixed Matmul specification changed')
    manifest=json.loads((a.fixture.parent/'cases.json').read_text())
    if meta['source']!=cases.check_source() or manifest['source']!=meta['source'] or [r for r in manifest['cases'] if r['case']==meta['case']]!=[meta]:
        raise ValueError('fixture source/inventory differs')
    live,padded,layout=meta['live_rows'],meta['padded_rows'],meta['layout']
    if (meta['case'],live,padded,layout) not in (('qk_m24',24,32,0),('pv_m126',126,128,1)) or meta['n']!=256 or meta['k']!=256:
        raise ValueError('declared geometry only')
    files={name:(a.fixture/name).read_bytes() for name in meta['files']}
    for name,data in files.items():
        record=meta['files'][name]
        if len(data)!=record['bytes'] or sha(data)!=record['sha256']:raise ValueError('fixture changed '+name)
    if list(map(int,files['shape.txt'].split()))!=[live,padded,layout]:raise ValueError('shape file differs')
    if meta['c0']!=8 or meta['b_transpose']!=(layout==0):raise ValueError('NZ geometry/transpose changed')
    if cases.unpack_nz(files['b_nz.bin'])!=files['b_nd.bin'] or cases.pack_nz(files['b_nd.bin'])!=files['b_nz.bin']:
        raise ValueError('B full raw-bit roundtrip differs')
    mother=cases.MOTHER/meta['case'];old=json.loads((mother/'fixture.json').read_text())
    if sha((mother/'fixture.json').read_bytes())!=meta['mother_fixture_sha256'] or sha((mother/'fixture.json').read_bytes())!=cases.CASE_PINS[meta['case']]:
        raise ValueError('fixed mother fixture changed')
    for name,mother_name in (('a.bin','a_fp32.bin'),('b_nd.bin','b_fp32.bin'),('expected_output.bin','expected_output.bin')):
        record=old['files'][mother_name]
        if sha(files[name])!=record['sha256'] or len(files[name])!=record['bytes']:
            raise ValueError('exact mother input/golden changed')
    if files['a.bin'][live*256*4:]!=bytes((padded-live)*256*4):raise ValueError('input padding not zero')
    golden=files['expected_output.bin']
    inputs={'a':files['a.bin'],'b':files['b_'+a.arm+'.bin']}
    arm=0 if a.arm=='nd' else 1
    if a.library.name!='liboscar_fp32_nz_b_matmul.so':raise ValueError('explicit NZ-B library required')
    (out/'expected_output.bin').write_bytes(golden)
    report={'arm':a.arm,'case':meta['case'],'live_rows':live,'padded_rows':padded,'layout':layout,'status':'running',
        'scope':'FP32 B ND versus host-prepacked NZ; packing excluded; upper bound only','library_sha256':sha(a.library.read_bytes()),
        'binding':binding,'binding_sha256':sha(a.binding.read_bytes()),'fixture_sha256':sha((a.fixture/'fixture.json').read_bytes()),
        'input_sha256':{n:sha(data) for n,data in inputs.items()},
        'mother_sha256':{n:old['files'][n+'_fp32.bin']['sha256'] for n in ('a','b')},
        'fixture_source':meta['source'],'B_format':a.arm.upper(),'prepack_in_timing':False,
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
    launch=getattr(library,'aclrtlaunch_oscar_fp32_'+a.arm+'_b_matmul_kernel')
    launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*4+[C.c_int64]*2;launch.restype=C.c_uint32
    initial={'a':inputs['a'],'b':inputs['b'],'c':b'\x85'*(padded*256*4),'status':b'\x85'*128};owners={};ptrs={};guard=b'\xa5'*64
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
    status=list(struct.unpack('<32i',actual['status']))
    wanted=[1,arm,live,layout,4,64,64,128,128,256,256]
    if status[:11]!=wanted or not 0<status[11]<=512*1024 or status[15]!=0 or status[25]!=8:failures.append('completion')
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
    instructions=instruction_spec(log,live)
    if instructions['status']!='passed':failures.append('MMAD_K128_sequence')
    report.update(status='passed' if not failures else 'failed',profiles=profiles,traces=traces,instructions=instructions,
                  task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None,
                  instruction_analysis='complete dumps saved; dtype/MMAD inspection separate',grid=1,N=256,K=256,
                  R_scope='prepacked B removes conversion inside Matmul; excludes publisher/packing cost')
    save(out/'report.json',report)
    print(json.dumps({'arm':a.arm,'status':report['status'],'task_ticks':report['task_ticks'],'failures':failures}))
    return 2 if failures else 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=('nd','nz'),required=True)
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
