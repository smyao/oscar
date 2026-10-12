"""Archive #94/#126/#140/#145/#154 and D.4: FP32 SDK versus K-major same-shape upper-bound experiment.

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
from csrc.experiments.fp32_kmajor_matmul import fixture as cases
def sha(data):return hashlib.sha256(data).hexdigest()
def save(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def records(node):
    if isinstance(node,dict):
        if {'start_cycle','end_cycle','duration'}<=node.keys():yield node
        else:
            for child in node.values():yield from records(child)


def instruction_spec(log,live,arm):
    sequences=[]
    for suffix in ('instr_popped_log','instr_log'):
        rows=[]
        for line in gzip.decompress((log/('core0.cubecore0.'+suffix+'.dump.gz')).read_bytes()).decode().splitlines():
            if not re.search(r'\bMMAD\s',line):continue
            m=re.search(r'MMAD\s+dtype:(\S+) XD:(0x[0-9a-f]+) XN:(0x[0-9a-f]+) XM:(0x[0-9a-f]+) XT:(0x[0-9a-f]+)',line)
            if not m:return {'status':'failed','reason':'unparsed MMAD'}
            xt=int(m[5],16)
            rows.append({'dtype':m[1],'m':xt&4095,'k':(xt>>12)&4095,'n':(xt>>24)&4095,
                         'control':hex(xt&~((1<<36)-1)),'c_l0':m[2],'a_l0':m[3],'b_l0':m[4]})
        sequences.append(rows)
    rows=sequences[0];blocks=(live+63)//64
    good=rows==sequences[1] and len(rows)==8*blocks
    expected=[]
    if arm=='sdk':
        for block in range(blocks):
            for n in range(4):
                for k in range(2):
                    expected.append((min(64,live-block*64),k,0,k*32768,k*32768))
    else:
        for k in range(2):
            for n in range(4):
                for block in range(blocks):
                    expected.append((min(64,live-block*64),k,(n*2+block)*16384,block*32768,(n%2)*32768))
    for row,(m,k,c,a,b) in zip(rows,expected):
        ctrl=('0x8100000000000000' if k==0 else '0x180000000000000') if arm=='sdk' else ('0x8000000000000000' if k==0 else '0x0')
        good=good and row['dtype']=='F32F32' and row['m']==m and row['k']==128 and row['n']==64 and row['control']==ctrl
        good=good and [int(row[x],16) for x in ('c_l0','a_l0','b_l0')]==[c,a,b]
    return {'status':'passed' if good else 'failed','mmad_count':len(rows),'sequence':rows,
            'scope':'each logical output keeps M24 or M64/M62,N64,K0-init then K128-acc; global scheduling order differs'}


def oracle_gate(actual,golden,live,exact):
    size=live*256*4;tail=actual[size:]==golden[size:]
    if len(actual)!=len(golden):return {'passed':False,'reason':'size'}
    if exact:return {'passed':actual==golden,'tail_poison':tail,'byte_differences':sum(x!=y for x,y in zip(actual,golden))}
    a=struct.unpack('<'+str(size//4)+'f',actual[:size]);b=struct.unpack('<'+str(size//4)+'f',golden[:size])
    bad=sum(not math.isfinite(x) or not math.isfinite(y) or abs(x-y)>0.005+0.005*abs(y) for x,y in zip(a,b))
    return {'passed':bad==0 and tail,'tail_poison':tail,'outside_frozen_tolerance':bad,'atol':0.005,'rtol':0.005}


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
    for key,file in [('kernel_sha256',Path(__file__).parent/'kernel.cpp'),('common_sha256',ROOT/'csrc/kernels/oscar_common.h'),('baseline_sha256',ROOT/'csrc/kernels/attention_cv_decode_bundle.cpp'),('mm_api_sha256',ROOT/'artifacts/cann_sdk_cube_public_20261010/include/basic_api/kernel_struct_mm.h'),('fix_api_sha256',ROOT/'artifacts/cann_sdk_cube_public_20261010/include/basic_api/kernel_struct_fixpipe.h')]:
        if sha(file.read_bytes())!=binding[key]:raise ValueError('build source differs '+key)
    if (binding['basic']!=[64,64,128] or binding['single']!=[128,256,256] or
            binding['C_dtype']!='float32' or binding['hf32'] is not False or
            binding['A_formats']!=['NZ','NZ'] or binding['B_formats']!=['NZ','NZ'] or binding['C0']!=8 or
            binding['org_m']!='physical_padded_rows' or binding['single_m']!='live_rows' or
            binding['M_blocks']!='24_or_64_62' or binding['K_order_per_output']!=[0,128] or binding['loop_order']!=['SDK','K_N_M']):
        raise ValueError('fixed Matmul specification changed')
    manifest=json.loads((a.fixture.parent/'cases.json').read_text())
    if meta['source']!=cases.check_source() or manifest['source']!=meta['source'] or [r for r in manifest['cases'] if r['case']==meta['case']]!=[meta]:
        raise ValueError('fixture source/inventory differs')
    live,padded,layout=meta['live_rows'],meta['padded_rows'],meta['layout']
    if (meta['case'],live,padded,layout) not in (('qk_m24',24,32,0),('pv_m126',126,128,1),('cancel_qk_m24',24,32,0)) or meta['n']!=256 or meta['k']!=256:
        raise ValueError('declared geometry only')
    files={name:(a.fixture/name).read_bytes() for name in meta['files']}
    for name,data in files.items():
        record=meta['files'][name]
        if len(data)!=record['bytes'] or sha(data)!=record['sha256']:raise ValueError('fixture changed '+name)
    if list(map(int,files['shape.txt'].split()))!=[live,padded,layout]:raise ValueError('shape file differs')
    if meta['c0']!=8 or meta['b_transpose']!=(layout==0):raise ValueError('NZ geometry/transpose changed')
    if (meta['a_physical_rows'],meta['org_m'],meta['single_m'])!=(padded,{'sdk':padded,'kmajor':padded},live):raise ValueError('physical A stride/live M changed')
    a_nd=cases.unpack_nz(files['a_nz.bin'],padded);b_nd=cases.unpack_nz(files['b_nz.bin'])
    if cases.pack_nz(a_nd,padded)!=files['a_nz.bin'] or cases.pack_nz(b_nd)!=files['b_nz.bin']:
        raise ValueError('A/B full raw-bit roundtrip differs')
    origin=ROOT/meta['origin_directory'];origin_meta=json.loads((origin/'fixture.json').read_text())
    if sha((origin/'fixture.json').read_bytes())!=meta['origin_fixture_sha256']:raise ValueError('origin fixture changed')
    for item in meta['origin_evidence'].values():
        raw=(ROOT/item['path']).read_bytes()
        if len(raw)!=item['bytes'] or sha(raw)!=item['sha256']:raise ValueError('origin evidence changed')
    exact=meta['exact_integer_oracle']
    if exact!=(meta['case']!='cancel_qk_m24') or meta['independent_oracle_tolerance']!={'atol':0.005,'rtol':0.005}:raise ValueError('oracle gate changed')
    old_a='a_nd.bin' if exact else 'a.bin';old_b='b_nd.bin' if exact else 'b.bin'
    if a_nd!=(origin/old_a).read_bytes() or b_nd!=(origin/old_b).read_bytes():raise ValueError('origin operands changed')
    origin_golden=(origin/('expected_output.bin' if exact else 'expected.bin')).read_bytes()
    if files['expected_output.bin'][:live*256*4]!=origin_golden[:live*256*4]:raise ValueError('origin live golden changed')
    if a_nd[live*256*4:]!=bytes((padded-live)*256*4):raise ValueError('input padding not zero')
    golden=files['expected_output.bin']
    inputs={'a':files['a_nz.bin'],'b':files['b_nz.bin']}
    arm=0 if a.arm=='sdk' else 1
    if a.library.name!='liboscar_fp32_kmajor_matmul.so':raise ValueError('explicit K-major library required')
    (out/'expected_output.bin').write_bytes(golden)
    report={'arm':a.arm,'case':meta['case'],'live_rows':live,'padded_rows':padded,'layout':layout,'status':'running',
        'scope':'FP32 same NZ operands: SDK versus explicit K-major; packing excluded; upper bound only','library_sha256':sha(a.library.read_bytes()),
        'binding':binding,'binding_sha256':sha(a.binding.read_bytes()),'fixture_sha256':sha((a.fixture/'fixture.json').read_bytes()),
        'input_sha256':{n:sha(data) for n,data in inputs.items()},
        'origin_fixture_sha256':meta['origin_fixture_sha256'],'exact_integer_oracle':exact,
        'fixture_source':meta['source'],'A_format':'NZ','B_format':'NZ','org_m':padded,'single_m':live,'prepack_in_timing':False,
        'oracle_sha256':sha(golden),'math_proof':meta['math'],'simulator_config':configuration,
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
    launch=getattr(library,'aclrtlaunch_oscar_fp32_sdk_kmajor_reference_kernel' if a.arm=='sdk' else 'aclrtlaunch_oscar_fp32_kmajor_matmul_kernel')
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
    if status[:11]!=wanted or not 0<status[11]<=512*1024 or status[15]!=0 or status[25]!=8 or status[26]!=status[24] or status[27:30]!=[padded,padded,live]:failures.append('completion')
    oracle=oracle_gate(actual['c'],golden,live,exact)
    if not oracle['passed']:failures.append('frozen_independent_oracle')
    report.update(status='numeric_passed_pending_profile' if not failures else 'failed',outputs=output_records,failures=failures,oracle=oracle,tiling_fields=status)
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
    instructions=instruction_spec(log,live,a.arm)
    if instructions['status']!='passed':failures.append('MMAD_K128_sequence')
    report.update(status='passed' if not failures else 'failed',profiles=profiles,traces=traces,instructions=instructions,
                  task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None,
                  instruction_analysis='complete dumps saved; dtype/MMAD inspection separate',grid=1,N=256,K=256,
                  R_scope='same prepacked A/B, explicit K-major versus SDK; excludes publisher/packing cost')
    save(out/'report.json',report)
    print(json.dumps({'arm':a.arm,'status':report['status'],'task_ticks':report['task_ticks'],'failures':failures}))
    return 2 if failures else 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=('sdk','kmajor'),required=True)
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
