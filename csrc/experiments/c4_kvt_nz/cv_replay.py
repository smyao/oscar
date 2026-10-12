"""Archive #13-22/#126/#145/#154/#155 and D.4: complete small C4 CAModel arm.

Requires complete baseline CPU and an independent publisher CPU byte gate.
Full candidate CPU is incomplete; full CAModel math must independently pass. No task filtering beyond
production current suppression; bad80/bad96 must retain original error lane.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time
import tomllib

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_kvt_nz import generate as generate_c4
from csrc.experiments.c4_b_nz.fixture import verify_rv
from csrc.experiments.window_batch_decode.replay_camodel import sha,save_json,entries,merged_oracle



def research_prerequisites(a):
    from csrc.experiments.c4_b_nz.cv_compare import load
    from csrc.experiments.c4_kvt_nz import publisher_probe
    baseline,control,hashes=load(a.prepared,a.cpu_control,'baseline')
    if 'outputs' in baseline:raise ValueError('real baseline CPU completion required')
    fixture_path=str(a.fixture.resolve())
    if baseline['_cpu_provenance']['fixture']!=fixture_path:
        raise ValueError('completed CPU baseline used another fixture')
    source,identity=publisher_probe.generated()
    if (Path(__file__).parent/'publisher_probe.cpp').read_text()!=source:
        raise ValueError('publisher CPU source changed')
    pc=json.loads(a.publisher_control.read_text())
    if (pc.get('child_returncode')!=0 or pc.get('supervisor_returncode')!=0 or
        pc.get('timed_out') is not False or pc.get('interrupt_signal') is not None or
        pc.get('cleanup_complete') is not True or pc.get('timeout_seconds')!=60):
        raise ValueError('completed bounded publisher CPU run required')
    command=pc['command']
    if (len(command)!=2 or Path(command[0]).name!='oscar_c4_kvt_nz_publisher_cpu' or
        (Path(pc['cwd'])/command[1]).resolve()!=a.publisher_cpu.resolve()):
        raise ValueError('publisher output/command identity mismatch')
    proof=publisher_probe.verify(a.publisher_cpu,identity)
    return {'baseline_CPU':'complete_independent_oracle_and_prepare',
            'baseline_control_sha256':sha(a.cpu_control.read_bytes()),
            'baseline_output_sha256':hashes,
            'publisher_CPU':proof,
            'publisher_control_sha256':sha(a.publisher_control.read_bytes()),
            'full_candidate_CPU':'not_run_for_KtVt_full_C4_B_only_120s_timeout_preserved',
            'scope':'research_only_full_CAModel_numerics_still_required'}

def run(a):
    prerequisites=research_prerequisites(a)
    source,identity=generate_c4.generate(a.variant=='candidate')
    if source!=(Path(__file__).parent/('c4_'+a.variant+'.cpp')).read_text():raise ValueError('generated C4 source drift')
    binding_path=a.library.parent.parent/'source_binding.json'
    binding=json.loads(binding_path.read_text())
    if binding!=generate_c4.build_binding():raise ValueError('configured build input binding differs')
    if a.library.name!='liboscar_c4_kvt_nz.so':raise ValueError('explicit Kt/Vt-NZ research library required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);log=out/'log_ca';log.mkdir()
    os.environ['CAMODEL_LOG_PATH']=str(log);os.environ['STARS_LOG_PATH']=str(log)
    meta=json.loads((a.fixture/'fixture.json').read_text());attrs=list(map(int,(a.fixture/'shape.txt').read_text().split()))
    context=meta['context'];bad=meta['bad_position'];receipt=json.loads((a.prepared/'report.json').read_text())
    if (attrs!=meta['attrs'] or len(attrs)!=17 or attrs[:5]!=[336,6,1,256,1] or attrs[6]!=1008 or
            attrs[13:]!=[64,256,3,1] or bad not in (-1,80,96,336) or bad>=context or context!=337 or receipt['status']!='passed' or
            receipt.get('arm')!='baseline' or receipt.get('rv_inverse') is not True or receipt.get('rv_layout')!='ND' or receipt['context']!=context or receipt['bad_position']!=bad or receipt['clusters']!=1 or
            receipt['guards'] is not True or receipt['input_immutable'] is not True or receipt['all_status_lanes'] is not True):
        raise ValueError('real complete Q336/C337 baseline CPU metadata/evidence required')
    inputs={name:(a.fixture/(name+'.bin')).read_bytes() for name in ('q','qr','ck','cv','rv','raw','table','wk','wv','tags')}
    oracle={name:(a.fixture/('expected_'+name+'.bin')).read_bytes() for name in ('output','lse')}
    for name,data in inputs.items():
        if sha(data)!=meta['input_sha256'][name]:raise ValueError('fixture input changed '+name)
    for name,data in oracle.items():
        if sha(data)!=meta['input_sha256']['expected_'+name]:raise ValueError('oracle changed '+name)
    natural_rv=inputs['rv'];packed_rv=(a.fixture/'rv_nz.bin').read_bytes()
    if sha(packed_rv)!=meta['rv_nz_sha256'] or meta['host_prepacked_KV'] is not False:raise ValueError('Rv identity/KV publisher scope changed')
    rv_proof=verify_rv(natural_rv,packed_rv)
    if a.variant=='candidate':inputs['rv']=packed_rv
    if (a.prepared/'rv_natural.bin').read_bytes()!=natural_rv or (a.prepared/'rv_input.bin').read_bytes()!=natural_rv:raise ValueError('CPU Rv evidence differs')
    (out/'rv_natural.bin').write_bytes(natural_rv);(out/'rv_input.bin').write_bytes(inputs['rv'])
    inputs['tasks']=(a.prepared/'tasks.bin').read_bytes();tasks=list(struct.iter_unpack('<16q',inputs['tasks']))
    if len(tasks)!=1008 or any(row[7]==2 and row[10]==0 and row[4]!=row[3] for row in tasks):raise ValueError('all tasks/current suppression required')
    config=Path(os.environ['CAMODEL_CONFIG_PATH']);configuration={name:json.loads((config/name).read_text()) for name in ('config.json','config_stars.json')}
    stock=json.loads(Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/config.json').read_text())
    if configuration['config.json']['L2CACHE']!=stock['L2CACHE'] or configuration['config_stars.json']['model_top']!={'sim_type':0,'num_aic':24,'num_aiv':48}:raise ValueError('hardware/cache model drift')
    report={'status':'initialized','variant':a.variant,'context':context,'attrs':attrs,'bad_position':bad,'cores':20,
        'scope':meta['scope'],'kernel':identity['symbol'],'library_sha256':sha(a.library.read_bytes()),
        'source_sha256':identity['source_sha256'],'H32_source_sha256':identity['H32_source_sha256'],'generated_sha256':sha(source.encode()),'pipeline_source_sha256':identity['pipeline_source_sha256'],
        'build_binding':binding,'build_binding_sha256':sha(binding_path.read_bytes()),
        'rv_inverse':True,'rv_layout':'NZ_C0_8' if a.variant=='candidate' else 'ND','rv_proof':rv_proof,
        'logical_input_sha256':{k:sha(natural_rv if k=='rv' else v) for k,v in inputs.items()},
        'input_sha256':{k:sha(v) for k,v in inputs.items()},'oracle_sha256':{k:sha(v) for k,v in oracle.items()},
        'CPU_prerequisites':prerequisites,'simulator_config':configuration,'NPU':'not_run','graph':'not_run','full_model':'not_run'}
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
    def api(name,types):f=getattr(acl,name);f.argtypes=types;f.restype=C.c_int;return f
    def check(name,rc):
        if rc:raise RuntimeError(name+' rc='+str(rc))
    init=api('aclInit',[C.c_char_p]);setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int]);final=api('aclFinalize',[])
    alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
    copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int])
    create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)]);sync=api('aclrtSynchronizeStream',[C.c_void_p]);destroy=api('aclrtDestroyStream',[C.c_void_p])
    check('init',init(None));check('device',setdev(0));lib=C.CDLL(str(a.library.resolve()),mode=C.RTLD_GLOBAL)
    launch=getattr(lib,'aclrtlaunch_'+identity['symbol']);launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*16+[C.c_int64]*17+[C.c_float];launch.restype=C.c_uint32
    sizes={'partial':336*6*3*256*4,'lse':336*6*3*4,'status':1008*2*4,'workspace':20*2494464,'cluster_stats':20*64}
    owners={};ptrs={};hosts=[];guard=b'\xa5'*64
    for name,data in {**inputs,**{name:b'\x85'*size for name,size in sizes.items()}}.items():
        ptr=C.c_void_p();check('allocate',alloc(C.byref(ptr),len(data)+128,0));owners[name]=ptr;ptrs[name]=C.c_void_p(ptr.value+64)
        payload=guard+data+guard;host=C.create_string_buffer(payload,len(payload));hosts.append(host);check('upload',copy(ptr,len(payload),host,len(payload),1))
    stream=C.c_void_p();check('stream',create(C.byref(stream)))
    order=('q','qr','ck','cv','rv','raw','table','wk','wv','tags','tasks','partial','lse','status','workspace','cluster_stats')
    check('launch',launch(20,stream,*[ptrs[k] for k in order],*attrs,.0625));check('sync',sync(stream))
    outputs={};saved={};failures=[]
    for name,size in sizes.items():
        host=C.create_string_buffer(size+128);check('download',copy(host,len(host),owners[name],len(host),2));data=host.raw
        if data[:64]!=guard or data[-64:]!=guard:failures.append(name+'_guard')
        outputs[name]=data[64:-64];(out/(name+'.bin')).write_bytes(outputs[name]);saved[name]={'bytes':size,'sha256':sha(outputs[name]),'path':name+'.bin'}
    save_json(out/'completion.json',{**report,'status':'device_complete_outputs_saved','outputs':saved,'failures':failures})
    for name,data in inputs.items():
        host=C.create_string_buffer(len(data)+128);check('input check',copy(host,len(host),owners[name],len(host),2))
        if name=='rv':(out/'rv_input.guarded.bin').write_bytes(host.raw)
        if host.raw!=guard+data+guard:failures.append(name+'_input_changed')
    status=struct.unpack('<2016i',outputs['status']);wanted=[]
    for row in tasks:
        for lane in range(2):wanted.append(3 if bad>=0 and row[1]>0 and row[7]==0 and row[3]<=bad<row[4] and ((bad-row[3])%32)//16==lane else 0)
    if list(status)!=wanted:failures.append('all_status_lane_oracle')
    stats=list(struct.iter_unpack('<8q',outputs['cluster_stats']))
    if any(any(v<0 for v in row) for row in stats) or [sum(row[i] for row in stats) for i in (0,1,6)]!=[1,4,4]:failures.append('cluster_contract')
    numerical=merged_oracle(outputs['partial'],outputs['lse'],oracle['output'],oracle['lse'],336,6,256,3) if bad<0 else {'scope':'expected_invalid_metadata; status/lane oracle required'}
    if bad<0 and (numerical['bad_output'] or numerical['bad_lse']):failures.append('independent_oracle')
    report.update(outputs=saved,oracle=numerical,failures=failures);save_json(out/'report.json',{**report,'status':'numeric_passed_pending_profile' if not failures else 'failed'})
    for ptr in owners.values():check('free',free(ptr))
    check('destroy',destroy(stream));check('reset',reset(0));check('finalize',final());stop.set();thread.join(timeout=2)
    profiles={}
    for kind,count in (('task',1),('aic',20),('aiv',40)):
        rows=list(entries(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
        if len(rows)!=count or any(r['duration']<=0 or r['duration']!=r['end_cycle']-r['start_cycle'] for r in rows):failures.append(kind+'_profile')
        profiles[kind]=rows
    if len(profiles['task'])==1:
        task=profiles['task'][0]
        for kind,count in (('aic',20),('aiv',40)):
            if {r['core_id'] for r in profiles[kind]}!=set(range(count)) or any((r['task_id'],r['stream_id'])!=(task['task_id'],task['stream_id']) for r in profiles[kind]):failures.append(kind+'_owners')
    report.update(status='failed' if failures else 'passed',profiles=profiles,task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None)
    save_json(out/'report.json',report);print(json.dumps({'variant':a.variant,'bad':bad,'status':report['status'],'task_ticks':report['task_ticks'],'failures':failures}),flush=True)
    return 2 if failures else 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('baseline','candidate'),required=True);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for flag in ('cpu-control','publisher-cpu','publisher-control'):p.add_argument('--'+flag,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
