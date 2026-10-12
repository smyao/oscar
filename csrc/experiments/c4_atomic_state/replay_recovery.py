# Independent revision3 replay; revision1/revision2 source and evidence stay unchanged.
"""Archive #94/#125/#126/#145/#154, D.4: one real paired atomic CAModel task.

Run under the parent's durable supervisor with a predeclared deadline <=900s.
No retry, instruction trace, CPU substitution or atomic speedup inference.
The two paths share one kernel; target NPU/graph/C4 error contracts stay open.
"""
from __future__ import annotations
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
import traceback

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_atomic_state import fixture_recovery as cases
from csrc.experiments.c4_atomic_state.replay_readback import compare_outputs as compare_readback
from csrc.experiments.window_batch_decode.replay_camodel import save_json,sha,entries


class SdkFailure(RuntimeError):
    def __init__(self,call,rc):
        super().__init__(f'{call} rc={rc}');self.call=call;self.sdk_returncode=rc


def classification(word):
    magnitude=word&0x7fffffff
    if magnitude>0x7f800000:return 'nan'
    if magnitude==0x7f800000:return '-inf' if word>>31 else '+inf'
    return 'finite'


def compare_outputs(outputs,case):
    # The unchanged revision2 comparator keeps all atomic/reference/readback,
    # padding, nonfinite-class and exact original actor-status checks.
    comparison=compare_readback(outputs,case)
    before=struct.unpack('<32768I',outputs['normal_before'])
    after=struct.unpack('<32768I',outputs['normal_after']);live=126*256
    live_diff=before_padding=after_padding=before_nonfinite=after_nonfinite=0
    before_finite=after_finite=0;first=[]
    for index,(left,right) in enumerate(zip(before,after)):
        if index<live:
            live_diff+=left!=right
            finite_left=classification(left)=='finite';finite_right=classification(right)=='finite'
            before_finite+=finite_left;after_finite+=finite_right
            before_nonfinite+=not finite_left;after_nonfinite+=not finite_right
            bad=left!=right or not finite_left or not finite_right
        else:
            before_padding+=left!=0x85858585;after_padding+=right!=0x3f800000
            bad=left!=0x85858585 or right!=0x3f800000
        if bad and len(first)<8:
            first.append({'index':index,'before_bits':hex(left),'after_bits':hex(right),
                          'scope':'live' if index<live else 'padding'})
    recovery={'status':'bitwise_passed' if not any((live_diff,before_padding,after_padding,before_nonfinite,after_nonfinite)) else 'failed',
              'live_bit_differences':live_diff,'before_padding_changes':before_padding,
              'after_padding_changes':after_padding,'before_live_finite':before_finite,
              'after_live_finite':after_finite,'before_live_nonfinite':before_nonfinite,
              'after_live_nonfinite':after_nonfinite,'first_differences':first,
              'padding_initial_words':{'before':'0x85858585','after':'0x3f800000'}}
    comparison['normal_recovery']=recovery
    if recovery['status']!='bitwise_passed':comparison['status']='failed'
    return comparison


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    save_json(output/'owner.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'script':str(Path(__file__).resolve()),
                                 'output':str(output)})
    report={'revision':3,'status':'initializing','backend':'local_CANN9.1_dav2201_CAModel','phase':'inputs',
        'npu':'not_run','graph':'not_run','full_c4_error_contract':'not_run',
        'performance':'not_established; one paired kernel is not a per-path speed comparison'}
    stopped=threading.Event();thread=None;owned={};redirected=set()
    try:
        directory=args.fixture.resolve();meta=json.loads((directory/'fixture.json').read_text())
        source=cases.check_source();manifest=json.loads((directory.parent/'cases.json').read_text())
        if (meta['revision']!=3 or manifest['revision']!=3 or meta['gm_pointer_count']!=10 or
                meta['gm_pointer_order']!=list(cases.POINTERS) or
                meta['case'] not in cases.CASES or meta['shape']!=[126,128,256,256] or
                manifest['source']!=source or meta['cube_cores']!=1 or meta['aiv_actors']!=2):
            raise ValueError('fixture geometry/source differs from the paired prototype')
        if [row for row in manifest['cases'] if row['case']==meta['case']]!=[meta]:
            raise ValueError('case metadata differs from its registered manifest')
        inputs={name:(directory/(name+'.bin')).read_bytes() for name in ('p','v','alpha')}
        sizes={'p':128*256*4,'v':256*256*4,'alpha':128*256*4}
        if any(len(data)!=sizes[name] or sha(data)!=meta['input_sha256'][name] for name,data in inputs.items()):
            raise ValueError('input bytes/hash differ')
        if {name:sha(data) for name,data in inputs.items()}!=cases.V2_RANDOM_SHA:
            raise ValueError('random inputs no longer match revision2 measured bytes')
        if cases.normal_transposed_proof(inputs)!=meta['normal_transposed_proof']:
            raise ValueError('random recovery arithmetic witness differs')
        initial={}
        for name,wanted in cases.initial_outputs().items():
            record=meta['initial_outputs'][name]
            data=(directory/record['path']).read_bytes()
            if data!=wanted or len(data)!=record['bytes'] or sha(data)!=record['sha256']:
                raise ValueError('initial recovery output differs: '+name)
            initial[name]=data
        config=args.simulator_config.resolve()
        if os.environ.get('CAMODEL_CONFIG_PATH')!=str(config):raise ValueError('SIM config must precede SDK startup')
        if 'libruntime_camodel.so' not in os.environ.get('LD_PRELOAD',''):
            raise ValueError('explicit CAModel runtime preload required; no target-device substitution')
        stars=json.loads((config/'config_stars.json').read_text());detail=json.loads((config/'config.json').read_text())
        stock=json.loads(Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/simulator/dav_2201/lib/config.json').read_text())
        if stars.get('model_top')!={'sim_type':0,'num_aic':24,'num_aiv':48} or detail.get('L2CACHE')!=stock.get('L2CACHE'):
            raise ValueError('SIM topology/cache model changed')
        library=args.library.resolve();kernel=cases.KERNEL
        if library.name!=cases.LIBRARY:raise ValueError('explicit revision3 library required')
        binding=library.parent.parent/'atomic_source_binding.json'
        if json.loads(binding.read_text())!=source:
            raise ValueError('revision3 compiled source binding differs from current fixture source')
        report.update(case=meta['case'],shape=meta['shape'],cube_cores=1,aiv_actors=2,kernel=kernel,
            source=source,build_source_binding_sha256=sha(binding.read_bytes()),
            library_sha256=sha(library.read_bytes()),input_sha256={k:sha(v) for k,v in inputs.items()},
            initial_output_sha256={k:sha(v) for k,v in initial.items()},gm_pointer_order=list(cases.POINTERS),
            simulator_configuration={'stars':stars,'detail':detail,'stars_sha256':sha((config/'config_stars.json').read_bytes()),
                                     'detail_sha256':sha((config/'config.json').read_bytes())})
        save_json(output/'inputs.json',report);save_json(output/'report.json',report)
        log=output/'log_ca';log.mkdir();os.environ.update(CAMODEL_LOG_PATH=str(log),STARS_LOG_PATH=str(log))
        def quiet():
            sink=os.open('/dev/null',os.O_WRONLY)
            try:
                while not stopped.is_set():
                    for name in os.listdir('/proc/self/fd'):
                        if not name.isdecimal() or int(name)==sink:continue
                        try:
                            target=os.readlink('/proc/self/fd/'+name)
                            if target.startswith(str(log)+'/') and target.endswith('.dump'):
                                os.dup2(sink,int(name));redirected.add(target)
                        except OSError:continue
                    time.sleep(.001)
            finally:os.close(sink)
        thread=threading.Thread(target=quiet,daemon=True);thread.start()
        acl=C.CDLL('libascendcl.so',mode=C.RTLD_GLOBAL)
        def api(name,types):
            fn=getattr(acl,name);fn.argtypes=types;fn.restype=C.c_int;return fn
        def check(call,rc):
            if rc:raise SdkFailure(call,rc)
        init=api('aclInit',[C.c_char_p]);final=api('aclFinalize',[])
        setdev=api('aclrtSetDevice',[C.c_int]);reset=api('aclrtResetDevice',[C.c_int])
        alloc=api('aclrtMalloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_int]);free=api('aclrtFree',[C.c_void_p])
        copy=api('aclrtMemcpy',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int])
        create=api('aclrtCreateStream',[C.POINTER(C.c_void_p)]);destroy=api('aclrtDestroyStream',[C.c_void_p])
        sync=api('aclrtSynchronizeStream',[C.c_void_p])
        report['phase']='sdk_init';save_json(output/'report.json',report)
        check('aclInit',init(None));check('set_device',setdev(0))
        lib=C.CDLL(str(library),mode=C.RTLD_GLOBAL);launch=getattr(lib,'aclrtlaunch_'+kernel)
        launch.argtypes=[C.c_uint32,C.c_void_p]+[C.c_void_p]*10;launch.restype=C.c_uint32
        out_sizes={'pv':128*256*4,'reference':128*256*4,'atomic':128*256*4,'status':3*64,'atomic_readback':128*256*4,
                   'normal_before':128*256*4,'normal_after':128*256*4}
        payloads={**inputs,**{name:b'\x85'*size for name,size in out_sizes.items()},**initial}
        guard=b'\xa5'*64;ptrs={};hosts=[]
        for name,data in payloads.items():
            pointer=C.c_void_p();check('allocate '+name,alloc(C.byref(pointer),len(data)+128,0));owned[name]=pointer
            ptrs[name]=C.c_void_p(pointer.value+64);payload=guard+data+guard
            host=C.create_string_buffer(payload,len(payload));hosts.append(host)
            check('upload '+name,copy(pointer,len(payload),host,len(payload),1))
        stream=C.c_void_p();check('create_stream',create(C.byref(stream)))
        report['phase']='kernel_running';save_json(output/'report.json',report)
        print('ATOMIC_RECOVERY_CAMODEL_START',meta['case'],'M126 N256 K256 Cube1 AIV2',flush=True)
        check('launch',launch(1,stream,*[ptrs[name] for name in cases.POINTERS]))
        check('synchronize',sync(stream))
        outputs={};saved={};guard_failures=[];input_changes=[]
        # Save every completed output before any input or numerical check.
        for name in (*out_sizes,*inputs):
            size=len(payloads[name]);host=C.create_string_buffer(size+128)
            check('download '+name,copy(host,size+128,owned[name],size+128,2));data=host.raw
            label=name if name in out_sizes else name+'_after'
            (output/(label+'.guarded.bin')).write_bytes(data);(output/(label+'.bin')).write_bytes(data[64:-64])
            saved[label]={'bytes':size,'sha256':sha(data[64:-64]),'guarded_sha256':sha(data),'path':label+'.bin'}
            if data[:64]!=guard or data[-64:]!=guard:guard_failures.append(name)
            if name in inputs and data[64:-64]!=inputs[name]:input_changes.append(name)
            if name in out_sizes:outputs[name]=data[64:-64]
        save_json(output/'completion.json',{**report,'status':'device_complete_outputs_saved','outputs':saved,
                                           'guard_failures':guard_failures,'input_changes':input_changes})
        comparison=compare_outputs(outputs,meta['case']);failures=[]
        if guard_failures:failures.append('guards')
        if input_changes:failures.append('input_changed')
        if comparison['status']!='bitwise_passed':failures.append('paired_comparison_or_actor_status')
        report.update(phase='finalizing',status='numeric_checked_pending_profile',outputs=saved,comparison=comparison,
                      failures=failures,guard_failures=guard_failures,input_changes=input_changes)
        save_json(output/'report.json',report)
        for pointer in owned.values():check('free',free(pointer))
        owned.clear();check('destroy_stream',destroy(stream));check('reset_device',reset(0));check('finalize',final())
        stopped.set();thread.join(timeout=2)
        profiles={}
        for kind,count in (('task',1),('aic',1),('aiv',2)):
            records=list(entries(tomllib.loads((log/f'profile_{kind}_log0.toml').read_text())))
            if len(records)!=count or any(r['duration']<=0 or r['duration']!=r['end_cycle']-r['start_cycle'] for r in records):
                failures.append(kind+'_profile_incomplete')
            profiles[kind]=records
        if len(profiles['task'])==1:
            task=profiles['task'][0]
            for kind,count in (('aic',1),('aiv',2)):
                if ({r['core_id'] for r in profiles[kind]}!=set(range(count)) or
                        any((r['task_id'],r['stream_id'])!=(task['task_id'],task['stream_id']) for r in profiles[kind])):
                    failures.append(kind+'_profile_identity')
        report.update(phase='complete',status='failed' if failures else 'passed',failures=failures,profiles=profiles,
            paired_probe_task_ticks=profiles['task'][0]['duration'] if len(profiles['task'])==1 else None,
            dump_files_redirected=len(redirected))
        save_json(output/'report.json',report)
        print('ATOMIC_RECOVERY_CAMODEL_RESULT',json.dumps({'status':report['status'],'case':meta['case'],
              'comparison':comparison,'failures':failures,'report':str(output/'report.json')}),flush=True)
        return 2 if failures else 0
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if isinstance(error,SdkFailure):report.update(failed_sdk_call=error.call,sdk_returncode=error.sdk_returncode)
        save_json(output/'report.json',report);traceback.print_exc();return 2
    finally:
        stopped.set()
        if thread is not None:thread.join(timeout=2)
        # No potentially blocking SDK cleanup after an execution failure.
        # The durable supervisor owns the process group and real return code.


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','output','library','simulator-config'):parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
