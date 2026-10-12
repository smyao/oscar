"""Bounded FIFO collector for the pinned per-page decode-bundle MOV markers.

Archive #132-135/#145/#150 and startup D.4: observation, never kernel math.
FIFO names are witnessed in reports/decode_marker_v2_camodel_20261007:
coreN.cubecore0, coreN.veccore0, coreN.veccore1.instr_log.dump.

The supervisor must start WITHOUT SDK LD_PRELOAD. It creates/read-opens all
60 FIFOs BEFORE exec of the SDK child. No racing live-file dup2 handoff, file
truncation, SDK change, cache-model change or increased legacy dump cap.
FIFO/FILE/seek compatibility is unverified until a real run passes every gate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
MOV=re.compile(rb'\[info\]\s+\[(\d+)\].*?MOV_XD_XN\s+.*?XD:X\d+=(0x[0-9a-fA-F]+)')
ENGINES=('cubecore0','veccore0','veccore1')
MAX_LINE=65536
MAX_MARKER_BYTES=32*1024*1024  # all actors together; final conversion peak <=2x.
DRAIN_SECONDS=5


def actor_names(cores=20):
    return [f'core{core}.{engine}.instr_log.dump' for core in range(cores) for engine in ENGINES]


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(65536):digest.update(data)
    return digest.hexdigest()


class MarkerError(ValueError):pass


class Intervals:
    """Streaming equivalent of analyze_markers.analyze; bounded stack/groups."""
    def __init__(self):
        self.stack=[];self.groups={};self.previous=-1;self.outer=None;self.events=0
    def accept(self,event):
        cycle=event['cycle'];key=(event['source'],event['field'])
        if cycle<self.previous:raise MarkerError('marker cycles moved backwards')
        self.previous=cycle;self.events+=1
        if not event['end']:
            if not self.stack and (key!=(3,23) or self.outer is not None):
                raise MarkerError('expected exactly one outer compute span')
            if len(self.stack)>=64:raise MarkerError('marker nesting exceeds bounded depth')
            self.stack.append([key,cycle,0])
        else:
            if not self.stack or self.stack[-1][0]!=key:raise MarkerError('unbalanced source/phase markers')
            _key,start,children=self.stack.pop();duration=cycle-start;exclusive=duration-children
            if duration<0 or exclusive<0:raise MarkerError('negative phase duration')
            row=self.groups.setdefault(key,{'count':0,'inclusive_cycles':0,'exclusive_cycles':0})
            row['count']+=1;row['inclusive_cycles']+=duration;row['exclusive_cycles']+=exclusive
            if self.stack:self.stack[-1][2]+=duration
            else:self.outer=duration
    def finish(self):
        if self.stack or self.outer is None or self.outer<=0:raise MarkerError('incomplete actor marker trace')
        if sum(row['exclusive_cycles'] for row in self.groups.values())!=self.outer:
            raise MarkerError('exclusive phase accounting does not conserve actor span')


class Actor:
    def __init__(self,owner,name,path):
        self.owner=owner;self.name=name;self.path=path
        os.mkfifo(path,0o600)
        info=path.stat();self.identity=(info.st_dev,info.st_ino)
        self.fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
        self.raw_bytes=0;self.digest=hashlib.sha256();self.head=b'';self.tail=b'';self.pending=b''
        self.intervals=Intervals();self.errors=[];self.eof=False;self.seen_data=False
        self.written_events=0;self.event_buffer=bytearray();self.disabled=False
        self.event_path=owner.directory/(name+'.events.jsonl')
        self.event_path.touch(exist_ok=False)
        self.thread=threading.Thread(target=self.read,name='MOV-'+name,daemon=True)
    def fail(self,message):
        if message not in self.errors and len(self.errors)<16:self.errors.append(message)
    def flush(self):
        if self.event_buffer:
            with self.event_path.open('ab') as stream:stream.write(self.event_buffer)
            self.event_buffer.clear()
    def feed(self,data):
        self.raw_bytes+=len(data);self.digest.update(data)
        self.head=(self.head+data)[:4096];self.tail=(self.tail+data)[-4096:]
        if self.disabled:return
        self.pending+=data
        last=self.pending.rfind(b'\n')
        if last>=0:
            lines=self.pending[:last+1];self.pending=self.pending[last+1:]
            for line in lines.splitlines():
                if len(line)>MAX_LINE:raise MarkerError('instruction line exceeds bounded parser size')
                if b'MOV_XD_XN' not in line:continue
                match=MOV.search(line)
                if not match:continue
                tag=int(match[2],16)
                if tag>>32!=0x5a77c01d:continue
                if tag&0xfffffc00:raise MarkerError('unknown phase marker bits')
                field=(tag&255)>>1
                if field not in range(3,24):raise MarkerError('unknown phase marker field')
                event={'cycle':int(match[1]),'source':(tag>>8)&3,'field':field,'end':bool(tag&1)}
                self.intervals.accept(event)
                encoded=(json.dumps(event,separators=(',',':'))+'\n').encode()
                if not self.owner.reserve(len(encoded)):
                    raise MarkerError('total marker output budget exhausted')
                self.event_buffer.extend(encoded);self.written_events+=1
                if len(self.event_buffer)>=65536:self.flush()
        if len(self.pending)>MAX_LINE:raise MarkerError('unterminated instruction line exceeds bound')
    def read(self):
        try:
            while True:
                # A producer may leave a descendant with a writable FIFO open.
                # Continuous data must not evade the same bounded EOF drain.
                if self.owner.stopped.is_set() and time.monotonic()>=self.owner.drain_deadline:
                    self.fail('bounded EOF drain deadline exceeded');break
                select.select([self.fd],[],[],.05)
                try:data=os.read(self.fd,65536)
                except BlockingIOError:
                    if self.owner.stopped.is_set():
                        self.fail('writer remains open after producer exit');break
                    continue
                if data:
                    self.seen_data=True
                    try:self.feed(data)
                    except Exception as error:
                        self.fail(str(error));self.disabled=True;self.pending=b''
                    continue
                if self.owner.stopped.is_set():self.eof=True;break
                # Keep the reader open between possible stdio writer sessions.
                # A second actor span is rejected by the interval validator.
                time.sleep(.005)
        except Exception as error:self.fail('FIFO reader: '+repr(error))
        finally:
            try:self.flush()
            except Exception as error:self.fail('marker output: '+repr(error))
            os.close(self.fd)


class Collector:
    def __init__(self,output,*,cores=20,max_marker_bytes=MAX_MARKER_BYTES):
        self.output=Path(output).resolve();self.log=self.output/'log_ca'
        self.directory=self.output/'instruction_samples'
        if not 1<=cores<=20 or not 1<=max_marker_bytes<=MAX_MARKER_BYTES:raise MarkerError('unsupported collector bound')
        self.cores=cores;self.limit=max_marker_bytes;self.used=0;self.lock=threading.Lock();self.stopped=threading.Event()
        self.drain_deadline=float('inf');self.finished=False
        self.output.mkdir(parents=True,exist_ok=False);self.log.mkdir();self.directory.mkdir()
        self.actors={name:Actor(self,name,self.log/name) for name in actor_names(cores)}
        self.descriptor=self.output/'stream_fifo_manifest.json'
        self.descriptor.write_text(json.dumps({'output':str(self.output),'log':str(self.log),
            'directories':{str(p):[p.stat().st_dev,p.stat().st_ino] for p in (self.output,self.log)},
            'actors':{name:list(actor.identity) for name,actor in self.actors.items()}},indent=2)+'\n')
        for actor in self.actors.values():actor.thread.start()
    def reserve(self,size):
        with self.lock:
            if self.used+size>self.limit:return False
            self.used+=size;return True
    def finish(self,expected=None,*,producer_rc=0,reason=None):
        if self.finished:raise MarkerError('collector already finished')
        self.finished=True
        self.drain_deadline=time.monotonic()+DRAIN_SECONDS
        self.stopped.set()
        # Readers check the deadline even when a leaked writer never stops.
        # The extra select interval permits orderly close before serialization.
        deadline=self.drain_deadline+.2
        for actor in self.actors.values():actor.thread.join(max(0,deadline-time.monotonic()))
        files=[];failures=[]
        if expected is not None and set(expected)!=set(self.actors):
            failures.append('task/profile witness actor set differs from owned FIFOs')
        for name,actor in self.actors.items():
            if actor.thread.is_alive():actor.fail('bounded EOF drain deadline exceeded')
            if not actor.eof:actor.fail('FIFO EOF missing')
            if actor.pending:actor.fail('trailing partial instruction line')
            try:
                info=actor.path.stat()
                if not stat.S_ISFIFO(info.st_mode) or (info.st_dev,info.st_ino)!=actor.identity:
                    actor.fail('SDK replaced the owned FIFO inode')
                actor.intervals.finish()
            except Exception as error:actor.fail(str(error))
            counts={key:row['count'] for key,row in actor.intervals.groups.items()}
            if expected is not None and counts!=dict(expected.get(name,{})):
                actor.fail('phase event counts differ from real tasks/profile counters')
            if actor.written_events!=actor.intervals.events:actor.fail('marker output was truncated')
            complete=not actor.errors and expected is not None and producer_rc==0 and not reason
            metadata={'source':name,'source_sha256':actor.digest.hexdigest(),'captured_bytes':actor.raw_bytes,
                'complete_for_actor':complete,'transport_complete':not actor.errors,'eof':actor.eof,
                'event_count':actor.written_events,'errors':actor.errors,'method':'precreated_owned_fifo_stream',
                'prefix_bytes':0,'existing_file_handoff':False,'raw_trace_retained':False}
            destination=self.directory/(name+'.markers.json')
            with destination.open('w') as stream:
                stream.write('{"events":[');first=True
                with actor.event_path.open() as events:
                    for line in events:
                        if not first:stream.write(',')
                        stream.write(line.strip());first=False
                stream.write('],'+json.dumps(metadata,separators=(',',':'))[1:])
            actor.event_path.unlink()
            (self.directory/(name+'.preview.txt')).write_bytes(actor.head+b'\n[bounded raw head/tail]\n'+actor.tail)
            files.append({'path':str(destination.relative_to(self.output)),'event_count':actor.written_events,
                          'complete_for_actor':complete,'source':name,'raw_bytes':actor.raw_bytes,
                          'raw_sha256':actor.digest.hexdigest(),'errors':actor.errors})
            failures.extend(name+': '+error for error in actor.errors)
        if reason:failures.append(reason)
        if producer_rc:failures.append('producer returncode='+str(producer_rc))
        if expected is None:failures.append('real task/profile count witness unavailable')
        result={'status':'complete' if not failures else 'incomplete','cores':self.cores,'actor_count':len(files),
            'raw_bytes':sum(row['raw_bytes'] for row in files),'marker_output_bytes':self.used,
            'marker_output_limit':self.limit,'raw_retention':'none; SHA256 and bounded head/tail per actor',
            'phase_marker_files':files,'failures':failures,
            'sdk_pipe_compatibility':'only evidenced for a numerically completed run with all actors/counts/EOF; seek support is not assumed',
            'scope':'shipped per-page bundle; not dual-window/NPU/whole-service evidence'}
        (self.output/'stream_marker_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        return result


def expected_counts(report,tasks_bytes,profile_bytes):
    """Exact pinned M32 bundle schedule and scope-call counts; no cycle guess."""
    attrs=report['attrs'];cores=report['cores'];mode=report['mode']
    if ((attrs[0],attrs[1],attrs[2],attrs[3],attrs[4],attrs[6],attrs[-1],cores)!=(128,6,1,256,32,1152,3,20) or
            mode not in ('q4','q1') or report.get('kernel')!=f'oscar_bundle_profile_{mode}_kernel' or
            report.get('profile_counters') is not True or report.get('phase_profile_schema')!=2 or
            report.get('all_sources_and_tasks') is not True or report.get('task_filtering') is not False):
        raise MarkerError('collector requires the complete R32/N128/S3/20-core pinned profile arm')
    if hashlib.sha256(tasks_bytes).hexdigest()!=report['input_sha256']['tasks']:
        raise MarkerError('real task witness hash mismatch')
    if len(tasks_bytes)!=1152*128 or len(profile_bytes)!=20*2304:
        raise MarkerError('task/profile witness shape mismatch')
    if hashlib.sha256(profile_bytes).hexdigest()!=report['outputs']['profile']['sha256']:
        raise MarkerError('profile witness hash mismatch')
    tasks=list(struct.iter_unpack('<16q',tasks_bytes));values=struct.unpack('<5760q',profile_bytes)
    if any(value<0 for value in values):raise MarkerError('profile contains unwritten/negative counters')
    owner=4 if mode=='q4' else 1;tiles=(128+owner-1)//owner;result={}
    for core in range(20):
        for engine,name in enumerate(ENGINES):
            counts=Counter({(3,23):1});work=[[0,0,0] for _ in range(3)]
            for wid in range(core,tiles*9,20):
                begin=(wid%tiles)*owner;segment=wid//tiles
                for token in range(begin,min(begin+owner,128)):
                    row=tasks[token*9+segment];qcount,first,last,source=row[1],row[3],row[4],row[7]
                    counts[3,20]+=1
                    if row[10] or source not in (0,1,2) or last<first:raise MarkerError('invalid real task in valid profile run')
                    if qcount==0:continue
                    if qcount<0:
                        if engine:counts[source,21]+=1
                        continue
                    if qcount>owner:raise MarkerError('task exceeds q4/q1 owner width')
                    units=(last-first+255)//256;work[source][0]+=1;work[source][1]+=units;work[source][2]+=last-first
                    counts[source,3]+=1
                    if engine==0:
                        for field in (13,14,15,16):counts[source,field]+=units
                        if source!=0:counts[source,17]+=units
                        if source==0 and units:counts[source,18]+=1;counts[source,22]+=2
                    else:
                        for field in (4,19,21):counts[source,field]+=1
                        for field in (9,10,11,12):counts[source,field]+=units
                        if source==0 and units:
                            counts[source,5]+=2*units;counts[source,6]+=4*units;counts[source,8]+=4*units;counts[source,22]+=1
                        elif source!=0:
                            live=sum(start+sub*128+(engine-1)*64<last for start in range(first,last,256) for sub in (0,1))
                            counts[source,7]+=2*live;counts[source,8]+=2*live
            for source in range(4):
                expected=work[source] if source<3 else [sum(row[field] for row in work) for field in range(3)]
                offset=((core*3+engine)*4+source)*24
                if list(values[offset:offset+3])!=expected:raise MarkerError('profile work counters differ from actual task ownership')
            result[f'core{core}.{name}.instr_log.dump']={key:value for key,value in counts.items() if value}
    return result


class OwnedFifoOS:
    """Only this run's exact FIFO inodes bypass the existing quiet_dumps dup2."""
    def __init__(self,descriptor):self.identities={tuple(v) for v in descriptor['actors'].values()}
    def __getattr__(self,name):return getattr(os,name)
    def dup2(self,source,destination,*args,**kwargs):
        info=os.fstat(destination)
        if stat.S_ISFIFO(info.st_mode) and (info.st_dev,info.st_ino) in self.identities:return destination
        return os.dup2(source,destination,*args,**kwargs)


def option(args,name):
    if args.count(name)!=1:raise MarkerError('requires one explicit '+name)
    index=args.index(name)+1
    if index>=len(args) or args[index].startswith('--'):raise MarkerError('missing value for '+name)
    return args[index]


def worker(replayer,descriptor_path,args):
    descriptor=json.loads(Path(descriptor_path).read_text());spec=importlib.util.spec_from_file_location('oscar_stream_replayer',replayer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.os=OwnedFifoOS(descriptor)
    class PreparedPath(type(Path())):
        def mkdir(self,*positional,**kwargs):
            identity=descriptor['directories'].get(str(self.resolve()))
            if identity is not None:
                info=self.stat()
                if not self.is_dir() or [info.st_dev,info.st_ino]!=identity:raise MarkerError('precreated run directory identity changed')
                return None
            return super().mkdir(*positional,**kwargs)
    module.Path=PreparedPath
    original_save=module.save_json
    def save(path,value):
        if Path(path).name=='report.json' and 'task_ticks' in value:
            value['replayer_numerical_status']=value['status']
            if value['status']=='passed':value['status']='numeric_passed_pending_stream_markers'
        return original_save(path,value)
    module.save_json=save;sys.argv=[str(replayer),*args]
    return module.main()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--fifo-manifest',type=Path)
    parser.add_argument('--replayer',type=Path,required=True)
    parser.add_argument('--runtime-preload',type=Path)
    parser.add_argument('--timeout-seconds',type=int)
    parser.add_argument('replay_args',nargs=argparse.REMAINDER)
    a=parser.parse_args(argv);args=a.replay_args[1:] if a.replay_args[:1]==['--'] else a.replay_args
    if a.worker:return worker(a.replayer,a.fifo_manifest,args)
    if os.environ.get('LD_PRELOAD'):raise MarkerError('supervisor must start without LD_PRELOAD; pass --runtime-preload for child only')
    if not a.runtime_preload or not a.runtime_preload.is_file() or not a.timeout_seconds or not 1<=a.timeout_seconds<=21600:
        raise MarkerError('explicit SDK preload and predetermined deadline required')
    if '--profile-counters' not in args or '--require-phase-markers' in args:
        raise MarkerError('use profile counters and this collector gate, not the legacy prefix marker gate')
    if '--instruction-budget-bytes' in args and option(args,'--instruction-budget-bytes')!='0':
        raise MarkerError('legacy raw instruction budget must stay zero')
    output=Path(option(args,'--output')).resolve();collector=Collector(output)
    env=dict(os.environ,LD_PRELOAD=str(a.runtime_preload.resolve()),CAMODEL_LOG_PATH=str(collector.log),STARS_LOG_PATH=str(collector.log))
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--fifo-manifest',str(collector.descriptor),
             '--replayer',str(a.replayer.resolve()),'--',*args]
    process=None;rc=-1;reason=None;report=None;interrupted=threading.Event();deadline=time.monotonic()+a.timeout_seconds
    owner_path=output/'stream_owner.json'
    owner={'supervisor_pid':os.getpid(),'supervisor_pgid':os.getpgrp(),
           'output':str(output),'runtime_pid':None,'runtime_pgid':None,
           'runtime_command':command,'runtime_returncode':None,
           'termination':'SIGTERM forwarded to owned runtime group; 5s then SIGKILL',
           'external_cleanup_grace_seconds_minimum':12}
    def save_owner():
        temporary=owner_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(owner,indent=2)+'\n');temporary.replace(owner_path)
    save_owner()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:interrupted.set())
    try:
        process=subprocess.Popen(command,env=env,start_new_session=True)
        owner.update(runtime_pid=process.pid,runtime_pgid=process.pid);save_owner()
        while process.poll() is None:
            if interrupted.is_set() or time.monotonic()>=deadline:
                reason='interrupted' if interrupted.is_set() else 'unchanged host deadline reached'
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                break
            time.sleep(.1)
        rc=process.returncode
        report=json.loads((output/'report.json').read_text())
        if rc or report.get('replayer_numerical_status')!='passed' or report.get('failures'):
            raise MarkerError('replayer did not complete its numerical/profile gate')
        root=Path(option(args,'--root'));export=Path(option(args,'--export-root')) if '--export-root' in args else Path('artifacts/window_batch_decode_cpu_export')
        if not export.is_absolute():export=root/export
        expected=expected_counts(report,(export/option(args,'--case')/'tasks.bin').read_bytes(),(output/'profile.bin').read_bytes())
    except Exception as error:
        reason=reason or repr(error);expected=None
    finally:
        if process is not None:
            # Also reap writers left in our runtime group after its leader
            # exits. Never signal the collector/supervisor group or scan PIDs.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            if process.poll() is None:process.wait()
            rc=process.returncode
            owner['runtime_returncode']=rc;owner['runtime_group_cleanup']='SIGKILL_or_already_absent'
            save_owner()
        result=collector.finish(expected if 'expected' in locals() else None,producer_rc=rc,reason=reason)
        if report is not None:
            report['phase_marker_files']=result['phase_marker_files'];report['stream_marker_collection']='stream_marker_summary.json'
            report['status']='passed' if result['status']=='complete' else 'failed'
            if result['status']!='complete':report.setdefault('failures',[]).append('stream_marker_collection_incomplete')
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('STREAM_MARKER_RESULT',json.dumps({key:result[key] for key in ('status','actor_count','raw_bytes','marker_output_bytes','failures')}),flush=True)
    return 0 if result['status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
