"""Archive #132-135/#145/#148/#154/#155: isolated C4 schema3/count witness.

No FIFO, runtime, kernel launch or shared decode parser mutation. Count every
real C1/C4 task and both scheduler passes before trusting completion markers.
"""
from collections import Counter
import hashlib
import re
import struct
import types

from csrc.experiments.c4_phase_profile.fixture import CONTEXT,CORES,task_ledger,declared_tokens,case_name
from csrc.experiments.window_batch_decode import replay_camodel as shared
from tools.experiments.stream_camodel_markers import Intervals

FIELDS=(*shared.PHASE_FIELDS,'state_spill','state_restore','clusters','member_kv_units',
        'state_bytes','c1_tasks','cluster_members','finite_reduce')
ENGINES=('cubecore0','veccore0','veccore1')
MARKED_FIELDS=frozenset(range(3,26))|{31}


def marker_events(lines):
    pattern=re.compile(r'\[info\]\s+\[(\d+)\].*?MOV_XD_XN\s+.*?XD:X\d+=(0x[0-9a-fA-F]+)')
    result=[]
    for line in lines:
        match=pattern.search(line)
        if not match:continue
        tag=int(match[2],16)
        if tag>>32!=0x5a77c01d:continue
        field=(tag&255)>>1
        if tag&0xfffffc00 or field not in MARKED_FIELDS:raise ValueError('unknown C4 schema3 marker')
        result.append({'cycle':int(match[1]),'source':(tag>>8)&3,'field':field,'end':bool(tag&1)})
    return result


def collect(output,log,directory,budget,limited,expected):
    # Same proven regular-file collector, private globals select the schema3
    # parser. The imported decode module and its strict schema2 remain intact.
    original=shared.collect_completion_markers
    local=types.FunctionType(original.__code__,{**original.__globals__,'marker_events':marker_events},
                             original.__name__,original.__defaults__,original.__closure__)
    return local(output,log,directory,budget,limited,expected,Intervals)


def decoded_profile(data):
    if len(data)!=CORES*3072:raise ValueError('C4 profile requires [20,3,4,32] int64')
    values=struct.unpack('<'+str(len(data)//8)+'q',data)
    if any(value<0 for value in values):raise ValueError('unwritten/negative C4 profile counter')
    actors=[]
    for core in range(CORES):
        for engine,name in enumerate(ENGINES):
            sources=[]
            for source in range(4):
                begin=((core*3+engine)*4+source)*32
                row=values[begin:begin+32]
                sources.append(dict(zip(FIELDS,row)))
            actors.append({'core':core,'engine':name,'sources':sources})
    available=all(actor['sources'][3]['compute_span']>0 for actor in actors)
    return {'schema':3,'shape':[CORES,3,4,32],'fields':FIELDS,'actors':actors,
            'timing_usable':available,'timing_status':'observed' if available else 'unavailable_zero_SYS_CNT'}


def expected_cluster_stats(tasks_bytes):
    rows=list(struct.iter_unpack('<16q',tasks_bytes));ledger=task_ledger(rows)
    expected=[[0]*8 for _ in range(CORES)];members=set()
    for anchor,begin,end in ledger['mature_clusters']:
        row=expected[(anchor//84)%CORES];units=(end-begin+255)//256
        row[0]+=1;row[1]+=4;row[3]+=units;row[4]+=3*units;row[7]+=1
        members.update(anchor+j*21 for j in range(4))
    for row in rows:
        if row[7]!=0 or row[1]<=0:continue
        owner=(row[0]//21)%CORES
        if row[0] in members:expected[owner][6]+=1
        else:expected[owner][2]+=1;expected[owner][5]+=(row[4]-row[3]+255)//256
    return expected


def finite_reduce_count(task,lane):
    """Exact Softmax any-visible branch for each real query/head and KV tile.

    Only a true `any` executes the three profiled statements. Source0 and
    window use their original two interval formulas; padded/empty rows do not
    get synthetic reductions. Fixed fixture geometry is D256/GQA6/S64/R256.
    """
    if lane not in (0,1):raise ValueError('AIV lane must be zero or one')
    qbegin,qcount,_,kvbegin,kvend,_,_,kind,context,request_begin,*_=task
    if qcount<=0:return 0
    if kind not in (0,1,2):raise ValueError('unknown source')
    sink_count=min(64,context)
    first_position=context+qbegin-request_begin
    tail_begin=min(context,max(64,first_position+1-256))
    count=0
    for start in range(kvbegin,kvend,256):
        end=min(kvend,start+256)
        for row in range(lane*64,min((lane+1)*64,qcount*6)):
            position=context+qbegin+row//6-request_begin
            cut=max(64,position+1-256)
            if kind==0:
                visible=min(end,context,cut,position+1)>max(start,64)
            elif kind==1:
                visible=(min(end,sink_count)>max(start,0) or
                    min(end,sink_count+context-tail_begin)>
                    max(start,sink_count+max(0,cut-tail_begin)))
            else:
                visible=min(end,position+1)>max(start,context)
            count+=bool(visible)
    return count


def expected_counts(report,tasks_bytes,profile_bytes):
    if (report.get('profile_schema')!=3 or report.get('kernel')!='oscar_c4_profile_kernel' or
            report.get('cores')!=CORES or report.get('task_filtering') is not False or
            report.get('source2_suppressed') is not True):
        raise ValueError('C4 schema3 whole-partition witness required')
    if hashlib.sha256(tasks_bytes).hexdigest()!=report['input_sha256']['tasks']:
        raise ValueError('prepared tasks witness differs')
    if hashlib.sha256(profile_bytes).hexdigest()!=report['outputs']['profile']['sha256']:
        raise ValueError('profile witness differs')
    rows=list(struct.iter_unpack('<16q',tasks_bytes));ledger=task_ledger(rows)
    profile=decoded_profile(profile_bytes)
    n=declared_tokens(len(rows)//3);query_tiles=n//21
    if report['attrs'][0]!=n or report['case']!=case_name(n):raise ValueError('profile/task Q differs')
    clusters={anchor:(begin,end) for anchor,begin,end in ledger['mature_clusters']}
    members={anchor+j*21 for anchor in clusters for j in range(4)}
    metadata=[0]*CORES
    c1=[[] for _ in range(CORES)];c4=[[] for _ in range(CORES)]
    # Pass one: all 3Q rows, plus four LoadTask calls in each eligible
    # IsClusterMember predicate. No metadata call is dropped for a follower.
    for index,row in enumerate(rows):
        token,kind=divmod(index,3);owner=(kind*query_tiles+token//21)%CORES
        metadata[owner]+=1
        eligible=kind==0 and row[1]==21 and row[3]<row[4] and row[4]==CONTEXT
        if eligible:metadata[owner]+=4
        if row[1]>0 and not (kind==0 and token in members):c1[owner].append(row)
    # Pass two scans every token in its request-relative 84-token bucket.
    for token in range(n):
        owner=(token//84)%CORES;metadata[owner]+=1
        if token in clusters:
            metadata[owner]+=4;c4[owner].append((token,*clusters[token]))
    result={}
    counter_fields=(0,1,2,26,27,28,29,30)
    for core in range(CORES):
        for engine,name in enumerate(ENGINES):
            counts=Counter({(3,23):1,(3,20):metadata[core]})
            counters=[[0]*32 for _ in range(4)]
            def add(source,field,value):
                counters[source][field]+=value;counters[3][field]+=value
            def phase(source,field,times=1):
                if times:counts[source,field]+=times
            def task(member_tasks):
                source,begin,end=member_tasks[0][7],member_tasks[0][3],member_tasks[0][4]
                members_count=len(member_tasks)
                cluster=members_count==4;units=(end-begin+255)//256
                member_units=units*members_count
                phase(source,3);add(source,0,1)
                add(source,1,units);add(source,2,end-begin);add(source,27,member_units)
                if cluster:add(source,26,1);add(source,30,4)
                else:add(source,29,1)
                if engine==0:
                    for field,times in ((13,units),(14,member_units),(15,member_units),
                                         (16,member_units),(17,member_units)):
                        phase(source,field,times)
                    if source==0 and units:phase(source,18,members_count);phase(source,22,2*members_count)
                else:
                    phase(source,4,members_count)
                    if source==0:
                        phase(source,5,8*units);phase(source,6,16*units)
                    else:phase(source,7,16*units)
                    phase(source,8,16*units)
                    for field,times in ((9,member_units),(10,2*member_units),(11,member_units),(12,member_units)):
                        phase(source,field,times)
                    if source==0 and units:phase(source,22,members_count)
                    phase(source,19,members_count);phase(source,21,members_count)
                    phase(source,31,sum(finite_reduce_count(row,engine-1) for row in member_tasks))
                    if cluster:
                        copies=4*(units+1)
                        phase(source,24,copies);phase(source,25,copies)
                        add(source,28,2*copies*(64*256+2*64)*4)
            for row in c1[core]:task([row])
            for anchor,begin,end in c4[core]:task([rows[(anchor+j*21)*3] for j in range(4)])
            actual=profile['actors'][core*3+engine]['sources']
            for source in range(4):
                for field in counter_fields:
                    if actual[source][FIELDS[field]]!=counters[source][field]:
                        raise ValueError(f'C4 real counter mismatch core={core} engine={engine} source={source} field={FIELDS[field]}')
            result[f'core{core}.{name}.instr_log.dump']=counts
    return result
