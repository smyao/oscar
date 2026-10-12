"""Archive #132-135/#145/#148/#154/#155: closed per-actor C4 phase attribution.

Requires the existing complete-arm and <=5% perturbation gates. No device
launch or ratios without complete markers; never sum actors as wall time.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_phase_profile import compare,fixture,markers
from csrc.experiments.window_batch_decode.replay_camodel import sha,save_json

WAIT_FIELDS={9,11,13,15,17,22}
VARIABLE_FIELDS={5,6,7,8,9,10,11,12,13,14,15,16,17,31}
FIXED_FIELDS={4,18,19,21,22}
LABELS={10:'softmax_remaining',19:'finalize_remaining',3:'task_body_unattributed'}


def task_order(data):
    """Exact two-pass order from already-validated real prepared task rows."""
    rows=list(struct.iter_unpack('<16q',data));ledger=fixture.task_ledger(rows)
    n=len(rows)//3;tiles=n//21;clusters={a:(b,e) for a,b,e in ledger['mature_clusters']}
    members={a+j*21 for a in clusters for j in range(4)}
    result=[]
    def item(tasks,family):
        units=(tasks[0][4]-tasks[0][3]+255)//256
        return {'family':family,'source':tasks[0][7],'qbegins':[r[0] for r in tasks],
            'members':len(tasks),'kv_units':units,'member_kv_units':units*len(tasks),
            'finite_reduce_counts':[sum(markers.finite_reduce_count(r,lane) for r in tasks) for lane in (0,1)]}
    for core in range(20):
        ordered=[]
        for wid in range(core,tiles*3,20):
            kind=wid//tiles;begin=(wid%tiles)*21
            for token in range(begin,min(n,begin+21)):
                row=rows[token*3+kind]
                if row[1]<=0 or (kind==0 and token in members):continue
                ordered.append(item([row],'C1'))
        for bucket in range(core,n//84,20):
            for token in range(bucket*84,(bucket+1)*84):
                if token in clusters:ordered.append(item([rows[(token+j*21)*3] for j in range(4)],'C4_mature'))
        result.append(ordered)
    for core,ordered in enumerate(result):
        owner=ledger['owners'][core]
        if (sum(r['family']=='C4_mature' for r in ordered)!=owner['clusters'] or
                sum(r['family']=='C1' for r in ordered)!=owner['c1_tasks'] or
                any(sum(r['kv_units'] for r in ordered if r['source']==s)!=owner[name]
                    for s,name in enumerate(('history','window','current')))):
            raise ValueError('actual task ledger/order disagreement')
    return result


def closed_tree(events):
    validator=markers.Intervals();stack=[];root=None
    for event in events:
        if event['field'] not in markers.MARKED_FIELDS:raise ValueError('unknown C4 phase')
        validator.accept(event)
        if not event['end']:
            node={'source':event['source'],'field':event['field'],'start':event['cycle'],'children':[]}
            if stack:stack[-1]['children'].append(node)
            else:root=node
            stack.append(node)
        else:
            node=stack.pop();node['inclusive']=event['cycle']-node['start']
            node['exclusive']=node['inclusive']-sum(c['inclusive'] for c in node['children'])
    validator.finish()
    return root


def walk(node):
    yield node
    for child in node['children']:yield from walk(child)


def attribute(tree,ordered,engine):
    tasks=[node for node in tree['children'] if node['field']==3]
    if len(tasks)!=len(ordered):raise ValueError('closed task span count differs from real schedule')
    groups={}
    for task,work in zip(tasks,ordered):
        source=work['source'];members=work['members'];units=work['kv_units']
        if task['source']!=source:raise ValueError('task span source/order drift')
        nodes=list(walk(task));counts=Counter(node['field'] for node in nodes)
        expect_finite=work['finite_reduce_counts'][engine-1] if engine else 0
        state_copies=members*(units+1) if engine and members==4 else 0
        if (counts[31]!=expect_finite or counts[24]!=state_copies or counts[25]!=state_copies or
                counts[4]!=(members if engine else 0)):
            raise ValueError('task finite/query/spill/restore count does not match real member work')
        key=f'{source}:{work["family"]}'
        group=groups.setdefault(key,{'source':('history','window','current')[source],'family':work['family'],
            'tasks':0,'members':0,'kv_units':0,'member_kv_units':0,'task_span_cycles':0,
            'fixed_exclusive_cycles':0,'variable_exclusive_cycles':0,'unattributed_exclusive_cycles':0,
            'wait_exclusive_cycles':0,'phase_exclusive_cycles':{},'phase_counts':{}})
        group['tasks']+=1;group['members']+=members;group['kv_units']+=units
        group['member_kv_units']+=work['member_kv_units'];group['task_span_cycles']+=task['inclusive']
        state_seen=Counter()
        for node in nodes:
            field=node['field'];label=LABELS.get(field,markers.FIELDS[field]);duration=node['exclusive']
            group['phase_exclusive_cycles'][label]=group['phase_exclusive_cycles'].get(label,0)+duration
            group['phase_counts'][label]=group['phase_counts'].get(label,0)+1
            if field in WAIT_FIELDS:group['wait_exclusive_cycles']+=duration
            category='unattributed'
            if field in FIXED_FIELDS:category='fixed'
            elif field in VARIABLE_FIELDS:category='variable'
            elif field in (24,25):
                state_seen[field]+=1
                # Four initial spills and four final restores are fixed.
                fixed=(field==24 and state_seen[field]<=members) or (field==25 and state_seen[field]>members*units)
                category='fixed' if fixed else 'variable'
            group[category+'_exclusive_cycles']+=duration
    for group in groups.values():
        total=group['task_span_cycles']
        if sum(group[name+'_exclusive_cycles'] for name in ('fixed','variable','unattributed'))!=total:
            raise ValueError('exclusive classification does not conserve task span')
        finite=group['phase_exclusive_cycles'].get('finite_reduce',0)
        group.update(fixed_cycles_per_task=group['fixed_exclusive_cycles']/group['tasks'],
            variable_cycles_per_shared_kv=group['variable_exclusive_cycles']/group['kv_units'] if group['kv_units'] else None,
            finite_reduce_fraction_of_this_actor_family=finite/total if total else None,
            ideal_finite_removal_local_speedup_bound=total/(total-finite) if total>finite else None)
    return list(groups.values())


def analyze(normal,profile,normal_control,profile_control):
    paired=compare.compare(normal,profile,normal_control,profile_control)
    if paired['perturbation_gate']!='passed':raise ValueError('instrumentation perturbation gate failed')
    report=json.loads((profile/'report.json').read_text())
    if not report['marker_collection_requested']:raise ValueError('complete actor markers were not collected')
    data=(profile/'task_witness.bin').read_bytes();ordered=task_order(data);actors=[]
    names=markers.expected_counts(report,data,(profile/'profile.bin').read_bytes())
    files={row['source']:row for row in report['phase_marker_files']}
    for core in range(20):
        for engine,name in enumerate(markers.ENGINES):
            record=files[f'core{core}.{name}.instr_log.dump'];evidence=json.loads((profile/record['path']).read_text())
            tree=closed_tree(evidence['events']);actual=Counter((node['source'],node['field']) for node in walk(tree))
            if actual!=names[record['source']]:raise ValueError('actor marker counts changed')
            groups=attribute(tree,ordered[core],engine)
            actors.append({'core':core,'engine':name,'compute_span_cycles':tree['inclusive'],'groups':groups,
                'outside_task_exclusive_cycles':tree['inclusive']-sum(g['task_span_cycles'] for g in groups)})
    return {'status':'completed_per_actor_C4_attribution','case':report['case'],'paired_gate':paired,
        'task_witness_sha256':sha(data),'actors':actors,
        'scope':'profile same-actor exclusive cycles, not normal wall or NPU; do not sum actors',
        'fixed_variable_rule':'query/finalize/rotation fixed; first member spills and last restores fixed; task-body residue unattributed',
        'finite_bound_scope':'ideal complete removal on this actor/family only, not achievable speedup or kernel/E2E claim',
        'npu':'not_run','whole_service':'not_run'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--normal',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--normal-control',type=Path,required=True);p.add_argument('--profile-control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:result=analyze(a.normal,a.profile,a.normal_control,a.profile_control)
    except (KeyError,OSError,ValueError) as error:
        save_json(a.output,{'status':'failed','reason':str(error),'phase_ratios':None});raise
    save_json(a.output,result);print(json.dumps({'status':result['status'],'case':result['case'],'actors':60,'output':str(a.output)}))
