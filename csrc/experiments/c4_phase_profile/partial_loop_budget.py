"""Archive #132-135/#145/#148/#154/#155: closed-loop evidence, failed trace stays failed.

Host-only scheduling hypothesis. No CANN edits, dtype change, extrapolation,
invented finalization, or promotion of incomplete actor traces to acceptance.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_phase_profile import compare,markers,analyze
from csrc.experiments.window_batch_decode.replay_camodel import sha,save_json


def mature_node(events):
    stack=[];tasks=[];previous=-1
    for event in events:
        if event['cycle']<previous:raise ValueError('cycles moved backwards')
        previous=event['cycle']
        if event['field'] not in markers.MARKED_FIELDS:raise ValueError('unknown phase')
        if not event['end']:
            node={'field':event['field'],'source':event['source'],'start':event['cycle'],'children':[]}
            if stack:stack[-1]['children'].append(node)
            if event['field']==3 and event['source']==0:tasks.append(node)
            stack.append(node)
        else:
            if not stack:raise ValueError('unmatched end')
            node=stack.pop()
            if (node['source'],node['field'])!=(event['source'],event['field']):raise ValueError('nesting mismatch')
            node['end']=event['cycle'];node['duration']=event['cycle']-node['start']
    if len(tasks)!=2:raise ValueError('expected core3 C1 then unique mature C4')
    node=tasks[1];closed=[item for item in analyze.walk(node) if 'end' in item]
    return {field:sorted((x for x in closed if x['field']==field),key=lambda x:x['start']) for field in markers.MARKED_FIELDS}


def dag(qk,pv,soft,consume,q_lag,p_lag,pv_lag,ack_lag,setup):
    durations={};edges={}
    def node(name,duration,deps):durations[name]=duration;edges[name]=deps
    cube_order=[('q',0),('q',1),('p',0),('q',2),('p',1),('q',3),('p',2),('p',3)]
    prior=None
    for kind,m in cube_order:
        name=f'{kind}{m}';deps=[] if prior is None else [(prior,0)]
        if kind=='p':
            deps += [(f's{lane}-{m}',p_lag[m]) for lane in (0,1)]
            if m:deps += [(f'c{lane}-{m-1}',ack_lag[m-1]) for lane in (0,1)]
        node(name,(qk if kind=='q' else pv)[m],deps);prior=name
    vector_order=[('s',0),('s',1),('c',0),('s',2),('c',1),('s',3),('c',2),('c',3)]
    for lane in (0,1):
        prior=None
        for kind,m in vector_order:
            name=f'{kind}{lane}-{m}';deps=[] if prior is None else [(prior,setup)]
            deps.append((f'q{m}' if kind=='s' else f'p{m}',q_lag[lane][m] if kind=='s' else pv_lag[lane][m]))
            node(name,(soft if kind=='s' else consume)[lane][m],deps);prior=name
    visiting=set()
    @lru_cache(None)
    def finish(name):
        if name in visiting:raise ValueError('pipeline DAG has cycle')
        visiting.add(name);start=max((finish(key)+lag for key,lag in edges[name]),default=0)
        visiting.remove(name);return start+durations[name]
    schedule={name:{'start':finish(name)-duration,'end':finish(name),'duration':duration} for name,duration in durations.items()}
    end=max(finish(f'c{lane}-3')+ack_lag[3] for lane in (0,1))
    return end,schedule


def prefetch_period(schedule,end,packets):
    """Put intact next-KV packets only into unused AIV gaps, no work moved."""
    result=[]
    for lane in (0,1):
        work=sorted((v for k,v in schedule.items() if k.startswith((f's{lane}-',f'c{lane}-'))),key=lambda v:v['start'])
        cursor=0;index=0;placed=[]
        for event in work+[{'start':end,'end':end}]:
            while index<len(packets[lane]) and cursor+packets[lane][index]<=event['start']:
                placed.append([index,cursor,cursor+packets[lane][index]]);cursor+=packets[lane][index];index+=1
            cursor=max(cursor,event['end'])
        cursor=max(cursor,end) if index<len(packets[lane]) else cursor
        while index<len(packets[lane]):
            placed.append([index,cursor,cursor+packets[lane][index]]);cursor+=packets[lane][index];index+=1
        result.append({'lane':lane,'next_kv_ready':placed[-1][2] if placed else 0,'packets':placed})
    return max([end]+[row['next_kv_ready'] for row in result]),result


def extract_round(cube,vectors,tile):
    ids=range(tile*4,tile*4+4);a=tile*4;lanes=[]
    qk=[cube[14][i]['duration'] for i in ids];pv=[cube[16][i]['duration'] for i in ids]
    p_lag=[max(0,cube[15][i]['end']-max(v[10][2*i+1]['end'] for v in vectors)) for i in ids]
    ack_lag=[max(0,cube[17][i]['end']-max(v[24][4+i]['end'] for v in vectors)) for i in ids]
    for v in vectors:
        restore=[v[25][i]['duration'] for i in ids];spill=[v[24][4+i]['duration'] for i in ids]
        soft=[v[10][2*i+1]['end']-v[9][i]['end']+restore[j]+spill[j] for j,i in enumerate(ids)]
        consume=[v[24][4+i]['end']-v[11][i]['end']+restore[j] for j,i in enumerate(ids)]
        packets=[v[8][tile*16+2*j+1]['end']-v[5][tile*8+j]['start'] for j in range(8)]
        q_lag=[max(0,v[9][i]['end']-cube[14][i]['end']) for i in ids]
        pv_lag=[max(0,v[11][i]['end']-cube[16][i]['end']) for i in ids]
        setup=max(v[25][i]['start']-v[24][4+i-1]['end'] for i in range(a+1,a+4))
        lanes.append(dict(soft=soft,consume=consume,packets=packets,q_lag=q_lag,pv_lag=pv_lag,setup=max(0,setup)))
    begin=min(v[5][tile*8]['start'] for v in vectors);ready=max(v[8][tile*16+15]['end'] for v in vectors)
    done=cube[17][a+3]['end'];qstart=cube[14][a]['start']
    baseline=done-cube[17][a-1]['end'] if tile else done-begin
    trials=[]
    for label,setup in (('optimistic_transition_cost',0),('observed_max_transition_cost',max(row['setup'] for row in lanes))):
        end,schedule=dag(qk,pv,[x['soft'] for x in lanes],[x['consume'] for x in lanes],
            [x['q_lag'] for x in lanes],p_lag,[x['pv_lag'] for x in lanes],ack_lag,setup)
        period,packets=prefetch_period(schedule,end,[x['packets'] for x in lanes])
        trials.append({'model':label,'transition_cycles':setup,'pipeline_after_kv_ready':end,
            'no_prefetch_round_cycles':ready-begin+end,'with_idle_gap_next_kv_period':period,
            'observed_round_over_no_prefetch':baseline/(ready-begin+end),
            'observed_round_over_prefetch_period':baseline/period if tile<2 else None,
            'next_prefetch_interpretation':'repeating full256 steady hypothesis' if tile<2 else 'no next tile; prefetch speed ratio not applicable',
            'schedule':schedule,'prefetch_schedule':packets})
    return {'tile':tile,'logical_range':[64+tile*256,min(769,64+(tile+1)*256)],
        'role':('first full256','steady full256','tail193')[tile],
        'observed':{'prepare_begin':begin,'kv_publish_end':ready,'first_qk_start':qstart,'last_ack_end':done,
            'kv_prepare_envelope':ready-begin,'after_kv_qk_ack_span':done-qstart,'round_envelope':done-begin,
            'comparison_round_cycles':baseline,'comparison_round_boundary':'previous Cube ack to current ack' if tile else 'first packed prepare to current ack',
            'cube_kv_wait':cube[13][tile]['duration'],'qk':qk,'pv':pv,'p_ready_lags':p_lag,'ack_lags':ack_lag},
        'lanes':lanes,'models':trials}


def run(directory):
    normal,_=compare.checked_arm(directory/'normal',directory/'normal-control.json','normal')
    p=directory/'profile-trace';report=json.loads((p/'report.json').read_text());complete=json.loads((p/'completion.json').read_text())
    control=json.loads((directory/'profile-trace-control.json').read_text())
    if (control['child_returncode']!=2 or control['timed_out'] or control['interrupt_signal'] is not None or
            control['cleanup_complete'] is not True or control['timeout_seconds']!=900 or
            complete['status']!='device_complete_outputs_saved' or complete['failures']):raise ValueError('partial numeric completion/deadline changed')
    summary=json.loads((directory/'partial_diagnostic_summary.json').read_text())
    if report['status']!='failed' or summary['status']!='full_marker_gate_failed_preserved':raise ValueError('preserve failed whole-trace scope')
    allowed={'incomplete_C4_markers'}|{f'core3.veccore{lane}.instr_log.dump: '+reason for lane in (0,1)
        for reason in ('partial or oversized instruction line','completion trace cap reached or prefix incomplete','incomplete actor marker trace')}
    if set(report['failures'])!=allowed:raise ValueError('failure includes more than the witnessed two capped actors')
    for name,item in report['outputs'].items():
        data=(p/item['path']).read_bytes()
        if len(data)!=item['bytes'] or sha(data)!=item['sha256'] or complete['outputs'][name]!=item:raise ValueError('output identity changed')
    if report['oracle']['bad_output'] or report['oracle']['bad_lse']:raise ValueError('numerical failure')
    for key in ('attrs','input_sha256','library_sha256','simulator_configuration','source_sha256','dependency_sha256','acceptance_sha256'):
        if report[key]!=normal[key]:raise ValueError('normal/profile identity changed')
    for key in ('partial','lse','status','cluster_stats'):
        if report['outputs'][key]['sha256']!=normal['outputs'][key]['sha256']:raise ValueError('bitwise changed')
    tasks=(p/'task_witness.bin').read_bytes()
    if sha(tasks)!=report['input_sha256']['tasks']:raise ValueError('task witness hash differs')
    markers.expected_counts(report,tasks,(p/'profile.bin').read_bytes())
    order=analyze.task_order(tasks)[3]
    if [(x['source'],x['family']) for x in order if x['source']==0]!=[(0,'C1'),(0,'C4_mature')]:raise ValueError('mature task identity changed')
    actors=[];witnesses={}
    for actor in markers.ENGINES:
        path=p/'instruction_samples'/f'core3.{actor}.instr_log.dump.markers.json'
        evidence=json.loads(path.read_text())
        witnesses[actor]={'marker_json_sha256':sha(path.read_bytes()),'retained_trace_sha256':evidence['source_sha256'],
                          'complete_for_actor':evidence['complete_for_actor']}
        actors.append(mature_node(evidence['events']))
    cube=actors[0];vectors=actors[1:]
    if any(len(cube[f])!=12 for f in (14,15,16,17)) or len(cube[13])!=3:raise ValueError('incomplete Cube loops')
    for v in vectors:
        if any(len(v[f])!=count for f,count in ((4,4),(5,24),(6,48),(8,48),(9,12),(10,24),(11,12),(12,12),(24,16))) or len(v[25])<12:
            raise ValueError('incomplete Vector loops')
    rounds=[extract_round(cube,vectors,t) for t in range(3)]
    selected_span=rounds[-1]['observed']['last_ack_end']-rounds[0]['observed']['prepare_begin']
    cube_service=sum(sum(r['observed'][field]) for r in rounds for field in ('qk','pv'))
    upper_saved=selected_span-cube_service
    chained=[]
    for index in range(2):
        cost=rounds[0]['observed']['kv_prepare_envelope'];details=[]
        for tile,round in enumerate(rounds):
            model=round['models'][index];period=model['pipeline_after_kv_ready']
            if tile<2:
                period,packets=prefetch_period(model['schedule'],period,[x['packets'] for x in rounds[tile+1]['lanes']])
                details.append({'tile':tile,'next_tile':tile+1,'period':period,'prefetch_schedule':packets})
            else:details.append({'tile':tile,'next_tile':None,'period':period})
            cost+=period
        saving=max(0,selected_span-cost)
        chained.append({'model':rounds[0]['models'][index]['model'],'selected_loop_predicted_cycles':cost,
            'selected_loop_observed_over_predicted':selected_span/cost,
            'whole_profile_fixture_ratio_if_rest_fixed':report['task_ticks']/(report['task_ticks']-saving),
            'rounds':details})
    return {'status':'partial_closed_loop_budget_only','whole_marker_gate':'failed_unchanged','rounds':rounds,
        'evidence':witnesses,'task_witness_sha256':sha(tasks),
        'three_round_chain_models':chained,
        'profile_task_ticks':report['task_ticks'],'selected_loop_span':selected_span,'measured_cube_service_sum':cube_service,
        'optimistic_whole_fixture_bound_if_all_other_costs_fixed':report['task_ticks']/(report['task_ticks']-upper_saved),
        'extra_storage_bytes_per_cube':{'second_score_P':131072,'second_KV_pair':524288},
        'extra_state_traffic_bytes_per_C4_KV':1056768,
        'assumptions':['same measured phase durations under new overlap','GM contention unchanged','two score/P slots; one PV slot ack-gated',
            'all old FP32 operations and each member KV reduction order retained','extra scaled-state spill/restore charged at measured durations',
            'next-KV intact packets use idle AIV gaps only; no overlap of shared UB users','first-tile startup and finalization unchanged; missing finalization never filled'],
        'limits':['scheduling hypothesis, not compiled implementation or accepted speedup','R1/Q336 low occupancy; no K32/64K inference',
            'recorded Matmul scopes plus measured readiness lags are model costs, not proof of hardware overlap capacity']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.directory);save_json(a.output,result)
    print(json.dumps({'status':result['status'],'steady':[{k:v for k,v in row.items() if k not in ('schedule','prefetch_schedule')}
        for row in result['rounds'][1]['models']],'whole_fixture_upper_bound':result['optimistic_whole_fixture_bound_if_all_other_costs_fixed']}))
