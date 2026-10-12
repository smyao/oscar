"""Archive #129/#132-135/#145/#148/#154/#155: complete two-pass ownership budget.

Host source formulas, byte-checked against real Q336 prepare output. Q1924 /
Q2048 long-history results are conditional budgets, not new device evidence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_phase_profile import analyze,partial_loop_budget,generate
from csrc.experiments.window_batch_decode.replay_camodel import sha,save_json


def prepared_formula(n,context):
    """attention_tasks.cpp, all positive slots, R1/GQA6/S1; source2 suppressed."""
    rows=[]
    for token in range(n):
        count=min(21,n-token) if token%21==0 else 0
        for source in range(3):
            first=last=0
            if count:
                if source==0:first,last=min(64,context),min(context,max(64,context+token+count-256))
                elif source==1:last=min(64,context)+context-min(context,max(64,context+token+1-256))
                else:first=last=context
            rows.append((token,count,0,first,last,0,source,source,context,0,0,0,0,0,0,0))
    return rows


def ownership(n,context):
    rows=prepared_formula(n,context);tiles=(n+20)//21
    clusters={};members=set()
    for anchor in range(0,n,84):
        if anchor+84>n:continue
        group=[rows[(anchor+j*21)*3] for j in range(4)]
        if all(r[1]==21 and r[3]<r[4] and r[4]==context and r[3:5]==group[0][3:5] for r in group):
            clusters[anchor]=group;members.update(r[0] for r in group)
    owners=[{'core':i,'history':[],'window_groups':0,'window_kv_units':0,'empty_current_groups':0} for i in range(20)]
    def work(kind,group):
        begin,end=group[0][3:5];width=end-begin
        return {'family':kind,'qbegins':[r[0] for r in group],'qcounts':[r[1] for r in group],
                'full_kv_units':width//256,'tail_kv_rows':width%256,'history_range':[begin,end]}
    for row in rows:
        if row[1]<=0:continue
        source=row[7];owner=owners[(source*tiles+row[0]//21)%20]
        if source==0 and row[0] not in members:
            owner['history'].append(work('C1_frontier' if row[4]<context else 'C1_tail_query_group',[row]))
        elif source==1:
            owner['window_groups']+=1;owner['window_kv_units']+=(row[4]-row[3]+255)//256
        elif source==2:owner['empty_current_groups']+=1
    for anchor,group in clusters.items():owners[(anchor//84)%20]['history'].append(work('C4_mature',group))
    counts=Counter(r['family'] for owner in owners for r in owner['history'])
    planned={q*3 for owner in owners for job in owner['history'] for q in job['qbegins']}
    retained=set(range(n*3))-planned
    if len(planned)!=tiles or len(planned|retained)!=n*3 or planned&retained:
        raise ValueError('healthy-shape task ID partition incomplete')
    return {'tokens':n,'context':context,'query_groups':tiles,'full_query_groups':n//21,'last_query_rows':n%21,
        'all_task_slots':n*3,'followers':n*3-tiles*3,'groups':dict(counts),'owners':owners,
        'healthy_task_id_coverage':{'planned_history_leaders':len(planned),'original_pass_retained_slots':len(retained),
            'union_slots':len(planned|retained),'overlap':0,
            'proof_scope':'host ID disposition only; not GPU execution or malformed-task validation'},
        'source':'current task-preparation and CvTaskSchedule/C4 bucket formulas, not a new device run'}


def assignment_witness(owners,c1,c4):
    """Host counterfactual assignment, NOT an implemented zero-cost GPU plan."""
    jobs=[]
    for owner in owners:
        for work in owner['history']:
            jobs.append({**work,'original_core':owner['core'],
                'full_cycle_weight':work['full_kv_units']*(c4 if work['family']=='C4_mature' else c1)})
    bins=[{'core':i,'full_history_cycles':0,'history_query_groups':0,'jobs':[]} for i in range(20)]
    for job in sorted(jobs,key=lambda r:(-r['full_cycle_weight'],r['qbegins'][0])):
        target=min(bins,key=lambda r:(r['full_history_cycles'],r['history_query_groups'],r['core']))
        target['jobs'].append(job);target['full_history_cycles']+=job['full_cycle_weight']
        target['history_query_groups']+=len(job['qcounts'])
    before=sorted(q for job in jobs for q in job['qbegins'])
    after=sorted(q for row in bins for job in row['jobs'] for q in job['qbegins'])
    if before!=after or len(set(after))!=len(after):raise ValueError('history work lost or duplicated')
    peak=max(row['full_history_cycles'] for row in bins)
    return {'assignment':'host LPT witness using measured full-tile weights; GPU algorithm/cost not established',
        'peak_full_history_cycles':peak,'critical_cores':[r['core'] for r in bins if r['full_history_cycles']==peak],
        'history_leader_set_preserved':True,'owners':bins,
        'planner_cost':'P unknown and strictly charged separately; not zero-cost evidence'}


def evaluate(n,context,c1,c4,new_c4):
    result=ownership(n,context)
    for owner in result['owners']:
        groups=Counter(work['family'] for work in owner['history'])
        c1_units=sum(work['full_kv_units'] for work in owner['history'] if work['family']!='C4_mature')
        c4_units=sum(work['full_kv_units'] for work in owner['history'] if work['family']=='C4_mature')
        old=c1_units*c1+c4_units*c4;new=c1_units*c1+c4_units*new_c4
        owner.update(group_counts=dict(groups),history_query_groups=sum(len(w['qcounts']) for w in owner['history']),
            full_history_units={'C1':c1_units,'C4_shared':c4_units},
            old_history_full_cycles=old,candidate_history_full_cycles=new,
            common_unknown_cost_ceiling_for_local_1p5=2*old-3*new,
            partial_query_groups=[w for w in owner['history'] if any(q!=21 for q in w['qcounts'])],
            partial_kv_units=[{'family':w['family'],'qbegins':w['qbegins'],'rows':w['tail_kv_rows']}
                              for w in owner['history'] if w['tail_kv_rows']])
    old=max(owner['old_history_full_cycles'] for owner in result['owners'])
    new=max(owner['candidate_history_full_cycles'] for owner in result['owners'])
    rebalanced=assignment_witness(result['owners'],c1,c4)
    combined=assignment_witness(result['owners'],c1,new_c4)
    scenarios=[]
    for name,peak,critical,placement in (
        ('pipeline_only_current_owners',new,[r['core'] for r in result['owners'] if r['candidate_history_full_cycles']==new],None),
        ('rebalance_only_old_C4',rebalanced['peak_full_history_cycles'],rebalanced['critical_cores'],rebalanced),
        ('pipeline_plus_rebalance',combined['peak_full_history_cycles'],combined['critical_cores'],combined)):
        scenarios.append({'scenario':name,'denominator':'same unchanged current-code critical-core history budget',
            'old_full_history_cycles':old,'new_full_history_cycles':peak,'history_proxy_ratio':old/peak,
            'new_critical_cores':critical,'common_F_plus_3P_ceiling_for_1p5':2*old-3*peak,
            'max_P_if_F_zero_for_1p5':(2*old-3*peak)/3,'assignment_witness':placement})
    result.update(old_history_proxy_critical_cores=[r['core'] for r in result['owners'] if r['old_history_full_cycles']==old],
        candidate_history_proxy_critical_cores=[r['core'] for r in result['owners'] if r['candidate_history_full_cycles']==new],
        history_full_only_ratio=old/new,common_uniform_unknown_cost_ceiling_for_1p5=2*old-3*new,
        all_full_tile_proxy_cycles={'old':old,'candidate':new},
        whole_kernel_formula='old=max_i(A_i+F_old_i+T_old_i); candidate=max_i(B_i+F_new_i+T_new_i)',
        unknown_fixed='F: query loading, task metadata/followers, exact window, empty current, initial state, final rotation/normalize/writeback and startup/drain',
        unknown_tail='T: every partial KV unit; plus mismatch of measured M126 proxy for last M66/M78 query group',
        same_fixed_scenario='Uniform common F (including tail proxy) plus candidate-only planner P: 1.5x requires F+3P <= 2*current_max(A)-3*new_max(B); baseline owners never prebalanced',
        scenarios=scenarios,
        actual_whole_kernel_speedup='not_established; long-context same-symbol device execution absent')
    return result


def run(directory):
    closed=partial_loop_budget.run(directory)
    actual=(directory/'normal/task_witness.bin').read_bytes()
    predicted=b''.join(struct.pack('<16q',*row) for row in prepared_formula(336,769))
    if predicted!=actual:raise ValueError('source formula fails real Q336 prepared tasks byte comparison')
    path=directory/'profile-trace/instruction_samples/core3.cubecore0.instr_log.dump.markers.json'
    tree=analyze.closed_tree(json.loads(path.read_text())['events'])
    c1_task=next(node for node in tree['children'] if node['source']==0 and node['field']==3)
    ack=sorted((node for node in analyze.walk(c1_task) if node['field']==17),key=lambda n:n['start'])
    if len(ack)!=3:raise ValueError('actual C1 full/full/tail loop changed')
    c1=(ack[1]['start']+ack[1]['inclusive'])-(ack[0]['start']+ack[0]['inclusive'])
    c4=closed['rounds'][1]['observed']['comparison_round_cycles']
    new_c4=closed['rounds'][1]['models'][1]['with_idle_gap_next_kv_period']
    cases=[evaluate(n,context,c1,c4,new_c4) for n in (2048,1924) for context in (20000,65100)]
    return {'status':'conditional_complete_partition_work_budget','whole_marker_gate':'failed_unchanged',
        'actual_metadata_crosscheck':{'Q336_C769_tasks_byte_equal':True,'sha256':sha(actual)},
        'source_sha256':{key:value for key,value in generate.PINS.items() if key in (
            'csrc/kernels/attention_tasks.cpp','csrc/kernels/attention_cv_window_range_cluster4.cpp','csrc/include/oscar_cv_schedule.h')},
        'coefficient_evidence':{'C1_full256_ack_period':c1,'C4_full256_ack_period':c4,
            'proposed_C4_full256_pipeline_prefetch_budget':new_c4,'profile_case':'Q336/R1/C769/20core',
            'transfer_to_long_history':'assumed stationary full256 service; unverified',
            'no_atomic_gain_assumed':True},
        'N1924_scope':'31 q4 prefix is a separate internal launch. Long suffix resets qstarts to zero; cv_shape_tokens=2048 retains S1, task tensor is N1924.',
        'cases':cases,'unverified':['actual native runtime budget=2048','new Q1924/Q2048 task export',
            'long-history service coefficients','partial-query and partial-KV latency','complete fixed costs','kernel performance','K32/E2E'],
        'not_included':['native current FIA','31 q4 prefix kernel','other attention wrapper stages','GDN/model/communication'],
        'rebalance_proof_scope':'healthy single-request positive-slot shape only; history leader set preserved, other 2Q leaders/followers remain outside changed history assignment',
        'GPU_work_plan_requirements':['classify from actual 3Q device task rows, never host-fabricated schedule',
            'C4 membership reuses all original same-domain/qcount/error predicates; group members written exactly once',
            'all qcount=0 followers still write both status lanes through original semantics',
            'all invalid/nonmature/tail tasks retain original validation/PublishEmpty/PublishStatus behavior; not dropped',
            'source1 and suppressed source2 still execute with canonical output/task IDs',
            'bounded preallocated plan, on-stream producer completion and graph replay identity; no host item readback',
            'measure planner/compaction/balancing/extra metadata traffic P, and count changes to per-core fixed F',
            'clusterStats owner expectations must follow new assignment; output/LSE/status/oracle gates remain frozen'],
        'interpretation':'full C4-symbol per-core work accounting, not measured complete-kernel acceleration or minutes'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.directory);save_json(a.output,result)
    print(json.dumps({'status':result['status'],'coefficients':result['coefficient_evidence'],
        'cases':[{k:row[k] for k in ('tokens','context','groups','old_history_proxy_critical_cores',
            'candidate_history_proxy_critical_cores','history_full_only_ratio','common_uniform_unknown_cost_ceiling_for_1p5')} for row in result['cases']]}))
