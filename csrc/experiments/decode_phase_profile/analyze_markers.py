"""Interpret complete explicit MOV phase markers, not zero SYS_CNT values.

Per-actor nesting must close exactly. Inclusive parents are never added to
children. Exclusive spans include scalar overhead and existing waits; they
are not claimed to be pure vector/Cube hardware execution occupancy.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from csrc.experiments.window_batch_decode.replay_camodel import PHASE_FIELDS
from csrc.experiments.window_batch_decode.compare_camodel import checked_arm


def analyze(events):
    stack=[];groups=defaultdict(lambda:{'count':0,'inclusive_cycles':0,'exclusive_cycles':0})
    previous=-1;outer=None
    for event in events:
        cycle=event['cycle'];key=(event['source'],event['field'])
        if cycle<previous:raise ValueError('marker cycles moved backwards')
        previous=cycle
        if not event['end']:
            if not stack and (key!=(3,23) or outer is not None):
                raise ValueError('expected exactly one outer compute span')
            stack.append({'key':key,'start':cycle,'children':0})
        else:
            if not stack or stack[-1]['key']!=key:raise ValueError('unbalanced source/phase markers')
            item=stack.pop();duration=cycle-item['start'];exclusive=duration-item['children']
            if duration<0 or exclusive<0:raise ValueError('negative phase duration')
            group=groups[key];group['count']+=1
            group['inclusive_cycles']+=duration;group['exclusive_cycles']+=exclusive
            if stack:stack[-1]['children']+=duration
            else:outer=duration
    if stack or outer is None or outer<=0:raise ValueError('incomplete actor marker trace')
    if sum(row['exclusive_cycles'] for row in groups.values())!=outer:
        raise ValueError('exclusive phase accounting does not conserve actor span')
    return {'actor_span_cycles':outer,'exclusive_accounting':'conserved',
            'phases':[{'source':('history','window','current','total')[source],
                       'phase':PHASE_FIELDS[field],**row} for (source,field),row in sorted(groups.items())]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--variant',choices=('baseline','candidate'),required=True)
    args=parser.parse_args()
    report,_=checked_arm(args.directory,args.case,args.variant)
    files=report.get('phase_marker_files',[])
    if len(files)!=report['cores']*3:raise ValueError('missing marker actors')
    arm=args.directory/(args.case+'-'+args.variant);actors=[]
    for item in files:
        evidence=json.loads((arm/item['path']).read_text())
        if not evidence['complete_for_actor'] or len(evidence['events'])!=item['event_count']:
            raise ValueError('partial marker trace')
        actors.append({'actor':evidence['source'],'source_sha256':evidence['source_sha256'],
                       **analyze(evidence['events'])})
    result={'status':'complete_marker_phase_evidence','source':'CAModel instruction completion cycles',
        'SYS_CNT':'unavailable; not used','scope':'per actor, instrumented full operator on this exact fixture',
        'task_ticks_with_instrumentation':report['task_ticks'],'actors':actors,
        'npu':'not_run','whole_service':'not_established'}
    output=arm/'marker_phase_analysis.json';output.write_text(json.dumps(result,indent=2)+'\n')
    for actor in actors:
        if actor['actor'].startswith('core0.'):
            rows=[r for r in actor['phases'] if r['source']=='history']
            print(actor['actor'],json.dumps(rows))
    print('MARKER_ANALYSIS',output)


if __name__=='__main__':main()
