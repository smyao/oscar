"""Archive #129/#140/#155: ownership counts from the actual prepared task bytes.

This is a work-count forecast, never a cycle/second forecast. Padding has
no KV weight; its execution and planner overhead are tested by full CAModel.
"""
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[3]

def analyze(directory, tile, cores=20):
    data=(directory/'tasks.bin').read_bytes()
    values=(directory/'attrs.txt').read_text().split()
    assert len(values)==19 and int(values[-1])==cores
    attrs=[int(x) for x in values[:17]]
    # attrs: N,Hq,Hkv,D,R,columns,tasks,block,blocks,prefix,strides...,sink,recent,spec,S.
    n,hq,hk,d=attrs[:4];splits=attrs[-1]
    assert n<=128 and hq==6 and hk==1 and d==256 and 1<=splits<=3
    per_token=3*splits;tiles=(n+tile-1)//tile;history=tiles*splits
    rows=list(struct.iter_unpack('<16q',data))
    assert len(rows)==n*per_token
    old=[0]*cores;fresh=[0]*cores;buckets=[[] for _ in range(128)]
    weights=[];old_ids=[[] for _ in range(cores)];new_ids=[[] for _ in range(cores)]
    for wid in range(history):
        start=(wid%tiles)*tile;split=wid//tiles;cost=0
        for token in range(start,min(start+tile,n)):
            row=rows[token*per_token+split]
            queries,first,last=row[1],row[3],row[4]
            if queries>0:
                cost+=1
                if queries>0 and first>=0 and last>=first:
                    width=last-first;cost+=min(width//256+bool(width%256),1<<24)
        weights.append(cost);old[wid%cores]+=cost;old_ids[wid%cores].append(wid)
        buckets[min(cost,127)].append(wid)
    for bucket in reversed(buckets):
        for wid in reversed(bucket):
            cost=weights[wid]
            owner=min(range(cores),key=lambda c:fresh[c]) if cost else wid%cores
            fresh[owner]+=cost;new_ids[owner].append(wid)
    selected=max(old)*100>max(fresh)*105
    if not selected:new_ids=old_ids
    # Independent coverage expansion: every complete ABI task row has exactly
    # one owner, including nonleaders, negative-slot padding, window/current.
    covered=[]
    for core in range(cores):
        for wid in new_ids[core]+[w for w in range(history,tiles*per_token) if w%cores==core]:
            start=(wid%tiles)*tile;segment=wid//tiles
            covered.extend(t*per_token+segment for t in range(start,min(start+tile,n)))
    assert sorted(covered)==list(range(len(rows)))
    assert sum(old)==sum(fresh)
    return {'case':directory.name,'tasks_sha256':hashlib.sha256(data).hexdigest(),
            'task_rows':len(rows),'all_tasks_once':True,'planned':selected,
            'per_core_work_before':old,'per_core_work_proposed':fresh,
            'maximum_before':max(old),'maximum_proposed':max(fresh),
            'idealized_maximum_ratio':max(old)/max(fresh),
            'new_fixed_workspace_bytes':4096*cores,
            'startup_task_read_bytes_per_active_lane':n*splits*128,
            'limitations':['weight1 for live-task initialization is a heuristic, not device timing',
                           'padding/fault task execution is not free and remains in the measured kernel',
                           'does not include planner execution, contention, window/current or host time',
                           'same cost units cannot be summed across q4 and q1 as cycles',
                           'full same-input CAModel and target NPU determine performance']}

if __name__=='__main__':
    base=ROOT/'artifacts/decode_batch32_tasks_20261007'
    report={'scope':'synthetic prepared task work only; no NPU or whole-run prediction',
            'cases':[analyze(base/f'{mode}_r32_n128_20to30k_s3',4 if mode=='q4' else 1)
                     for mode in ('q4','q1')]}
    out=ROOT/'reports/decode_weighted_work_20261007.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    for row in report['cases']:
        print(row['case'],row['maximum_before'],row['maximum_proposed'],row['all_tasks_once'])
