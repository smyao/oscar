"""Archive #126/#129/#148/#155, D.4: exact synthetic source0 work accounting.

This plans bounded shared FP32 rotated-history tiles; it is not a serving
route, HBM traffic measurement, or predicted NPU latency. Existing QK/PV
KV256 boundaries and query-specific validation domains must be preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Shape:
    query_tokens: int
    context_tokens: int
    cluster: int
    head_dim: int = 256
    query_heads: int = 6
    sink: int = 64
    recent: int = 256
    history_tile: int = 8192

    def validate(self):
        if (self.query_tokens <= 0 or self.context_tokens < 0 or self.cluster not in (4,16)
                or self.head_dim != 256 or self.query_heads != 6 or self.sink < 0
                or self.recent <= 0 or self.history_tile <= 0 or self.history_tile % 256):
            raise ValueError('D256 GQA6 S1 positive query and aligned bounded history tile required')


def ceildiv(x, y):
    return (x+y-1)//y


def account(shape: Shape):
    shape.validate()
    qtile=128//shape.query_heads
    first=min(shape.sink,shape.context_tokens)
    groups=[]
    # Exact prepare_attention_tasks.cpp kind0 formula, contiguous valid slots/S1.
    for begin in range(0,shape.query_tokens,qtile):
        count=min(qtile,shape.query_tokens-begin)
        end=min(shape.context_tokens,max(shape.sink,shape.context_tokens+begin+count-shape.recent))
        groups.append({'begin':begin,'count':count,'first':first,'end':end,
                       'kv256':ceildiv(end-first,256)})
    grouped=set();clusters=[]
    for start in range(0,len(groups),shape.cluster):
        selected=groups[start:start+shape.cluster]
        # Exact old C4/C16 predicate: complete aligned query groups, same
        # nonempty interval, terminal kvend=context; partial groups do not share.
        if (len(selected)==shape.cluster and all(g['count']==qtile and
                g['end']==shape.context_tokens and g['end']>first for g in selected)):
            grouped.update(range(start,start+shape.cluster))
            clusters.append(selected)
    units=sum(g['kv256'] for i,g in enumerate(groups) if i not in grouped)
    units+=sum(c[0]['kv256'] for c in clusters)
    logical_expanded_rows=sum(g['end']-first for i,g in enumerate(groups) if i not in grouped)
    logical_expanded_rows+=sum(c[0]['end']-first for c in clusters)
    old_state_bytes=(128*shape.head_dim+2*128)*4
    legacy_state=sum(shape.cluster*2*(c[0]['kv256']+1)*old_state_bytes for c in clusters)
    shared_rows=max((g['end']-first for g in groups),default=0)
    tile_count=ceildiv(shared_rows,shape.history_tile)
    boundary_state=0
    private_tails=0
    consumer_units=sum(g['kv256'] for g in groups)
    for group in groups:
        consumer_tiles=ceildiv(group['end']-first,shape.history_tile)
        active_state_bytes=group['count']*shape.query_heads*(shape.head_dim+2)*4
        boundary_state+=2*max(0,consumer_tiles-1)*active_state_bytes
        if (group['end']-first)%256:
            private_tails+=1
    return {'shape':asdict(shape),'scope':'synthetic_exact_loop_count_not_time_or_HBM',
            'query_groups':len(groups),'eligible_clusters':len(clusters),
            'legacy':{'expanded_KV256_units':units,'expanded_logical_rows':logical_expanded_rows,
                      'acc_max_sum_logical_GM_bytes':legacy_state},
            'shared':{'expanded_KV256_units':ceildiv(shared_rows,256),
                      'expanded_logical_rows':shared_rows,'producer_tiles':tile_count,
                      'acc_max_sum_boundary_GM_bytes':boundary_state,
                      'private_tail_KV256_blocks':private_tails,
                      'cache_K_plus_V_bytes':shape.history_tile*shape.head_dim*2*4,
                      'row_error_bytes':shape.history_tile*4},
            'unchanged':{'QK_calls':consumer_units,'PV_calls':consumer_units,
                         'online_softmax_KV_width':256,'quantized_cache_bits':2,
                         'GDN':'native_unchanged'},
            'row_reuse_factor':logical_expanded_rows/shared_rows if shared_rows else None,
            'precision_gate':'not_implemented_or_run','speed_gate':'not_measured'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    # First chunks are cold; continuation + 31 q4 neighbours leave 16260
    # prompt slots. These are explicit synthetic shapes, not reconstructed requests.
    scenarios=[Shape(3740,16260,4),Shape(6740,16260,4),Shape(10740,16260,16),
               Shape(16260,13740,16),Shape(16260,48840,16),Shape(16260,0,16)]
    report={'status':'structural_candidate_only','cases':[account(s) for s in scenarios],
            'mechanism':'shared bounded FP32 rotated K/V, original KV256 recurrence, raw online state across tiles',
            'missing':'actual new kernels, numerical/graph/device gates and calibrated end-to-end latency',
            'promotion_allowed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'report':str(args.output)}))
    return 0

if __name__=='__main__':raise SystemExit(main())
