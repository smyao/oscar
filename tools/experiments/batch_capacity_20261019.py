"""Source-bound resident-batch accounting; archive #127/#128/#140/#148/#154/#163/#166/#167.

Read-only native planner arithmetic, current physical-page snapshots, and
actual Python CV selection. No serving mutations, request-window projection,
NPU timing, fake throughput, or claim of numerical/graph acceptance.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS

import torch

from oscar_ascend.layout import HybridPageLayout, SlotLayout, ceil_div
from oscar_ascend.lifecycle import SnapshotLayout
from oscar_ascend.runtime import WorkspaceGeometry
from oscar_ascend.ops.cv_dispatch import select_cv_op
from oscar_ascend.integration.compact_later_mtp import should_compact_rows
from oscar_ascend.integration.mixed_decode_split import plan_mixed_decode
from tools.analyze_memory_64k import reference_geometry, geometry_from_model
from tools.serving_variants import variant_config

ROOT = Path(__file__).resolve().parents[2]
GIB = 1 << 30
BATCHES = (1, 4, 8, 16, 32, 64, 128)
PLANNER = 'references/vllm/vllm/v1/core/kv_cache_utils.py'
SHAPES = 'references/vllm/vllm/model_executor/layers/mamba/mamba_utils.py'
SPECS = 'references/vllm/vllm/v1/kv_cache_interface.py'
LOG = 'reports/target_runtime_trap_20261019.txt'


def native_functions(path, names, env):
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names) or len(selected) != len(names):
        raise ValueError(f'native source selection drift: {path}')
    code = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(ROOT/path), 'exec'), env)
    return env


@dataclass(frozen=True)
class ShapeSpec:
    """Only the fields used by native grouping; not a runtime KVCacheSpec."""
    kind: str
    block_size: int
    page_size_bytes: int

    @staticmethod
    def merge(specs):
        assert specs and all(item == specs[0] for item in specs)
        return specs[0]


def planner():
    names = {'_get_kv_cache_groups_uniform_page_size', 'create_kv_cache_group_specs',
             'get_kv_cache_config_from_groups', 'get_num_blocks',
             'may_override_num_blocks', 'get_uniform_page_size', '_pool_bytes_per_block'}
    env = {'defaultdict': defaultdict, 'cdiv': ceil_div,
           'logger': NS(warning=lambda *args: None),
           'UniformTypeKVCacheSpecs': type('UniformTypeKVCacheSpecs', (), {}),
           'KVCacheGroupSpec': lambda names, spec: NS(layer_names=names, kv_cache_spec=spec),
           'KVCacheTensor': lambda **kwargs: NS(**kwargs),
           'KVCacheConfig': lambda **kwargs: NS(**kwargs)}
    return native_functions(PLANNER, names, env)


def observed_budgets():
    text = (ROOT/LOG).read_text()
    current = re.findall(r'Current KV cache memory: ([0-9.]+) GiB', text)
    suggested = re.findall(r'--kv-cache-memory=(\d+)` .*?to fit into requested memory', text)
    if len(current) != 4 or len(suggested) != 4:
        raise ValueError('expected four rank-specific memory records in source log')
    low = min(int((Decimal(x)-Decimal('.005'))*GIB) for x in current)
    high = min(int((Decimal(x)+Decimal('.005'))*GIB) for x in current)
    return {
        'source': LOG, 'lines': [1, 2, 3, 5], 'current_kv_gib_printed_by_rank': current,
        'current_oscar_budget_interval_bytes': [low, high],
        'current_budget_rounding': 'Two-decimal log rounding interval; exact runtime KV allocation absent.',
        'suggested_fit_requested_utilization_bytes_by_rank': list(map(int, suggested)),
        'scenarios': {'observed_oscar_rounding_lower': low,
                      'observed_oscar_rounding_upper': high,
                      'suggested_fit_0_9_min_rank': min(map(int, suggested))},
        'native_same_run_profiled_budget_bytes': None,
        'comparison_scope': 'Native uses the SAME supplied net KV pool budget as a conditional comparison. Native post-profile budget is not measured by this OSCAR log.',
        'do_not_double_subtract': 'The observed KV budget already follows model/workspace profiling; do not subtract OSCAR scratch or rotations a second time.',
    }


def source_record(path):
    content = (ROOT/path).read_bytes()
    record = {'path': path, 'sha256': hashlib.sha256(content).hexdigest()}
    if path.endswith('.py'):
        selected = {'_get_kv_cache_groups_uniform_page_size', 'get_kv_cache_config_from_groups',
                    'get_num_blocks', 'gated_delta_net_state_shape', 'MambaSpec', 'max_memory_usage_bytes',
                    'ensure_workspace', '_prepare_rotations', 'prepare_device_rotations', 'select_cv_op',
                    'should_compact_rows', 'attn_update_stack_num_spec_norm', 'allocate_slots',
                    'HybridPageLayout', 'SnapshotLayout', 'forward', 'verify_and_update_config',
                    'plan_mixed_decode', '_slice_metadata', 'dispatch_mixed_decode'}
        lines = {}
        for node in ast.parse(content).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected:
                lines[node.name] = node.lineno
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name in selected:
                        lines[f'{node.name}.{child.name}'] = child.lineno
        record['definition_lines'] = lines
    return record


def build_report(*, model_config=None, cube_cores=20):
    target = json.loads((ROOT/'configs/target.json').read_text())
    cfg = variant_config(target, 'candidate')
    if cfg.get('experimental_request_windows', False):
        raise ValueError('this report only supports current physical-page snapshots')
    geometry = (reference_geometry() if model_config is None else
                geometry_from_model(json.loads(model_config.read_text()), target))
    if (target['tensor_parallel_size'] != 4 or target['max_num_batched_tokens'] != 16384
            or target['max_num_seqs'] != 128 or cfg['enable_prefix_caching'] is not False
            or target['speculative_config']['num_speculative_tokens'] != 3
            or target['speculative_config']['enforce_eager'] is not True):
        raise ValueError('this audit requires revalidation after TP/budget/sequence/APC/MTP changes')
    native = planner()
    shape_env = native_functions(SHAPES, {'gated_delta_net_state_shape'},
                                 {'divide': lambda a,b: a//b})
    shape = shape_env['gated_delta_net_state_shape'].__func__(
        NS(_orient_conv_shape=lambda dim, length: (length, dim)), 4,
        geometry.linear_key_heads, geometry.linear_value_heads, geometry.linear_key_dim,
        geometry.linear_value_dim, geometry.conv_kernel, 3)
    conv, ssm = (math.prod(x)*2 for x in shape)
    native_block = ceil_div(ssm, 128*geometry.local_kv_heads*geometry.head_dim*2)*128
    slot = SlotLayout(geometry.head_dim, num_kv_heads=geometry.local_kv_heads)
    page_bytes = native_block*slot.bf16_token_bytes + conv
    full_count = geometry.target_full_layers + geometry.draft_full_layers
    # Select the Mamba class method specifically (other specs define the same name).
    tree = ast.parse((ROOT/SPECS).read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MambaSpec')
    fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='max_memory_usage_bytes')
    env = {'cdiv':ceil_div}
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(ROOT/SPECS),'exec'),env)
    budgets = observed_budgets()
    results=[]
    for mode in ('none','align'):
        page = HybridPageLayout(slot,conv,ssm,target['max_model_len'] if mode=='none' else native_block,
                                native_page_size_bytes=page_bytes,native_mamba_cache_mode=mode)
        snapshots = SnapshotLayout(page,cfg['sink_tokens'],cfg['recent_tokens'],3)
        all_groups={}
        for variant, block in [('native',native_block),('oscar',page.block_size)]:
            specs={f'full{i}':ShapeSpec('full',block,page_bytes) for i in range(full_count)}
            specs.update({f'gdn{i}':ShapeSpec('gdn',target['max_model_len'] if mode=='none' else native_block,page_bytes)
                          for i in range(geometry.gdn_layers)})
            groups=native['_get_kv_cache_groups_uniform_page_size'](specs)
            all_groups[variant]=groups
        groups=all_groups['oscar']; width=max(len(g.layer_names) for g in groups)
        pool_unit=native['_pool_bytes_per_block'](groups)
        full_groups=sum(g.kv_cache_spec.kind=='full' for g in groups)
        gdn_groups=sum(g.kv_cache_spec.kind=='gdn' for g in groups)
        peak_gdn=env['max_memory_usage_bytes'](NS(page_size_bytes=page_bytes,num_speculative_blocks=3),
                    NS(cache_config=NS(mamba_cache_mode=mode)))//page_bytes
        minimum_gdn=4
        allocations={}
        for label,budget in budgets['scenarios'].items():
            p=native['get_kv_cache_config_from_groups'](
                NS(cache_config=NS(num_gpu_blocks_override=None)),groups,budget)
            allocations[label]={'num_pool_blocks':p.num_blocks,'tensor_count':len(p.kv_cache_tensors),
                'allocated_bytes':sum(t.size for t in p.kv_cache_tensors),'unused_budget_bytes':budget-sum(t.size for t in p.kv_cache_tensors)}
        rows=[]
        for length in (32768,65536,131072):
            for output in (0,512):
                need=length+output+3
                counts={v:ceil_div(need,b)*full_groups for v,b in [('native',native_block),('oscar',page.block_size)]}
                demand={v:{'full_pool_blocks':count,'full_bytes':count*pool_unit,
                    'total_min_live_bytes':(count+gdn_groups*minimum_gdn)*pool_unit,
                    'total_peak_reserved_bytes':(count+gdn_groups*peak_gdn)*pool_unit}
                    for v,count in counts.items()}
                caps={}
                for label,allocated in allocations.items():
                    available=allocated['num_pool_blocks']-1
                    caps[label]={v:{'max_B_by_minimum_live':min(128,available//(c+gdn_groups*minimum_gdn)),
                        'max_B_by_peak_reservation':min(128,available//(c+gdn_groups*peak_gdn))} for v,c in counts.items()}
                sweep=[]
                for batch in BATCHES:
                    by_variant={v:{'min_live_pool_bytes_including_one_null':batch*values['total_min_live_bytes']+pool_unit,
                        'peak_reserved_pool_bytes_including_one_null':batch*values['total_peak_reserved_bytes']+pool_unit,
                        'fits_observed_lower_at_peak':batch<=caps['observed_oscar_rounding_lower'][v]['max_B_by_peak_reservation']} for v,values in demand.items()}
                    sweep.append({'batch':batch,'variants':by_variant,'q4_scheduler_tokens':4*batch,
                                  'within_max_seqs_and_token_budget':batch<=128 and 4*batch<=16384})
                rows.append({'context_tokens':length,'additional_output_reserve':output,'mtp_lookahead_tokens':3,
                    'reserved_full_tokens':need,'per_request_bytes_per_rank':demand,'capacity':caps,'batch_sweep':sweep,
                    'native_over_oscar_full_physical_ratio':demand['native']['full_bytes']/demand['oscar']['full_bytes'],
                    'native_over_oscar_all_kv_peak_ratio':demand['native']['total_peak_reserved_bytes']/demand['oscar']['total_peak_reserved_bytes']})
        results.append({'mode':mode,'production_default':mode=='none','native_full_tokens_per_page':native_block,
            'oscar_full_tokens_per_page':page.block_size,'page_bytes_per_layer':page_bytes,
            'shared_pool_width':width,'group_layer_counts':[len(g.layer_names) for g in groups],
            'full_groups':full_groups,'gdn_groups':gdn_groups,'gdn_padding_slots':width*gdn_groups-geometry.gdn_layers,
            'one_pool_block_and_null_bytes':pool_unit,'gdn_states_per_group_minimum':minimum_gdn,
            'gdn_states_per_group_peak':peak_gdn,'gdn_per_request_raw_payload_at_peak':geometry.gdn_layers*peak_gdn*(conv+ssm),
            'oscar_full_page_components':{'int2_capacity':page.payload_bytes,'precise_snapshot':snapshots.required_bytes,
                'unused_conv':conv,'unused_ssm_tail':page.unused_ssm_bytes,'unused_padding':page.native_padding_bytes-snapshots.required_bytes},
            'native_planner_allocations':allocations,'requests':rows})
    work=WorkspaceGeometry(16384,geometry.local_query_heads,geometry.local_kv_heads,geometry.head_dim,
        cube_cores=cube_cores,history_cluster_size=16,c4_member_pipeline=True,bf16_history=True)
    route=[]
    flags={key:cfg['experimental_'+key] for key in ('fast_unpack','mixed_cv','striped_cache','decode_bundle','c4_member_pipeline','mdl_decode','bf16_history')}
    captures=cfg['compilation_config']['cudagraph_capture_sizes']
    for batch in BATCHES:
        stages=[]
        for name,n,q,layers,repeat in [('target',next(x for x in captures if x>=4*batch),4,geometry.target_full_layers,1),
            ('first_mtp',4*batch,4,geometry.draft_full_layers,1),('later_mtp',4*batch,1,geometry.draft_full_layers,2)]:
            splits=work.splits_for_tokens(n)
            op=select_cv_op(16,geometry.local_query_heads,geometry.local_kv_heads,n,q,q1_draft=q==1,**flags)
            tile_work=[]
            for length in (32768,65536,131072):
                # Actual prepare source0 range [sink, L+q-recent), where L
                # denotes cached context and q<=4. Splits partition this range.
                span=max(0,length+q-cfg['recent_tokens']-cfg['sink_tokens'])
                width=ceil_div(span,splits)
                tiles=sum(ceil_div(max(0,min(span,(s+1)*width)-min(span,s*width)),256) for s in range(splits))
                tile_work.append({'context':length,'history_kv256_tiles_per_request_per_layer':tiles,
                    'history_kv256_tiles_stage':tiles*batch*layers*repeat})
            stages.append({'stage':name,'calls_per_step':repeat,'full_layers':layers,'input_rows':n,
                'live_rows':batch*q,'padding_rows':n-batch*q,'source_splits':splits,'operator':op,
                'task_descriptors_per_layer':n*geometry.local_kv_heads*3*splits,
                'live_leader_source_split_tasks_per_layer':batch*geometry.local_kv_heads*3*splits,
                'native_current_source2':name=='first_mtp','compact_later_applies':name=='later_mtp' and should_compact_rows(n,batch),
                'history_tiles':tile_work,'history_tile_scope':'At the same cached context L in each stage; actual rejection/accepted-position shifts can change boundary tiles. No latency or bandwidth inference.'})
        route.append({'batch':batch,'stages':stages})
    mixed=[]
    for batch in (31,32,33,64,127):
        for warm in (512,8232,16384-4*batch):
            n=4*batch+warm
            metadata=NS(mixed_decode_split=cfg['experimental_mixed_decode_split'],
                capture_origin=False,dummy_origin=False,is_draft=False,
                query_start_loc_cpu=torch.tensor([4*i for i in range(batch+1)]+[n],dtype=torch.int32),
                num_reqs=batch+1,num_actual_tokens=n,cv_shape_tokens=None)
            plan=plan_mixed_decode(metadata,n,query_heads=geometry.local_query_heads,
                                   kv_heads=geometry.local_kv_heads)
            whole_op=select_cv_op(16,geometry.local_query_heads,geometry.local_kv_heads,n,warm,**flags)
            launches=[]
            segments=(('short_prefix',4*batch,4),('long_suffix',warm,warm)) if plan else (('whole_batch',n,warm),)
            original=plan.original_tokens if plan else n
            for segment,rows,max_query in segments:
                launches.append({'segment':segment,'input_rows':rows,'max_query_len':max_query,
                    'cv_shape_tokens':original,'source_splits':work.splits_for_tokens(original),
                    'operator':select_cv_op(16,geometry.local_query_heads,geometry.local_kv_heads,
                                            rows,max_query,**flags)})
            mixed.append({'q4_prefix_requests':batch,'continuation_requests':1,
                'q4_prefix_tokens':4*batch,'continuation_query_tokens':warm,'actual_tokens':n,
                'within_max_seqs_and_token_budget':batch+1<=128 and n<=16384,
                'plan':asdict(plan) if plan else None,
                'whole_batch_operator':whole_op,'launches':launches,
                'comparison_scope':'Target eager continued prefill with positive cached context on every request; current-only suffix removal is inapplicable. Planner and selector execute on host; no device execution.',
                'unpartitioned_reason':None if plan else 'q4 prefix has more than 128 tokens; normal whole-batch CV route remains active.'})
    source_paths=['configs/target.json',PLANNER,SHAPES,SPECS,LOG,'oscar_ascend/layout.py','oscar_ascend/lifecycle.py',
        'oscar_ascend/runtime.py','oscar_ascend/rotations.py','oscar_ascend/ops/cv_dispatch.py','oscar_ascend/integration/impl.py',
        'oscar_ascend/integration/compact_later_mtp.py','oscar_ascend/integration/mixed_decode_split.py','tools/serving_variants.py',
        'references/vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py',
        'references/vllm/vllm/v1/core/kv_cache_manager.py','references/vllm/vllm/v1/core/single_type_kv_cache_manager.py',
        'references/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py','csrc/kernels/attention_tasks.cpp',
        'tests/test_runtime_native.py','tests/test_memory_64k_accounting.py', 'tools/experiments/batch_capacity_20261019.py']
    return {'schema':1,'evidence_kind':'source_arithmetic_and_native_planner_host_execution_not_NPU_profiling',
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'configuration':{'tp':4,'devices':cfg['devices'],'port':cfg['port'],'apc':cfg['enable_prefix_caching'],
                         'budget_tokens':16384,'max_seqs':128,'mtp':3},
        'model':{'geometry':asdict(geometry),'status':'conditional_existing_native_shape_fixture' if model_config is None else 'supplied_model_config_not_runtime_verified',
                 'model_config':str(model_config) if model_config else None,
                 'missing':'Current target model config/KVCacheSpec dump absent; 16+1 FULL and 48 GDN are conditional existing fixture dimensions.' if model_config is None else None,
                 'gdn_shapes':shape,'gdn_dtype':'bfloat16','conv_bytes':conv,'ssm_bytes':ssm},
        'budgets':budgets,'modes':results,
        'fixed_oscar_allocations':{'workspace_instances_per_rank_if_uniform_target_and_draft':1,
            'workspace_buffer_bytes':work.buffer_bytes,'workspace_total_bytes':work.total_bytes,
            'cube_cores':cube_cores,'cube_cores_status':'caller_supplied_conditional_not_current_log_measured',
            'persistent_rotations_bytes':(3*geometry.target_full_layers+1)*geometry.head_dim**2*4,
            'source': 'runtime.ensure_workspace key=(device,geometry); rotation cache=(device,D), target three matrices, shared draft identity.',
            'native_fia_causal_mask_bytes_if_materialized':2048**2,
            'native_fia_current_output_lse_and_other_temporaries':'Dynamic/device-framework peaks not known; included in target profiling, not invented as fixed bytes.',
            'double_subtraction_from_profiled_budget':False},
        'pure_decode_work':route,
        'mixed_continued_prefill_work':mixed,
        'boundaries':['Zero cross-request prefix sharing; align checkpoint caching is evictable within the same pool.',
            'One null pool block deducted globally, not one per request.',
            'MTP lookahead +3 FULL tokens is distinct from 3 reserved GDN states.',
            'All capacity counts assume resident full histories; scheduler admission, phase peaks, fragmentation and output growth can lower live B.',
            'Pure q4 verification assumes three draft tokens available. Mixed prefill and first-token q1 steps differ.',
            'Mixed continued prefill allows a q4 prefix up to B32/128 rows. B33/132 rows remains on the normal whole-batch C4/C16 route. Qualified child launches retain the original batch split count, not pure-decode parallelism.',
            'Later MTP keeps N=4B with only B live rows for these shapes; no shape recovery or automatic compact applied.',
            'Task counts and KV tile work are not device time or HBM bytes. Numerical gates and target Trap remain unresolved.'],
        'sources':[source_record(p) for p in source_paths]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config',type=Path)
    parser.add_argument('--cube-cores',type=int,default=20)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/batch_capacity_20261019.json')
    args=parser.parse_args()
    report=build_report(model_config=args.model_config,cube_cores=args.cube_cores)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {args.output}; conditional host capacity, not measured throughput.')
    for mode in report['modes']:
        for row in mode['requests']:
            if row['additional_output_reserve']==0:
                cap=row['capacity']['observed_oscar_rounding_lower']
                print(mode['mode'],row['context_tokens'],'native/oscar peak B',
                      cap['native']['max_B_by_peak_reservation'],cap['oscar']['max_B_by_peak_reservation'])


if __name__=='__main__':
    main()
