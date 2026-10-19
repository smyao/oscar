"""Archive #127/#128/#154/#157/#160/#161/#163/#165/#167: serial host audit.

Run the actual production forward/current-only dispatch with one persistent
workspace and real physical snapshot views. Operators are explicit host
contract doubles: they check addresses, source ownership and order, not INT2
arithmetic, native FIA accuracy, CANN completion, graph replay or Trap cause.
"""
from contextlib import nullcontext
from enum import Enum
from types import MethodType, SimpleNamespace

import pytest
import torch

from oscar_ascend.bf16_history_policy import validate_bf16_history
from oscar_ascend.integration import current_attention, current_only_dispatch
from oscar_ascend.integration.metadata import from_common
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.layout import HybridPageLayout, SlotLayout
from oscar_ascend.lifecycle import SnapshotLayout
from oscar_ascend.ops.cv_dispatch import CLUSTER_CV_OPS, select_cv_op
from oscar_ascend.runtime import GraphWorkspace, WorkspaceGeometry
from test_runtime_native import definitions
from tools.serving_variants import variant_config


class AscendAttentionState(Enum):
    ChunkedPrefill = 1
    SpecDecoding = 2


def _workspace():
    n, h, d = 16384, 6, 256
    work = object.__new__(GraphWorkspace)
    work.geometry = WorkspaceGeometry(n, h, 1, d, cube_cores=20,
        history_cluster_size=16, c4_member_pipeline=True, bf16_history=True)
    for name, shape, dtype in (
        ('query_input', (n,h,d), torch.bfloat16), ('key_input', (n,1,d), torch.bfloat16),
        ('value_input', (n,1,d), torch.bfloat16), ('query_rot', (n,h,d+16), torch.bfloat16),
        ('partial', (n,h,3,d), torch.float32), ('partial_lse', (n,h,3), torch.float32),
        ('output', (n,h,d), torch.float32), ('lse', (n,h), torch.float32),
        ('rotate_status', (n,h), torch.int32), ('merge_status', (n,h), torch.int32),
        ('store_status', (n,1), torch.int32), ('tasks', (n*3,16), torch.int64),
        ('attention_status', (n*3,2), torch.int32), ('positions', (n,), torch.int64),
        ('slots', (n,), torch.int64), ('cluster_stats', (20,8), torch.int64)):
        setattr(work, name, torch.empty(shape, dtype=dtype))
    # Device workspace addressing is tested separately with actual source/SDK
    # models. This double never accesses the byte arena.
    work.cv = torch.empty(1, dtype=torch.uint8)
    return work


@pytest.mark.parametrize('first_draft', [False, True], ids=['target', 'first_mtp'])
@pytest.mark.parametrize('mode,warm', [('none',8232), ('none',16384), ('align',8232), ('align',16384)])
def test_serial_first_chunk_history_q4_q1_preserves_cache_and_workspace(monkeypatch, mode, warm, first_draft):
    work = _workspace()
    page = HybridPageLayout(SlotLayout(256), 15360, 393216,
        262144 if mode == 'none' else 768,
        native_page_size_bytes=801792, native_mamba_cache_mode=mode)
    snapshot = SnapshotLayout(page,64,256,3)
    pages = (16384 + warm + 8 + page.block_size - 1)//page.block_size
    raw = torch.full((pages*page.page_size_bytes,),91,dtype=torch.uint8)
    wk,wv,tags = snapshot.views(raw,pages)
    tags.fill_(-1)
    before_payload = raw[:snapshot.base(pages)].clone()
    # Reverse physical pages so logical ordinals cannot accidentally serve as
    # physical slot IDs; each physical page expands into virtual128 blocks.
    order = list(reversed(range(pages)))
    per_page = page.block_size//128
    table = torch.tensor([[p*per_page+j for p in order for j in range(per_page)]],dtype=torch.int32)
    state = SimpleNamespace(raw=raw,packed=raw,spec=SimpleNamespace(block_size=page.block_size,
        conv_bytes=page.conv_bytes,ssm_bytes=page.ssm_bytes),snapshots=snapshot,
        num_blocks=pages,window_key=wk,window_value=wv,window_tags=tags,workspace=work,
        rotation_k_transpose=torch.eye(256),rotation_v=torch.eye(256),
        rotation_v_transpose=torch.eye(256),hadamard=True,cache_format='striped_v1')
    events=[]; context={}; stored={}

    def guard(ops,site,parts,**_):
        assert all(t.eq(0).all() for _,t in parts), site
        events.append(('guard',site))

    def native(q,k,v,cumulative,**_):
        assert cumulative == (context['count'],)
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        events.append(('native',q.shape[0]))
        return torch.full_like(q,3),torch.zeros(q.shape[0],6,1)

    def current_slots(ops,status,slots,active):
        assert slots[:active].ge(0).all()
        events.append(('slots',active))

    def publish_partial(partial,lse,out,out_lse,active,splits):
        events.append(('current_partial',active))
        partial[:active,:,2*splits].copy_(out)
        lse[:active,:,2*splits].copy_(out_lse.squeeze(-1))

    class Ops:
        def prepare_attention_tasks_out(self,starts,lens,slots,tasks,positions,h,hk,sink,recent,splits,block_table,slotctx):
            events.append(('prepare',slots.numel()))
            assert starts.tolist()==[0,context['count']]
            assert lens.tolist()==[context['end']]
            assert slotctx is context['later']
            assert (block_table is not None) is slotctx
            active=context['count']; begin=context['begin']
            expected_positions=torch.arange(begin,begin+active)
            expected_slots=table[0,expected_positions//128].to(torch.int64)*128+expected_positions%128
            assert torch.equal(slots[:active],expected_slots)
            assert slots[active:].eq(-1).all()
            positions.fill_(-1);positions[:active].copy_(expected_positions)
            tasks.zero_()
            view=tasks.view(slots.numel(),1,3,splits,16)
            view[...,1]=-1
            view[:active,...,1]=1
            view[:active,...,8]=begin
            for source in range(3):
                view[:,:,source,:,7]=source
            view[:active,:,2,:,3]=begin
            view[:active,:,2,:,4]=begin+active

        def rotate_bf16_query_out(self,q,rotation,qr,status,hadamard,slots):
            events.append(('rotate',q.shape[0]))
            qr[...,:256].copy_(q);qr[...,256:].zero_();status.zero_()

        def __getattr__(self,name):
            if name not in {'attention_bf16_history_window_range_cluster16_out',
                    'attention_bf16_history_mdl_out','attention_bf16_history_mdl_q1_out'}:
                raise AttributeError(name)
            def cv(*args):
                events.append(('cv',name))
                q,qr,k,v,rotation,owner,blocks,key,value,seen,task,partial,lse,status,arena=args[:15]
                assert owner is raw and blocks.data_ptr()==table.data_ptr()
                assert key.data_ptr()==wk.data_ptr() and value.data_ptr()==wv.data_ptr() and seen.data_ptr()==tags.data_ptr()
                splits=partial.shape[2]//3
                current=task.view(q.shape[0],1,3,splits,16)[:,:,2]
                if context['native']:
                    assert torch.equal(current[...,3],current[...,4])
                else:
                    assert current[:context['count'],...,4].gt(current[:context['count'],...,3]).all()
                # Prior writes must cover the complete history domain before
                # current publication. Validate every exact sink/recent tag.
                begin=context['begin']
                assert all(p in stored for p in range(begin))
                for p in list(range(min(64,begin)))+list(range(max(64,begin-256),begin)):
                    physical=order[p//page.block_size]; offset=p%page.block_size
                    row=p if p<64 else 64+offset%259
                    assert int(tags[physical,row])==offset
                    assert int(wk[physical,row,0,0])==stored[p]
                partial.zero_();lse.fill_(float('-inf'));status.zero_()
                partial[:context['count'],:,0].fill_(7);lse[:context['count'],:,0].zero_()
            return cv

        def merge_lse_out(self,partial,lse,out,out_lse,status):
            events.append(('merge',out.shape[0]))
            out.copy_(partial[:,0]);out_lse.zero_();status.zero_()

        def copy_validate_current_out(self,current,lse,out,out_lse,status):
            events.append(('copy',current.shape[0]))
            out.copy_(current);out_lse.copy_(lse.squeeze(-1));status.zero_()

        def rotate_clip_store_striped_out(self,k,v,rk,rv,slots,positions,owner,key,value,seen,status,*attrs):
            assert any(event[0] in {'merge','copy'} for event in events)
            assert not any(event[0]=='store' for event in events)
            events.append(('store',k.shape[0]))
            for token in range(context['count']):
                p=int(positions[token]);physical=int(slots[token])//page.block_size
                offset=p%page.block_size; rows=[64+offset%259]
                if p<64: rows.append(p)
                stored[p]=context['stamp']
                for row in rows:
                    key[physical,row].fill_(context['stamp'])
                    value[physical,row].fill_(context['stamp'])
                    seen[physical,row]=offset
            status.zero_()

    forward=definitions('oscar_ascend/integration/impl.py','oscar_ascend.integration._serial_forward_audit',set(),
        {'torch':torch,'OscarReadinessError':OscarReadinessError,'use_native_current':current_attention.use_native_current,
         'validate_bf16_history':validate_bf16_history,'select_cv_op':select_cv_op,'CLUSTER_CV_OPS':CLUSTER_CV_OPS,
         'phase':lambda *a,**k:nullcontext(),'run_status_guard':guard,'guard_current_slots':current_slots,
         'native_current_partial':native,'write_current_partial':publish_partial,
         'suppress_current_source_tasks':current_attention._empty_current_source_ranges,
         'emit_once':lambda *a,**k:None,'emit_throttled':lambda *a,**k:None},monkeypatch,
         methods=[('OscarAttentionImpl','forward')]).forward
    monkeypatch.setattr(current_only_dispatch,'run_status_guard',guard)
    monkeypatch.setattr(current_only_dispatch,'guard_current_slots',current_slots)
    monkeypatch.setattr(current_only_dispatch,'native_current_partial',native)
    impl=SimpleNamespace(num_heads=6,num_kv_heads=1,head_size=256,scale=1/16,
        provider=SimpleNamespace(config=variant_config({},'candidate'),ops=Ops(),layer_state=lambda _:state))
    impl.forward=MethodType(forward,impl)
    specs=[(0,16384,16384,False),(16384,warm,warm,False),
           (16384+warm,4,128,False),(16384+warm+1,1,128,True)]
    addresses=tuple(getattr(work,name).data_ptr() for name in ('tasks','partial','partial_lse','query_rot','slots'))
    for stamp,(begin,count,padded,later) in enumerate(specs,1):
        events.clear()
        # Later q1 follows rejection of three q4 proposals. Native optimistic
        # seq_lens retains all four target rows; the slot identifies the one
        # accepted logical position independently of that upper bound.
        end = 16384 + warm + 5 if later else begin + count
        context.update(begin=begin,count=count,end=end,later=later,
                       native=stamp<=2 or (first_draft and not later),stamp=stamp)
        work.partial.fill_(float('nan'));work.partial_lse.fill_(float('nan'))
        work.tasks.fill_(987654321);work.attention_status.fill_(9)
        positions=torch.arange(begin,begin+count)
        slots=torch.full((padded,),-1,dtype=torch.int32)
        slots[:count]=(table[0,positions//128]*128+positions%128).to(torch.int32)
        starts=torch.tensor([0,count],dtype=torch.int32)
        common=SimpleNamespace(causal=True,query_start_loc=starts,query_start_loc_cpu=starts.clone(),
            seq_lens=torch.tensor([end],dtype=torch.int32),seq_lens_cpu_upper_bound=torch.tensor([end]),
            block_table_tensor=table,slot_mapping=slots,num_reqs=1,num_actual_tokens=count,
            num_input_tokens=padded,max_query_len=count,max_seq_len=end,
            attn_state=AscendAttentionState.ChunkedPrefill if stamp<=2 else AscendAttentionState.SpecDecoding)
        meta=from_common(common,current_only=True,is_draft=first_draft or later,draft_index=int(later),
            first_draft_current_only=first_draft,first_draft_current_fia=first_draft)
        q=torch.zeros(padded,6*256,dtype=torch.bfloat16)
        k=torch.zeros(padded,256,dtype=torch.bfloat16)
        v=torch.zeros(padded,3*256,dtype=torch.bfloat16)[:,256:512]
        out=torch.full_like(q,float('nan'))
        assert impl.forward(SimpleNamespace(layer_name='audit'),q,k,v,raw,meta,output=out) is out
        assert out[:count].eq(3 if stamp==1 else 7).all()
        assert out[count:].eq(0).all()
        assert events[-1][0]=='guard'
        assert [event[1] for event in events if event[0]=='cv']==([] if stamp==1 else [{
            2:'attention_bf16_history_window_range_cluster16_out',3:'attention_bf16_history_mdl_out',
            4:'attention_bf16_history_mdl_q1_out'}[stamp]])
        assert (any(event[0]=='native' for event in events)) is context['native']
        assert tuple(getattr(work,name).data_ptr() for name in ('tasks','partial','partial_lse','query_rot','slots'))==addresses
        # Only physical snapshot padding was written by this host double;
        # native GDN/packed payload bytes remain exactly unchanged.
        assert torch.equal(raw[:snapshot.base(pages)],before_payload)
