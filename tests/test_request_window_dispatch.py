"""Archive #34/#36/#37-49/#69/#127-129/#155: no format confusion or unguarded stores."""
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.cache_storage_ops import (
    REQUEST_FORMAT, REQUEST_READERS, request_map, storage_reader, store_current)
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.ops.cv_dispatch import select_cv_op


@pytest.mark.parametrize('tokens,qlen,draft,expected', [
    (128,4,False,'decode'), (128,1,True,'decode_q1'),
    (8,8,False,''), (512,8,False,'balanced'),
    (512,128,False,'cluster4'), (16384,16384,False,'cluster16')])
def test_whole_qualified_dispatch_selects_matching_storage_reader(tokens,qlen,draft,expected):
    original=select_cv_op(16,6,1,tokens,qlen,q1_draft=draft,
                         fast_unpack=True,mixed_cv=True,striped_cache=True,decode_bundle=True)
    name=storage_reader(SimpleNamespace(cache_format=REQUEST_FORMAT),original)
    assert name=='attention_request_window'+('_'+expected if expected else '')+'_out'
    assert storage_reader(SimpleNamespace(cache_format='striped_v1'),original)==original


def test_refuse_unqualified_reader_and_bad_identity_slice():
    state=SimpleNamespace(cache_format=REQUEST_FORMAT)
    for name in ('attention_cv_out','attention_cv_striped_q1_out'):
        with pytest.raises(OscarReadinessError,match='complete reader bundle'):
            storage_reader(state,name)
    mapping=torch.tensor([[5,40],[3,41],[2,42]],dtype=torch.int64)
    meta=SimpleNamespace(window_map=mapping,num_reqs=3)
    selected=request_map(state,meta,1,2)
    assert selected.tolist()==[[3,41],[2,42]]
    assert selected.untyped_storage().data_ptr()==mapping.untyped_storage().data_ptr()
    for first,count in ((-1,1),(2,2),(0,4)):
        with pytest.raises(OscarReadinessError):request_map(state,meta,first,count)
    with pytest.raises(OscarReadinessError):
        request_map(state,SimpleNamespace(num_reqs=3,window_map=None))


@pytest.mark.parametrize('error',[0,7])
def test_new_store_publishes_once_then_guards_window_status(error):
    # Values only identify the ABI slots; no host numerical store substitutes
    # for AscendC. Injecting a sidecar error must stop before the next reader.
    n=4
    state=SimpleNamespace(cache_format=REQUEST_FORMAT, rotation_k_transpose=object(),
        rotation_v_transpose=object(),raw=object(),window_key=object(),window_value=object(),
        window_tags=object(),window_status=torch.zeros(128,dtype=torch.int32),
        workspace=SimpleNamespace(store_status=torch.zeros(n,1,dtype=torch.int32)),
        spec=SimpleNamespace(block_size=5632,conv_bytes=30720,ssm_bytes=393216),
        num_blocks=40,hadamard=True,
        snapshots=SimpleNamespace(sink_tokens=64,ring_tokens=259,max_num_seqs=128,position_bits=18))
    qstarts=torch.tensor([0,4],dtype=torch.int32)
    mapping=torch.tensor([[9,24]],dtype=torch.int64)
    k=torch.empty(n,1,256,dtype=torch.bfloat16);v=torch.empty_like(k)
    slots=torch.arange(n,dtype=torch.int64);positions=slots+65100
    order=[]
    class Ops:
        def rotate_clip_store_request_window_out(self,*args):
            order.append('packed_then_window')
            assert len(args)==25
            assert args[4] is slots and args[5] is positions
            assert args[6] is qstarts and args[7] is mapping
            assert args[8] is state.raw and args[13] is state.window_status
            assert args[14:]==(5632,40,40*30720,393216,64,259,0.95,0.9,True,128,18)
            state.window_status[0]=error
        def status_guard(self,*args):
            order.append('guard')
            assert args[0] is state.window_status
            if bool(args[0].ne(0).any()):raise RuntimeError('device trap')
    run=lambda:store_current(Ops(),state,k,v,slots,positions,qstarts,mapping,
                             {'k_clip_ratio':.95,'v_clip_ratio':.9})
    if error:
        with pytest.raises(RuntimeError,match='device trap'):run()
    else:run()
    assert order==['packed_then_window','guard']
