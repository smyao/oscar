"""Archive #108/#125/#126/#145/#148/#154: real-NPU candidate rotation gate.

The frozen rotate_out remains an exact paired reference; ops.reference supplies
an independent FP32 oracle. Host math is only the oracle, never the device path.
No performance conclusion follows from this short numerical/graph gate.
"""
from __future__ import annotations

OPERATOR='rotate_vector_init_out'
CASE_SPECS=((64,'bfloat16',True),(128,'bfloat16',True),(256,'bfloat16',True),
            (256,'float16',True),(256,'float32',True),(64,'bfloat16',False))


def validate_evidence(report):
    required={'status':'passed','operator':OPERATOR,'device_completion':'passed',
              'precision':'bitwise_and_independent_oracle_passed','error_contract':'passed',
              'graph_capture':'passed','graph_replay':'passed',
              'same_address_changed_query_and_slots':True,'graph_eager_bitwise':'passed','device_type':'npu',
              'oracle_tolerance':{'atol':0.002,'rtol':0.002}}
    missing=[key for key,value in required.items() if not isinstance(report,dict) or report.get(key)!=value]
    rows=report.get('cases',[]) if isinstance(report,dict) else []
    expected={(dim,'torch.'+dtype,hadamard) for dim,dtype,hadamard in CASE_SPECS}
    actual={(r.get('dim'),r.get('dtype'),r.get('hadamard')) for r in rows if isinstance(r,dict) and r.get('precision')=='passed'} if isinstance(rows,list) else set()
    if not isinstance(rows,list) or len(rows)!=len(expected) or actual!=expected:missing.append('cases')
    return missing


def gate(torch,ops,device,acceptance):
    if device.type!='npu' or not torch.npu.is_available():
        raise RuntimeError('vector Init rotation gate requires real target NPU')
    from oscar_ascend.ops.reference import rotate_clip
    from .probe_history_reuse import _hadamard
    from .probe_striped_cache import _store_bits_equal
    tolerance=acceptance['store_dequant']
    if tolerance!={'atol':0.002,'rtol':0.002}:raise RuntimeError('frozen rotation tolerance changed')
    generator=torch.Generator(device='cpu').manual_seed(20261011)
    cases=[];graph_evidence=None
    for dim,dtype_name,hadamard in CASE_SPECS:
        dtype=getattr(torch,dtype_name)
        cpu=torch.randn(9,3,dim,generator=generator).to(dtype)
        cpu[0,0,0]=-0.0
        cpu[0,1,1]={torch.float16:2.0**-24,torch.bfloat16:2.0**-133,torch.float32:2.0**-149}[dtype]
        # A padded nonfinite token must be cleared before shared UB transforms.
        cpu[8].fill_(float('nan'))
        slots_cpu=torch.tensor([0,-1,2,3,4,5,6,7,-1],dtype=torch.int64)
        matrix_cpu=_hadamard(torch,dim)
        if not hadamard:matrix_cpu=matrix_cpu.roll(1,0).contiguous()
        x=cpu.to(device);matrix=matrix_cpu.T.contiguous().to(device);slots=slots_cpu.to(device)
        outputs=[]
        for _ in range(2):
            ybase=torch.full((cpu.numel()+128,),-777.,dtype=torch.float32,device=device)
            sbase=torch.full((27+128,),-777,dtype=torch.int32,device=device)
            outputs.append((ybase[64:-64].view(cpu.shape),sbase[64:-64].view(9,3),ybase,sbase))
        def reset():
            for y,status,_,_ in outputs:y.fill_(float('nan'));status.fill_(-99)
        def launch(index):
            y,status,_,_=outputs[index]
            getattr(ops,('rotate_out',OPERATOR)[index])(x,matrix,y,status,hadamard,slots)
        def check():
            old,new=(tuple(t.cpu() for t in output[:2]) for output in outputs)
            if not all(_store_bits_equal(torch,a,b) for a,b in zip(old,new)):
                raise RuntimeError('vector Init changed output/status bits')
            if not bool((new[1]==0).all()):raise RuntimeError('vector Init valid status failed')
            clean=cpu.clone();clean[slots_cpu<0]=0
            expected=rotate_clip(clean,matrix_cpu);expected[slots_cpu<0]=0
            for y,status,ybase,sbase in outputs:
                torch.testing.assert_close(y.cpu(),expected,**tolerance)
                if not bool((y.cpu()[slots_cpu<0]==0).all()):raise RuntimeError('rotation padding was not zero')
                if not all(bool((base[:64].cpu()==-777).all() and (base[-64:].cpu()==-777).all()) for base in (ybase,sbase)):
                    raise RuntimeError('rotation guard overwritten')
            if (not _store_bits_equal(torch,x.cpu(),cpu) or
                    not _store_bits_equal(torch,matrix.cpu(),matrix_cpu.T.contiguous()) or
                    not torch.equal(slots.cpu(),slots_cpu)):
                raise RuntimeError('rotation mutated an input')
            return tuple(t.clone() for t in new)
        reset();launch(0);launch(1);torch.npu.synchronize();initial=check()
        if dim==256 and dtype==torch.bfloat16 and hadamard:
            graphs=[torch.npu.NPUGraph(),torch.npu.NPUGraph()]
            addresses=(x.data_ptr(),matrix.data_ptr(),slots.data_ptr(),
                       *[t.data_ptr() for output in outputs for t in output[:2]])
            for i,graph in enumerate(graphs):
                with torch.npu.graph(graph,capture_error_mode='thread_local',auto_dispatch_capture=True):launch(i)
            reset()
            for graph in graphs:graph.replay()
            torch.npu.synchronize();captured=check()
            if not all(_store_bits_equal(torch,a,b) for a,b in zip(initial,captured)):
                raise RuntimeError('rotation original-input graph differs')
            cpu.neg_();x.copy_(cpu)
            slots_cpu[0]=-1;slots_cpu[1]=1;slots.copy_(slots_cpu)
            if addresses!=(x.data_ptr(),matrix.data_ptr(),slots.data_ptr(),*[t.data_ptr() for output in outputs for t in output[:2]]):
                raise RuntimeError('rotation graph input/output addresses changed')
            reset()
            for graph in graphs:graph.replay()
            torch.npu.synchronize();changed=check()
            if _store_bits_equal(torch,initial[0],changed[0]):raise RuntimeError('rotation graph ignored changed inputs')
            reset();launch(0);launch(1);torch.npu.synchronize();eager=check()
            if not all(_store_bits_equal(torch,a,b) for a,b in zip(changed,eager)):
                raise RuntimeError('rotation changed-input graph differs from eager')
            graph_evidence={'graph_capture':'passed','graph_replay':'passed',
                            'same_address_changed_query_and_slots':True,'graph_eager_bitwise':'passed'}
        cases.append({'dim':dim,'dtype':str(dtype),'hadamard':hadamard,'precision':'passed'})
    # Retain live NaN/Inf/overflow error detection and all-padding suppression.
    cpu=torch.ones(9,3,64,dtype=torch.float32);cpu[0].fill_(float('nan'))
    cpu[2].fill_(float('inf'));cpu[3].fill_(torch.finfo(torch.float32).max)
    x=cpu.to(device);matrix=_hadamard(torch,64).T.contiguous().to(device)
    for all_padding in (False,True):
        slot=torch.full((9,),-1 if all_padding else 0,dtype=torch.int64,device=device)
        pair=[]
        for name in ('rotate_out',OPERATOR):
            y=torch.full(cpu.shape,float('nan'),dtype=torch.float32,device=device);status=torch.full((9,3),-99,dtype=torch.int32,device=device)
            getattr(ops,name)(x,matrix,y,status,True,slot);torch.npu.synchronize();pair.append((y.cpu(),status.cpu()))
        if not all(_store_bits_equal(torch,a,b) for a,b in zip(*pair)):
            raise RuntimeError('vector Init invalid-input output/status bits differ')
        y,status=pair[1]
        if all_padding:
            if not bool((status==0).all() and (y==0).all()):raise RuntimeError('all-padding rotation retained an error')
        else:
            expected=torch.zeros_like(status);expected[[0,2,3]]=2
            if not torch.equal(status,expected):raise RuntimeError('rotation live NaN/Inf/overflow status changed')
    if graph_evidence is None:raise RuntimeError('rotation graph case did not execute')
    return {'status':'passed','operator':OPERATOR,'device_completion':'passed',
            'precision':'bitwise_and_independent_oracle_passed','error_contract':'passed',
            'device_type':device.type,'oracle_tolerance':tolerance,**graph_evidence,'cases':cases,'performance':'not_measured','full_model_quality':'not_established'}
