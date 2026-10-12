"""Real-NPU six-route numerical/address/graph gate; no CPU fallback.

Archive #126-129/#140/#142/#145/#148/#151-155 and D.4: separate old/new
physical layouts, independent PR oracle, exact status/poison bits, immutable
inputs, real changed-map graph replay. CPU/CANN evidence cannot pass this gate.
The C16 CPU source2-suppressed experiment is NOT used here: all three sources
execute for every numerical/graph reader case. No performance claim is made.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json

from csrc.experiments.request_windows_attention import fixture as inputs


class ReaderGateError(RuntimeError):
    pass


ROUTES = {
    "plain": ("attention_cv_window_range_out", "attention_request_window_out", 1),
    "balanced": ("attention_cv_window_range_balanced_out", "attention_request_window_balanced_out", 1),
    "q4": ("attention_cv_bundle_decode_out", "attention_request_window_decode_out", 1),
    "q1": ("attention_cv_bundle_q1_out", "attention_request_window_decode_q1_out", 1),
    "c4": ("attention_cv_window_range_cluster4_out", "attention_request_window_cluster4_out", 4),
    "c16": ("attention_cv_window_range_cluster16_out", "attention_request_window_cluster16_out", 16),
    "c4_member_pipeline": ("attention_cv_member_pipeline_cluster4_out",
                           "attention_request_window_member_pipeline_cluster4_out", 4),
}


def case_plan(*, c4_member_pipeline=False):
    if type(c4_member_pipeline) is not bool:
        raise ReaderGateError("C4 member pipeline selection must be an explicit boolean")
    # Replace the existing C4 slot, retaining six routes. C337 exercises both
    # pipeline KV buffers, while the old C4 gate keeps its original fixture.
    c4 = (("c4_member_pipeline", inputs.Case("reader_c4_member_pipeline", "c4", (337,),
            padded=84, qlen=84, recent=1, second_segment=True)) if c4_member_pipeline else
          ("c4", inputs.Case("reader_c4", "c4", (65,), padded=84, qlen=84,
                            recent=1, second_segment=True)))
    return (
        ("plain", inputs.Case("reader_plain", "q4", (641, 511), padded=12)),
        ("balanced", inputs.Case("reader_balanced", "q4", (1301, 511), padded=12, order=(1, 0))),
        ("q4", inputs.Case("reader_q4", "q4", (1301,511)*16, padded=128, splits=3,
                           order=tuple(reversed(range(32))))),
        ("q1", inputs.Case("reader_q1", "q1", (1299,1025,511,641)*8, padded=128, splits=3,
                           order=tuple(reversed(range(32))))),
        c4,
        ("c16", inputs.Case("reader_c16", "c16", (65,), padded=336, qlen=336, recent=1, second_segment=True)),
    )


def required_capabilities(*, c4_member_pipeline=False):
    return frozenset({"prepare_attention_tasks_out", "merge_lse_out"} |
                     {name for route, _spec in case_plan(c4_member_pipeline=c4_member_pipeline)
                      for name in ROUTES[route][:2]})


REQUIRED_CAPABILITIES = required_capabilities()


def expected_shape(spec, cores):
    """Declared integer fixture geometry only; never regenerate oracle tensors."""
    qlen=spec.qlen if spec.qlen is not None else 4 if spec.mode=="q4" else 1
    old_b=spec.block;new_b=2*old_b;shift=old_b if spec.second_segment else 0
    old_blocks=sum((context+qlen+old_b-1)//old_b for context in spec.contexts)
    new_blocks=sum((context+qlen+shift+new_b-1)//new_b for context in spec.contexts)
    return [spec.padded,6,1,256,len(spec.contexts),(max(spec.contexts)+qlen+127)//128,
            old_b,old_blocks,new_b,new_blocks,64,393216 if old_b==2816 else old_b*136,
            len(spec.contexts)+3,19,64,spec.recent,3,spec.splits,cores]


def _is_sha256(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def _require_npu(torch, device):
    if (getattr(device, "type", None) != "npu" or not hasattr(torch, "npu") or
            not torch.npu.is_available()):
        raise ReaderGateError("reader gate requires a real NPU; CPU results cannot pass")


def _bytes(tensor):
    return inputs._bytes(tensor.detach().cpu())


def _equal(actual, expected, label):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or _bytes(actual) != _bytes(expected):
        raise ReaderGateError(label+" shape/dtype/bytes differ")


def _fingerprint(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        digest.update(name.encode()+b"\0"+_bytes(tensor))
    return digest.hexdigest()


def _geometry(fixture):
    names = ("n", "hq", "hk", "d", "requests", "columns", "old_b", "old_blocks", "new_b",
             "new_blocks", "prefix", "stride", "capacity", "bits", "sink", "recent", "spec", "splits", "cores")
    return dict(zip(names, fixture["shape"]))


def _fixture(torch, spec, cores):
    result=inputs.make_case(torch,spec)
    result["shape"]=(*result["shape"][:-1],cores)
    return result


def _allocate(torch, fixture, device, cores, route):
    g = _geometry(fixture); n,h,k,d,s = (g[x] for x in ("n","hq","hk","d","splits"))
    cluster = ROUTES[route][2]
    per_core = (2494464 if route=="c4_member_pipeline" else
                4997120 if cluster==16 else 1839104 if cluster==4 else
                1146880 if route in {"q4","q1"} else 917504)
    out = {"tasks": torch.empty((n*k*3*s,16),dtype=torch.int64,device=device),
           "positions": torch.empty(n,dtype=torch.int64,device=device),
           "partial": torch.empty((n,h,3*s,d),dtype=torch.float32,device=device),
           "lse": torch.empty((n,h,3*s),dtype=torch.float32,device=device),
           "status": torch.empty((n*k*3*s,2),dtype=torch.int32,device=device),
           "workspace": torch.empty(cores*per_core,dtype=torch.uint8,device=device),
           "merged": torch.empty((n*h,d),dtype=torch.float32,device=device),
           "merged_lse": torch.empty(n*h,dtype=torch.float32,device=device),
           "merge_status": torch.empty(n*h,dtype=torch.int32,device=device)}
    if cluster>1:
        backing = torch.empty(cores*8+8,dtype=torch.int64,device=device)
        offset = (-backing.data_ptr())%64//8
        out["stats_storage"] = backing
        out["stats"] = backing[offset:offset+cores*8].view(cores,8)
        if out["stats"].data_ptr()%64: raise ReaderGateError("cluster_stats alignment mismatch")
    return out


def _device_inputs(fixture):
    return {name:value for name,value in fixture["inputs"].items() if not name.startswith("expected_")}


def _poison(buffers):
    buffers["partial"].fill_(float("nan")); buffers["lse"].fill_(float("nan"))
    buffers["status"].fill_(0x55555555); buffers["workspace"].fill_(0x85)
    if "stats" in buffers: buffers["stats"].fill_(-1)


def _prepare(torch, ops, tensors, fixture, buffers, side):
    g = _geometry(fixture)
    args = (tensors["starts"],tensors["lens"],tensors[side+"_slots"],buffers["tasks"],
            buffers["positions"],g["hq"],g["hk"],g["sink"],g["recent"],g["splits"])
    if fixture["case"].mode=="q1": args += (tensors[side+"_table"], True)
    ops.prepare_attention_tasks_out(*args)


def reader_arguments(tensors, buffers, fixture, cores, route, side):
    g = _geometry(fixture)
    args = [tensors[key] for key in ("q","qr","ck","cv","rv")]
    args += [tensors[side+"_"+key] for key in ("raw","table","wk","wv","tags")]
    if side=="new": args.append(tensors["request_map"][:g["requests"]])
    args += [buffers[key] for key in ("tasks","partial","lse","status","workspace")]
    if ROUTES[route][2]>1: args.append(buffers["stats"])
    args += [g[side+"_b"],g[side+"_blocks"],g["prefix"],g["stride"],g["sink"],g["recent"],
             g["spec"],g["splits"],g["d"]**-.5,cores]
    if side=="new": args += [g["capacity"],g["bits"]]
    return tuple(args)


def _launch(ops, tensors, buffers, fixture, cores, route, side):
    getattr(ops, ROUTES[route][0 if side=="old" else 1])(*reader_arguments(tensors,buffers,fixture,cores,route,side))


def _metadata_equal(torch, old, new, fixture, *, padding=False):
    _equal(old["tasks"],new["tasks"],"old/new real prepared tasks")
    _equal(old["positions"],new["positions"],"old/new real positions")
    expected = fixture["inputs"]["expected_positions"].clone()
    if padding: expected.fill_(-1)
    _equal(new["positions"],expected,"independent positions")
    tasks = new["tasks"].cpu()
    if not padding and not bool(((tasks[:,7]==1)&(tasks[:,1]>0)&(tasks[:,4]>tasks[:,3])).any()):
        raise ReaderGateError("fixture never executed live exact-window source")


def _oracle(torch, ops, buffers, fixture, acceptance):
    g = _geometry(fixture); n,h,d,s = (g[key] for key in ("n","hq","d","splits"))
    ops.merge_lse_out(buffers["partial"].view(n*h,3*s,d),buffers["lse"].view(n*h,3*s),
                      buffers["merged"],buffers["merged_lse"],buffers["merge_status"])
    torch.npu.synchronize()
    if bool(buffers["merge_status"].cpu().ne(0).any()): raise ReaderGateError("merge error on valid reader case")
    expected = fixture["inputs"]
    torch.testing.assert_close(buffers["merged"].cpu().view(n,h,d),expected["expected_output"],
                               atol=acceptance["atol"],rtol=acceptance["rtol"],equal_nan=False)
    torch.testing.assert_close(buffers["merged_lse"].cpu().view(n,h),expected["expected_lse"],
                               atol=acceptance["atol"],rtol=acceptance["rtol"],equal_nan=False)


def _check_pair(torch, ops, old, new, fixture, acceptance, *, error_code=0, cluster=False, padding=False):
    for key in ("partial","lse","status"):
        _equal(new[key],old[key],"old/new "+key)
    status = new["status"].cpu()
    if error_code:
        if not bool(status.eq(error_code).any()) or bool(((status!=0)&(status!=error_code)).any()):
            raise ReaderGateError("reader did not reject exact requested error domain")
    elif bool(status.ne(0).any()): raise ReaderGateError("valid reader returned error")
    if "stats" in new:
        _equal(new["stats"],old["stats"],"cluster counters")
        if cluster:
            g = _geometry(fixture)
            shared_tiles = (max(0, fixture["case"].contexts[0]-g["sink"])+255)//256
            if int(new["stats"][:,0].sum().cpu())!=1 or int(new["stats"][:,3].sum().cpu())!=shared_tiles:
                raise ReaderGateError("cluster gate did not execute the declared shared history tiles")
    if padding:
        if not bool(new["partial"].cpu().eq(0).all()) or not bool(new["lse"].cpu().eq(-float("inf")).all()):
            raise ReaderGateError("all-padding capture path left poison")
    elif not error_code:
        _oracle(torch,ops,old,fixture,acceptance); _oracle(torch,ops,new,fixture,acceptance)


def _copy_inputs(tensors, fixture):
    for name, value in _device_inputs(fixture).items(): tensors[name].copy_(value)


def _assert_inputs(tensors, fixture):
    for name, value in _device_inputs(fixture).items(): _equal(tensors[name],value,"immutable "+name)


def _candidate_window_fault(torch, old, new, fixture, code):
    tasks=new["tasks"].cpu();status=new["status"].cpu()
    affected=(tasks[:,7]==1)&(tasks[:,5]==0)&(tasks[:,1]>0)&(tasks[:,4]>tasks[:,3])
    if (not bool(affected.any()) or not bool(status[affected].eq(code).any(-1).all()) or bool(status[~affected].ne(0).any()) or
            bool(((status!=0)&(status!=code)).any()) or bool(old["status"].cpu().ne(0).any())):
        raise ReaderGateError("candidate map/range fault escaped live request0/window domain")
    begin,end=map(int,fixture["inputs"]["starts"][:2]);splits=_geometry(fixture)["splits"]
    for key in ("partial","lse"):
        a,b=old[key].cpu(),new[key].cpu()
        mask=torch.ones_like(a,dtype=torch.bool);mask[begin:end,:,splits:2*splits]=False
        _equal(b[mask],a[mask],"unaffected map/range "+key)
        if not bool(torch.isnan(b[begin:end,:,splits:2*splits]).any()):raise ReaderGateError("fault did not poison live window "+key)
    if "stats" in new:_equal(new["stats"],old["stats"],"map fault cluster counters")


def _relocate(fixture, *, epoch_delta=7):
    result = {**fixture,"inputs":{name:tensor.clone() for name,tensor in fixture["inputs"].items()}}
    g = _geometry(result); source=fixture["inputs"]; target=result["inputs"]
    for request in range(g["requests"]):
        slot,epoch = map(int,source["request_map"][request])
        changed_slot=(slot+g["requests"])%g["capacity"]
        target["request_map"][request,0]=changed_slot; target["request_map"][request,1]=epoch+epoch_delta
        for key in ("new_wk","new_wv"): target[key][changed_slot]=source[key][slot]
        row=source["new_tags"][slot].clone(); valid=row>=0
        row[valid]=((epoch+epoch_delta)<<g["bits"]) | (row[valid]&((1<<g["bits"])-1))
        target["new_tags"][changed_slot]=row
    return result


def run_case(torch, ops, device, cores, acceptance, route, spec):
    _require_npu(torch,device)
    fixture=_fixture(torch,spec,cores)
    if fixture["oracle_scope"]!="complete_three_sources": raise ReaderGateError("NPU gate cannot use reduced CPU oracle")
    tensors={name:value.to(device) for name,value in _device_inputs(fixture).items()}
    old,new=(_allocate(torch,fixture,device,cores,route) for _ in range(2))
    for side,buffers in (("old",old),("new",new)):
        _prepare(torch,ops,tensors,fixture,buffers,side); _poison(buffers)
    torch.npu.synchronize(); _metadata_equal(torch,old,new,fixture)
    before_tasks={side:_bytes(buffers["tasks"]) for side,buffers in (("old",old),("new",new))}
    for side,buffers in (("old",old),("new",new)): _launch(ops,tensors,buffers,fixture,cores,route,side)
    torch.npu.synchronize()
    _check_pair(torch,ops,old,new,fixture,acceptance,cluster=ROUTES[route][2]>1)
    _assert_inputs(tensors,fixture)
    if any(_bytes(buffers["tasks"])!=before_tasks[side] for side,buffers in (("old",old),("new",new))):
        raise ReaderGateError("reader mutated prepared metadata")
    # Same real task table, one explicit metadata error; no host fallback.
    old["tasks"][0,10]=1;new["tasks"][0,10]=1
    for side,buffers in (("old",old),("new",new)):
        _poison(buffers);_launch(ops,tensors,buffers,fixture,cores,route,side)
    torch.npu.synchronize();_check_pair(torch,ops,old,new,fixture,acceptance,error_code=1)
    for side,buffers in (("old",old),("new",new)):
        _prepare(torch,ops,tensors,fixture,buffers,side);_poison(buffers)
    tensors["request_map"][0,0]=_geometry(fixture)["capacity"]
    for side,buffers in (("old",old),("new",new)):_launch(ops,tensors,buffers,fixture,cores,route,side)
    torch.npu.synchronize();_candidate_window_fault(torch,old,new,fixture,1)
    tensors["request_map"].copy_(fixture["inputs"]["request_map"])
    _assert_inputs(tensors,fixture)
    return {"case":spec.name,"route":route,"status":"passed","device_completion":"passed",
        "operators":list(ROUTES[route][:2]),"all_partial_lse_status_bits":"passed","independent_oracle":"passed",
        "oracle_scope":"complete_three_sources","metadata_error_rejection":"passed","invalid_map_rejection":"passed","metadata_parity":"passed",
        "input_immutable":True,"live_window_exercised":True,"cluster_exercised":ROUTES[route][2]>1,
        "input_sha256":_fingerprint(_device_inputs(fixture)),"shape":list(fixture["shape"])}


def run_graph(torch, ops, device, cores, acceptance, route, spec):
    _require_npu(torch,device)
    fixture=_fixture(torch,spec,cores);g=_geometry(fixture)
    tensors={name:value.to(device) for name,value in _device_inputs(fixture).items()}
    old,new=(_allocate(torch,fixture,device,cores,route) for _ in range(2))
    pointers={name:value.data_ptr() for name,value in tensors.items()}
    pointers.update({side+"/"+name:value.data_ptr() for side,buf in (("old",old),("new",new)) for name,value in buf.items()})
    tensors["old_slots"].fill_(-1);tensors["new_slots"].fill_(-1)
    tensors["request_map"][:,0].fill_(-1);tensors["request_map"][:,1].zero_()
    for side,buf in (("old",old),("new",new)):
        _prepare(torch,ops,tensors,fixture,buf,side);_poison(buf);_launch(ops,tensors,buf,fixture,cores,route,side)
    torch.npu.synchronize();_metadata_equal(torch,old,new,fixture,padding=True)
    _check_pair(torch,ops,old,new,fixture,acceptance,padding=True)
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph,capture_error_mode="thread_local",auto_dispatch_capture=True):
        _launch(ops,tensors,new,fixture,cores,route,"new")
    graph.replay();torch.npu.synchronize();_check_pair(torch,ops,old,new,fixture,acceptance,padding=True)
    stages=[]
    changed=_relocate(_fixture(torch,replace(spec,seed_offset=313),cores))
    for current in (fixture,changed):
        _copy_inputs(tensors,current)
        for side,buf in (("old",old),("new",new)):
            _prepare(torch,ops,tensors,current,buf,side);_poison(buf)
        torch.npu.synchronize();_metadata_equal(torch,old,new,current)
        _launch(ops,tensors,old,current,cores,route,"old")
        graph.replay();torch.npu.synchronize()
        _check_pair(torch,ops,old,new,current,acceptance,cluster=ROUTES[route][2]>1)
        replay={key:new[key].clone() for key in ("partial","lse","status")}
        _poison(new);_launch(ops,tensors,new,current,cores,route,"new");torch.npu.synchronize()
        for name,value in replay.items():_equal(new[name],value,"changed eager/graph "+name)
        _assert_inputs(tensors,current)
        stages.append({"status":"passed","input_sha256":_fingerprint(_device_inputs(current)),
                       "output_sha256":_fingerprint(replay),"eager_graph_bits":"passed"})
    if stages[0]["input_sha256"]==stages[1]["input_sha256"] or stages[0]["output_sha256"]==stages[1]["output_sha256"]:
        raise ReaderGateError("changed seed/map graph replay did not change input/output")
    # The old window tags are invalidated independently. Candidate retains all
    # old sidecar bytes and changes only the requested epoch: both must reject.
    tensors["old_tags"].fill_(-1)
    tensors["request_map"][:g["requests"],1].add_(1)
    for buf in (old,new):_poison(buf)
    _launch(ops,tensors,old,changed,cores,route,"old");graph.replay();torch.npu.synchronize()
    _check_pair(torch,ops,old,new,changed,acceptance,error_code=4)
    final={name:value.data_ptr() for name,value in tensors.items()}
    final.update({side+"/"+name:value.data_ptr() for side,buf in (("old",old),("new",new)) for name,value in buf.items()})
    if pointers!=final:raise ReaderGateError("graph reader input/output addresses changed")
    return {"case":spec.name,"route":route,"status":"passed","capture":"passed","replay":"passed",
        "operators":list(ROUTES[route][:2]),
        "capture_all_padding_unpublished_map":True,"same_addresses":True,"changed_map_slot_epoch_and_kv":True,
        "stale_epoch_rejection":"passed","stages":stages,"shape":list(fixture["shape"]),
        "graph_scope":"standalone_reader_not_full_service"}


def validate_evidence(report, *, expected_cores=None, c4_member_pipeline=False):
    if report.get("first_error"):raise ReaderGateError("reader report retains an execution failure")
    plan=case_plan(c4_member_pipeline=c4_member_pipeline)
    expected={spec.name:route for route,spec in plan}
    accuracy,graph=report.get("accuracy",{}),report.get("graph",{})
    cores=accuracy.get("cores")
    if (type(cores) is not int or not 1<=cores<=32 or graph.get("cores")!=cores or
            type(graph.get("cores")) is not int or
            (expected_cores is not None and (type(expected_cores) is not int or cores!=expected_cores))):
        raise ReaderGateError("reader evidence differs from the current measured core count")
    declared={spec.name:expected_shape(spec,cores) for _route,spec in plan}
    if (any(accuracy.get(key)!="passed" for key in ("device_completion","independent_oracle")) or
            any(graph.get(key)!="passed" for key in ("capture","replay","stale_epoch_rejection")) or
            graph.get("same_address_changed_input") is not True):
        raise ReaderGateError("reader completion/graph aggregate evidence is incomplete")
    for section in (accuracy,graph):
        rows=section.get("cases",[])
        if (section.get("status")!="passed" or len(rows)!=len(expected) or
                {row.get("case") for row in rows}!=set(expected)):
            raise ReaderGateError("complete reader gate requires exactly all six readers")
        for row in rows:
            if row.get("status")!="passed" or row.get("route")!=expected[row["case"]]:
                raise ReaderGateError("reader route/evidence is incomplete")
            shape=row.get("shape")
            if (not isinstance(shape,list) or any(type(value) is not int for value in shape) or
                    shape!=declared[row["case"]]):
                raise ReaderGateError("reader shape does not match the declared case and measured cores")
    for row in accuracy["cases"]:
        if (row.get("oracle_scope")!="complete_three_sources" or row.get("operators")!=list(ROUTES[row["route"]][:2]) or
                any(row.get(key)!="passed" for key in ("device_completion","all_partial_lse_status_bits",
                    "independent_oracle","metadata_error_rejection","invalid_map_rejection","metadata_parity")) or
                row.get("input_immutable") is not True or row.get("live_window_exercised") is not True or
                row.get("cluster_exercised")!=(ROUTES[row["route"]][2]>1) or not _is_sha256(row.get("input_sha256"))):
            raise ReaderGateError("reader accuracy evidence missing or scoped incorrectly")
    for row in graph["cases"]:
        if (row.get("operators")!=list(ROUTES[row["route"]][:2]) or
                any(row.get(key)!="passed" for key in ("capture","replay","stale_epoch_rejection")) or
                any(row.get(key) is not True for key in ("capture_all_padding_unpublished_map","same_addresses","changed_map_slot_epoch_and_kv")) or
                len(row.get("stages",[]))!=2 or any(stage.get("eager_graph_bits")!="passed" or stage.get("status")!="passed" for stage in row["stages"]) or
                any(len({stage.get(key) for stage in row["stages"]})!=2 for key in ("input_sha256","output_sha256")) or
                any(not _is_sha256(stage.get(key))
                    for stage in row["stages"] for key in ("input_sha256","output_sha256"))):
            raise ReaderGateError("reader changed-map graph evidence incomplete")


def run_reader_gate(torch, ops, device, *, cores, acceptance, manifest_path=None, evidence=None, checkpoint=None,
                    c4_member_pipeline=False):
    report={} if evidence is None else evidence
    report.update(accuracy={"status":"running","cases":[],"cores":cores},
                  graph={"status":"running","cases":[],"cores":cores},performance="not_run",
                  c4_member_pipeline=c4_member_pipeline)
    try:
        _require_npu(torch,device)
        if type(cores) is not int or not 1<=cores<=32: raise ReaderGateError("invalid measured Cube count")
        policy=acceptance["fused_attention"]
        if (acceptance.get("frozen_before_measurement") is not True or policy.get("atol")!=0.005 or policy.get("rtol")!=0.005):
            raise ReaderGateError("frozen 0.005 reader oracle policy required")
        from oscar_ascend.ops.loader import require_capabilities
        plan=case_plan(c4_member_pipeline=c4_member_pipeline)
        require_capabilities(required_capabilities(c4_member_pipeline=c4_member_pipeline),manifest_path)
        for route,spec in plan:
            report["active_case"]=spec.name;report["active_phase"]="accuracy"
            report["accuracy"]["cases"].append(run_case(torch,ops,device,cores,policy,route,spec))
            if checkpoint:checkpoint(report)
        report["accuracy"].update(status="passed",device_completion="passed",independent_oracle="passed")
        for route,spec in plan:
            report["active_case"]=spec.name;report["active_phase"]="graph"
            report["graph"]["cases"].append(run_graph(torch,ops,device,cores,policy,route,spec))
            if checkpoint:checkpoint(report)
        report["graph"].update(status="passed",capture="passed",replay="passed",same_address_changed_input=True,stale_epoch_rejection="passed")
        validate_evidence(report,c4_member_pipeline=c4_member_pipeline)
        report["status"]="passed"
    except Exception as error:
        import traceback
        traceback.print_exc()
        report["status"]="failed";report["first_error"]=repr(error)
        report["traceback"]=traceback.format_exc()
        if report["accuracy"]["status"]!="passed":report["accuracy"]["status"]="failed"
        report["graph"]["status"]=("failed" if report.get("active_phase")=="graph" else "not_run")
    if checkpoint:checkpoint(report)
    return report
