"""Archive #126/#145/#151/#153-155: failure/evidence/ABI host contracts only.

No mocked numerical success is treated as real NPU or graph evidence.
"""
import json
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from tools import probe_request_windows as gate


def test_actual_reader_handoff_records_one_measured_core_query_before_failure():
    """Execute only the real handoff statements; no device success is mocked."""
    tree=ast.parse(Path(gate.__file__).read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='probe')
    index=next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign) and
               any(isinstance(target,ast.Name) and target.id=='reader_cores' for target in node.targets))
    statements=function.body[index:index+3]
    assert isinstance(statements[-1],ast.Assign) and isinstance(statements[-1].value,ast.Call)
    assert statements[-1].value.func.id=='run_reader_gate'
    calls=[];report={'device':{}};torch_marker=object();target_marker={'experimental_c4_member_pipeline':True}
    class StopBeforeDeviceWork(Exception):pass
    def measured(torch,target):
        assert torch is torch_marker and target is target_marker
        calls.append('measured');return 20
    def reader(torch,ops,device,*,cores,**kwargs):
        assert calls==['measured'] and cores==20
        assert report['device']['cube_cores']==20
        assert kwargs['c4_member_pipeline'] is True
        raise StopBeforeDeviceWork
    scope={'_core_count':measured,'run_reader_gate':reader,'torch':torch_marker,
           'target':target_marker,'ops':object(),'device':object(),'report':report,
           'acceptance':{},'manifest_path':Path('unused'),'reader_checkpoint':lambda _value:None}
    with pytest.raises(StopBeforeDeviceWork):
        exec(compile(ast.Module(body=statements,type_ignores=[]),gate.__file__,'exec'),scope)
    assert calls==['measured'] and report=={'device':{'cube_cores':20}}


def test_gate_plan_uses_production_d256_bf16_and_legal_segments():
    plan=gate.case_plan()
    assert len({case["name"] for case in plan})==len(plan)
    assert all(case["dim"]==256 and case["dtype_name"]=="bf16" and case["segment"]%128==0 for case in plan)
    assert {"segment_boundary","physical_page_boundary","reordered_hkv2","rollback_0_to_3",
            "numeric_and_address_faults","bad_epoch","positive_qstarts_origin","active_tail"}.issubset({x["name"] for x in plan})


def test_fixture_recreates_legal_raw_views_and_sliced_request_map():
    fixture=gate.make_fixture(torch,gate.case_plan()[0])
    s=fixture["shape"]
    assert s["key_stride"]%16==0 and s["value_stride"]%16==0
    assert fixture["cpu"]["mapping"].shape==(s["requests"],2)
    assert s["requests"]<s["capacity"]
    assert fixture["cpu"]["rk"].is_contiguous()
    assert fixture["cpu"]["rv"].is_contiguous()
    wk=gate._raw_view(torch,fixture["initial"]["raw_key"],s["capacity"],323,s["heads"],256,s["key_stride"])
    assert wk.stride(0)%16==0 and wk.dtype==torch.bfloat16
    assert torch.isnan(wk).all()
    assert fixture["expected"]["window_status"].shape==(s["capacity"],)


def test_store_argv_matches_checked_binding_and_two_status_outputs():
    names=("key","value","rk","rv","slots","positions","qstarts","mapping")
    tensors={name:object() for name in names}
    buffers={name:object() for name in ("packed","wk","wv","window_tags","status","window_status")}
    shape={"segment_tokens":2816,"blocks":4,"offset":64,"page_stride":393216,"sink":64,
           "recent_with_spec":259,"key_clip":0.75,"value_clip":0.95,"hadamard":1,"capacity":128,"position_bits":18}
    args=gate.store_arguments(tensors,buffers,shape)
    assert len(args)==25
    assert args[:8]==tuple(tensors[x] for x in names)
    assert args[12] is buffers["status"] and args[13] is buffers["window_status"]
    assert args[14:]==(5632,4,64,393216,64,259,0.75,0.95,True,128,18)


def test_no_npu_cannot_enter_operator_or_graph_gate(monkeypatch):
    from tools import probe_native_current_fia as native
    monkeypatch.setattr(native,"_select_target_npu",lambda target:None)
    monkeypatch.setitem(sys.modules,"torch_npu",ModuleType("torch_npu"))
    monkeypatch.setattr(torch,"npu",SimpleNamespace(is_available=lambda:False),raising=False)
    with pytest.raises(gate.RequestWindowGateError,match="real target NPU"):
        gate._require_npu({"devices":[0,1,2,3],"soc_version":"ascend910b4"})
    with pytest.raises(gate.RequestWindowGateError,match="real NPU device"):
        gate.run_store_case(torch,None,None,torch.device("cpu"))
    with pytest.raises(gate.RequestWindowGateError,match="real NPU graph"):
        gate.run_store_graph(torch,None,torch.device("cpu"))


def test_byte_comparator_detects_nan_payload_and_signed_zero_changes():
    a=torch.tensor([float("nan"),-0.0],dtype=torch.bfloat16)
    gate._equal(torch,a,a.clone(),"unchanged host byte helper")
    b=a.clone();b.view(torch.int16)[0]+=1
    with pytest.raises(gate.RequestWindowGateError,match="differs"):
        gate._equal(torch,b,a,"nan payload")
    b=a.clone();b[1]=0.0
    with pytest.raises(gate.RequestWindowGateError,match="differs"):
        gate._equal(torch,b,a,"signed zero")


def test_cli_preserves_failure_trace_partial_evidence_and_argv(monkeypatch,tmp_path,capsys):
    config,acceptance,manifest,output=(tmp_path/name for name in ("target.json","acceptance.json","manifest.json","result.json"))
    calls=[]
    def fail(cfg,acc,man,*,report,checkpoint):
        calls.append((cfg,acc,man));report["active_phase"]="store_accuracy"
        report["active_case"]="segment_boundary"
        report["accuracy"]["store"]["cases"].append({"case":"segment_boundary","status":"failed"})
        checkpoint(report)
        raise RuntimeError("sentinel real-device failure")
    monkeypatch.setattr(gate,"probe",fail)
    rc=gate.main(["--config",str(config),"--acceptance",str(acceptance),"--manifest",str(manifest),"--output",str(output)])
    assert rc==2 and calls==[(config,acceptance,manifest)]
    report=json.loads(output.read_text())
    assert report["status"]=="failed"
    assert report["accuracy"]["store"]["status"]=="failed"
    assert report["accuracy"]["store"]["cases"][0]["status"]=="failed"
    assert report["graph"]["store"]["status"]=="not_run"
    assert report["accuracy"]["reader"]["status"]=="pending"
    assert report["performance"]["status"]=="not_run"
    captured=capsys.readouterr()
    assert captured.out.count("REQUEST_WINDOW_GATE ")==1
    assert "RuntimeError: sentinel real-device failure" in captured.err


def test_pending_reader_cli_is_nonzero_and_never_claims_performance(monkeypatch,tmp_path):
    # Control flow only: no operator outputs or successful NPU math are mocked.
    monkeypatch.setattr(gate,"probe",lambda *a,**k:gate.empty_report())
    output=tmp_path/"pending.json"
    assert gate.main(["--output",str(output)])==2
    report=json.loads(output.read_text())
    assert report["status"]=="pending"
    assert report["device_completion"]=="not_run"
    assert report["accuracy"]["reader"]["status"]=="pending"
    assert report["performance"]["status"]=="not_run"


def test_incomplete_evidence_cannot_be_promoted_by_passed_status_strings():
    report=gate.empty_report();report["status"]="passed";report["device_completion"]="passed"
    report["accuracy"]["store"]["status"]="passed"
    report["graph"]["store"].update(status="passed",capture="passed",replay="passed")
    assert gate.final_status(report)=="failed"
    report["artifact_signature"]="claimed";report["artifact_sha256"]={"claimed":"digest"};report["device"]={"logical":"npu:0"}
    assert gate.final_status(report)=="failed"  # Missing all per-case oracle/status evidence.


def test_unfrozen_policy_fails_before_any_device_work(monkeypatch,tmp_path):
    target=tmp_path/"target.json";target.write_text("{}")
    acceptance=tmp_path/"acceptance.json";acceptance.write_text(json.dumps({"frozen_before_measurement":False,"pack_unpack":{"exact":True}}))
    monkeypatch.setattr(gate,"_require_npu",lambda target:pytest.fail("must fail before touching device"))
    with pytest.raises(gate.RequestWindowGateError,match="frozen exact pack"):
        gate.probe(target,acceptance,tmp_path/"absent.json")


def test_invalid_reader_selection_fails_before_any_device_work(monkeypatch,tmp_path):
    target=tmp_path/"target.json";target.write_text(json.dumps({"experimental_c4_member_pipeline":"true"}))
    monkeypatch.setattr(gate,"_require_npu",lambda _target:pytest.fail("must reject selection before device work"))
    with pytest.raises(gate.RequestWindowGateError,match="explicit boolean"):
        gate.probe(target,tmp_path/"unused-acceptance.json",tmp_path/"unused-manifest.json")
