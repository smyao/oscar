"""Archive #126-129/#142/#145/#151-155: host contracts, never NPU evidence.

No fake operator run is labelled passed hardware. Synthetic report dictionaries
exercise only the completeness validator; real launch functions reject CPU.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as S

import pytest
import torch

from tools import request_window_reader_gate as gate


def complete_schema(*, c4_member_pipeline=False):
    accuracy=[];graphs=[]
    for route,spec in gate.case_plan(c4_member_pipeline=c4_member_pipeline):
        accuracy.append({"case":spec.name,"route":route,"status":"passed",
            "shape":gate.expected_shape(spec,20),"input_sha256":"a"*64,
            "operators":list(gate.ROUTES[route][:2]),"oracle_scope":"complete_three_sources",
            **{name:"passed" for name in ("device_completion","all_partial_lse_status_bits",
                "independent_oracle","metadata_error_rejection","invalid_map_rejection","metadata_parity")},
            "input_immutable":True,"live_window_exercised":True,
            "cluster_exercised":gate.ROUTES[route][2]>1})
        graphs.append({"case":spec.name,"route":route,"status":"passed","capture":"passed",
            "operators":list(gate.ROUTES[route][:2]),
            "shape":gate.expected_shape(spec,20),
            "replay":"passed","stale_epoch_rejection":"passed","capture_all_padding_unpublished_map":True,
            "same_addresses":True,"changed_map_slot_epoch_and_kv":True,
            "stages":[{"status":"passed","input_sha256":"1"*64,"output_sha256":"3"*64,"eager_graph_bits":"passed"},
                      {"status":"passed","input_sha256":"2"*64,"output_sha256":"4"*64,"eager_graph_bits":"passed"}]})
    return {"accuracy":{"status":"passed","cases":accuracy,"cores":20,"device_completion":"passed","independent_oracle":"passed"},
            "graph":{"status":"passed","cases":graphs,"cores":20,"capture":"passed","replay":"passed",
                     "stale_epoch_rejection":"passed","same_address_changed_input":True}}


def test_every_reader_and_all_error_graph_contracts_are_mandatory():
    report=complete_schema();gate.validate_evidence(report)
    for section in ("accuracy","graph"):
        for index in range(6):
            changed=deepcopy(report);changed[section]["cases"].pop(index)
            with pytest.raises(gate.ReaderGateError,match="all six"):
                gate.validate_evidence(changed)
    for key,value in (("oracle_scope","source0_and_source1_only"),("metadata_error_rejection","pending"),
                      ("invalid_map_rejection","not_run"),("live_window_exercised",False)):
        changed=deepcopy(report);changed["accuracy"]["cases"][0][key]=value
        with pytest.raises(gate.ReaderGateError):gate.validate_evidence(changed)
    for key,value in (("stale_epoch_rejection","pending"),("same_addresses",False),
                      ("capture_all_padding_unpublished_map",False)):
        changed=deepcopy(report);changed["graph"]["cases"][0][key]=value
        with pytest.raises(gate.ReaderGateError):gate.validate_evidence(changed)


def test_cpu_cannot_run_or_pass_reader_gate(capsys):
    report=gate.run_reader_gate(torch,None,torch.device("cpu"),cores=2,acceptance={})
    assert report["status"]=="failed" and report["accuracy"]["status"]=="failed"
    assert report["graph"]["status"]=="not_run" and "real NPU" in report["first_error"]
    assert "ReaderGateError" in report["traceback"]
    assert "real NPU" in capsys.readouterr().err
    assert report["performance"]=="not_run"
    with pytest.raises(gate.ReaderGateError):
        gate.run_case(torch,None,torch.device("cpu"),2,{},*gate.case_plan()[0])


@pytest.mark.parametrize('damage',['missing_shape','wrong_tokens','wrong_core','missing_input_hash','invalid_input_hash','missing_actual_core'])
def test_reader_evidence_rejects_unknown_or_different_geometry_and_identity(damage):
    report=complete_schema();row=report['accuracy']['cases'][2]
    if damage=='missing_shape':row.pop('shape')
    elif damage=='wrong_tokens':row['shape'][0]=8
    elif damage=='wrong_core':report['graph']['cases'][0]['shape'][-1]=2
    elif damage=='missing_input_hash':row.pop('input_sha256')
    elif damage=='invalid_input_hash':row['input_sha256']='unknown'
    else:report['accuracy'].pop('cores')
    with pytest.raises(gate.ReaderGateError):gate.validate_evidence(report,expected_cores=20)
    with pytest.raises(gate.ReaderGateError,match='core count'):
        gate.validate_evidence(complete_schema(),expected_cores=16)


def test_all_six_host_datasets_use_complete_independent_oracle_and_legal_shapes():
    torch.set_num_threads(min(4,torch.get_num_threads()))
    for route,spec in gate.case_plan():
        fixture=gate.inputs.make_case(torch,spec);g=gate._geometry(fixture)
        assert list(fixture['shape'])==gate.expected_shape(spec,2)
        assert fixture["oracle_scope"]=="complete_three_sources" and not spec.suppress_current
        active=int(fixture["inputs"]["starts"][-1])
        assert bool(torch.isfinite(fixture["inputs"]["expected_lse"][:active]).all())
        assert g["hq"]==6 and g["hk"]==1 and g["new_b"]==2*g["old_b"]
        assert gate.REQUIRED_CAPABILITIES >= set(gate.ROUTES[route][:2])
        if route in {"q4","q1"}: assert g["n"]==128 and g["requests"]==32 and g["splits"]==3
        if route in {"c4","c16"}:
            assert g["n"]==21*gate.ROUTES[route][2]
            assert g["recent"]==1 and spec.contexts==(65,)


def test_graph_changes_coherent_logical_data_and_moves_epoch_without_erasing_old_tags():
    torch.set_num_threads(min(4,torch.get_num_threads()))
    route,spec=gate.case_plan()[3]
    original=gate.inputs.make_case(torch,spec)
    changed=gate.inputs.make_case(torch,replace(spec,seed_offset=313))
    relocated=gate._relocate(changed)
    assert gate._fingerprint(gate._device_inputs(original))!=gate._fingerprint(gate._device_inputs(relocated))
    assert not torch.equal(original["inputs"]["expected_output"],changed["inputs"]["expected_output"])
    g=gate._geometry(changed)
    new_slots=set(int(x) for x in relocated["inputs"]["request_map"][:g["requests"],0])
    preserved=0
    for request in range(g["requests"]):
        old_slot,old_epoch=map(int,changed["inputs"]["request_map"][request])
        new_slot,new_epoch=map(int,relocated["inputs"]["request_map"][request])
        assert new_slot!=old_slot and new_epoch==old_epoch+7
        if old_slot not in new_slots:
            assert torch.equal(relocated["inputs"]["new_tags"][old_slot],changed["inputs"]["new_tags"][old_slot])
            preserved+=1
        old_tags=changed["inputs"]["new_tags"][old_slot];valid=old_tags>=0
        assert bool(relocated["inputs"]["new_tags"][new_slot][valid].eq(
            (new_epoch<<g["bits"])|(old_tags[valid]&((1<<g["bits"])-1))).all())
    assert preserved>0
    assert torch.equal(relocated["inputs"]["expected_output"],changed["inputs"]["expected_output"])


def test_torch_binding_argument_order_and_map_request_slice_with_meta_buffers():
    for route,spec in gate.case_plan():
        fixture=gate.inputs.make_case(torch,spec);g=gate._geometry(fixture)
        tensors={name:value.to("meta") for name,value in gate._device_inputs(fixture).items()}
        # No hardware launch. Stable sentinel values make ordering observable.
        buffers={name:object() for name in ("tasks","partial","lse","status","workspace","stats")}
        old=gate.reader_arguments(tensors,buffers,fixture,2,route,"old")
        new=gate.reader_arguments(tensors,buffers,fixture,2,route,"new")
        cluster=gate.ROUTES[route][2]>1
        assert len(old)==25+int(cluster) and len(new)==28+int(cluster)
        assert new[9] is tensors["new_tags"] and new[10].shape==(g["requests"],2)
        assert new[11] is buffers["tasks"] and new[15] is buffers["workspace"]
        assert new[-4:]==(g["d"]**-.5,2,g["capacity"],g["bits"])
        assert old[-10]==g["old_b"] and new[-12]==g["new_b"]


def test_enabled_pipeline_replaces_only_c4_slot_and_launches_both_new_abis():
    old=gate.case_plan();new=gate.case_plan(c4_member_pipeline=True)
    assert len(old)==len(new)==6
    assert old[:4]==new[:4] and old[5:]==new[5:]
    route,spec=new[4]
    assert route=="c4_member_pipeline" and spec.contexts==(337,) and not spec.suppress_current
    names=("attention_cv_member_pipeline_cluster4_out",
           "attention_request_window_member_pipeline_cluster4_out")
    assert gate.ROUTES[route][:2]==names
    required=gate.required_capabilities(c4_member_pipeline=True)
    assert set(names)<=required
    assert "attention_request_window_cluster4_out" not in required
    assert not set(names).intersection(gate.REQUIRED_CAPABILITIES)
    fixture=gate._fixture(torch,spec,20)
    tensors={name:value.to("meta") for name,value in gate._device_inputs(fixture).items()}
    buffers=gate._allocate(torch,fixture,"meta",20,route)
    assert buffers["workspace"].numel()==20*2494464
    seen=[]
    class Ops:
        def __getattr__(self,name):
            assert name in names
            def call(*args):seen.append((name,args))
            return call
    for side in ("old","new"):
        gate._launch(Ops(),tensors,buffers,fixture,20,route,side)
    assert [name for name,_ in seen]==list(names)
    assert len(seen[0][1])==26 and len(seen[1][1])==29
    assert seen[1][1][15] is buffers["workspace"]
    assert seen[1][1][16] is buffers["stats"]


def test_old_c4_evidence_cannot_qualify_pipeline_selection_or_its_graph():
    with pytest.raises(gate.ReaderGateError):
        gate.validate_evidence(complete_schema(),c4_member_pipeline=True)
    report=complete_schema(c4_member_pipeline=True)
    gate.validate_evidence(report,c4_member_pipeline=True)
    for section in ("accuracy","graph"):
        wrong=deepcopy(report)
        wrong[section]["cases"][4]["operators"]=list(gate.ROUTES["c4"][:2])
        with pytest.raises(gate.ReaderGateError):
            gate.validate_evidence(wrong,c4_member_pipeline=True)


def test_pipeline_gate_requires_its_actual_capabilities_before_launch(monkeypatch,capsys):
    from oscar_ascend.ops import loader
    seen=[]
    # Deliberate rejection at the capability boundary; no operator result or
    # successful NPU execution is fabricated by this host control-flow test.
    monkeypatch.setattr(gate,"_require_npu",lambda *_:None)
    def reject(required,_manifest):
        seen.append(required)
        raise gate.ReaderGateError("missing pipeline reader")
    monkeypatch.setattr(loader,"require_capabilities",reject)
    report=gate.run_reader_gate(torch,None,None,cores=20,
        acceptance={"frozen_before_measurement":True,"fused_attention":{"atol":.005,"rtol":.005}},
        c4_member_pipeline=True)
    assert seen==[gate.required_capabilities(c4_member_pipeline=True)]
    assert report["status"]=="failed" and report["accuracy"]["cases"]==[]
    assert "missing pipeline reader" in capsys.readouterr().err
