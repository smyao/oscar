"""Archive #34/#36/#94/#95/#125/#140/#142/#148/#155: host gate contracts only.

No CPU substitutes for NPU operators; these checks exercise disabled routing,
bounded case selection, exact call delegation and failure/cleanup propagation.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import observe_serve, probe_current_only as probe


def target():
    return {"experimental_current_only": True, "experimental_history_reuse": True,
        "experimental_fast_unpack": True,"experimental_mixed_cv": True,"experimental_striped_cache": True,
        "devices": [0,1,2,3], "soc_version": "ascend910b4"}


def complete_report():
    rows=[]
    for case in probe.case_plan():
        fresh,draft,slot_type,paired=case
        row={"case":probe._case_name(case),"status":"passed","first_draft":draft,
            "slot_dtype":slot_type,"rotation":"draft_identity" if draft else "nontrivial_dense_fixture",
            "independent_oracle":"passed","copy_lse_oracle":"passed","max_output_abs":0.001,
            "sampled_queries":len(probe.reuse._samples(fresh))+(4 if paired else 0),
            "real_operator_calls":{"copy_validate_current_out":1 if paired else 8,
                "rotate_clip_store_striped_out":3 if paired else 8}}
        if paired:
            row.update(ab_output="frozen_tolerance_passed",store_bitwise="passed",
                baseline="complete_current_CV" if draft else "existing_main_current_FIA_merge",
                decode_after_store={"status":"passed","reader":"production_subsequent_q1",
                    "output_bitwise":"passed","independent_oracle":"passed",
                    "subsequent_store_bitwise":"passed","sampled_queries":2,"max_output_abs":0.001})
            row["real_operator_calls"].update(attention_cv_striped_decode_out=1,attention_cv_striped_q1_out=1)
        else:
            row.update(device_event_ms=[1.,2.,3.,4.,5.],device_event_median_ms=3.,
                timing_scope="production_fresh_FIA_copy_validate_INT2_store_status_guard",
                latency_gate="not_established_no_long_CV_baseline")
        rows.append(row)
    return {"cases":rows,**{key:"passed" for key in (
        "status","precision","store_bitwise","decode_after_store","device_completion")}}


@pytest.mark.parametrize("corruption", ["empty","missing","duplicate","copy","q1","event","rotation"])
def test_report_rejects_incomplete_case_or_operator_evidence(corruption):
    report=complete_report();acceptance={"performance":{"warmup":2,"repeats":5}}
    probe.validate_case_evidence(report,target(),acceptance)
    if corruption=="empty":report["cases"]=[]
    elif corruption=="missing":report["cases"].pop()
    elif corruption=="duplicate":report["cases"][-1]=report["cases"][0]
    elif corruption=="copy":report["cases"][0]["real_operator_calls"].pop("copy_validate_current_out")
    elif corruption=="q1":report["cases"][0]["decode_after_store"]["status"]="failed"
    elif corruption=="event":report["cases"][-1]["device_event_ms"][2]=float("nan")
    elif corruption=="rotation":report["cases"][-1]["rotation"]="unknown"
    with pytest.raises(RuntimeError,match="current-only"):
        probe.validate_case_evidence(report,target(),acceptance)


def test_small_gate_covers_first_main_and_long_cases_do_not_launch_cv_ab():
    cases = probe.case_plan()
    paired = [row for row in cases if row[3]]
    assert len(paired) == 4
    assert {(r[0],r[1]) for r in paired} == {(17,False),(17,True),(385,False),(385,True)}
    assert {r[2] for r in paired} == {"int32","int64"}
    assert {r[0] for r in cases if not r[3]} == {20000,30000}
    for draft in (True,False):
        assert {r[2] for r in paired if r[1]==draft} == {"int32","int64"}


def test_witness_delegates_exact_inputs_outputs_and_errors():
    expected = object(); first = object(); calls = []
    def op(arg, *, option):
        calls.append((arg, option))
        if option:raise RuntimeError("native operator error")
        return expected
    counted=probe.CountedOps(SimpleNamespace(real=op))
    assert counted.real(first,option=False) is expected
    with pytest.raises(RuntimeError,match="native operator error"):
        counted.real(first,option=True)
    assert calls == [(first,False),(first,True)] and counted.calls["real"]==2


def test_probe_disabled_rejects_before_torch_or_fixture(tmp_path):
    config=tmp_path/"target.json";acceptance=tmp_path/"acceptance.json"
    config.write_text(json.dumps({**target(),"experimental_current_only":False}))
    acceptance.write_text("{}")
    with pytest.raises(RuntimeError,match="explicit experimental_current_only=true"):
        probe.probe(config,acceptance)
    with pytest.raises(RuntimeError,match="experimental_striped_cache=true"):
        probe._configuration({**target(),"experimental_striped_cache":False},True)


@pytest.mark.parametrize("tokens", [21, 389, 20000, 30000])
def test_independent_runtime_capacity_is_fixture_sized_and_does_not_mutate_target(tokens):
    source = {**target(), "use_native_token_budget": True,
              "compilation_config": {"cudagraph_capture_sizes": [128, 512], "cudagraph_mode": "FULL_DECODE_ONLY"}}
    source.pop("max_num_batched_tokens", None)
    before = json.dumps(source, sort_keys=True)
    config = probe._configuration(source, True, probe_tokens=tokens)
    assert config["max_num_batched_tokens"] == tokens
    assert config["use_native_token_budget"] is False
    assert config["compilation_config"]["cudagraph_capture_sizes"] == []
    assert json.dumps(source, sort_keys=True) == before
    with pytest.raises(ValueError, match="positive integer"):
        probe._configuration(source, True, probe_tokens=0)


def test_probe_real_npu_missing_fails_without_cpu_substitution(tmp_path, monkeypatch):
    import sys
    import torch
    config=tmp_path/"target.json";acceptance=tmp_path/"acceptance.json"
    config.write_text(json.dumps(target()));acceptance.write_text('{"frozen_before_measurement":true}')
    monkeypatch.setitem(sys.modules,"torch_npu",SimpleNamespace())
    monkeypatch.setattr(torch,"npu",SimpleNamespace(is_available=lambda:False),raising=False)
    monkeypatch.setattr(probe,"run_case",lambda *args:pytest.fail("NPU absent must never execute a case"))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES","0,1,2,3")
    with pytest.raises(RuntimeError,match="requires a real NPU"):
        probe.probe(config,acceptance)


def test_main_prints_real_traceback_and_keeps_nonzero_result(tmp_path, monkeypatch, capsys):
    def failure(*_):raise RuntimeError("device error evidence")
    monkeypatch.setattr(probe,"probe",failure)
    output=tmp_path/"probe.json"
    assert probe.main(["--output",str(output)])==2
    captured=capsys.readouterr()
    assert "Traceback" in captured.err and "device error evidence" in captured.err
    assert "CURRENT_ONLY_RESULT" in captured.out
    assert json.loads(output.read_text())["status"]=="failed"


def test_observer_disabled_does_not_enter_phase_or_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(observe_serve,"read_npu_resources",lambda *a,**k:pytest.fail("disabled gate"))
    monkeypatch.setattr(observe_serve,"_phase",lambda *a,**k:pytest.fail("disabled phase"))
    observe_serve._current_only_gate(tmp_path/"config",{}, {},tmp_path,{})
    with pytest.raises(ValueError,match="explicit boolean"):
        observe_serve._current_only_gate(tmp_path/"config",{"experimental_current_only":1},{},tmp_path,{})


@pytest.mark.parametrize("failure", [None,"precision","release","phase"])
def test_observer_requires_signed_npu_evidence_and_preserves_phase_rc(tmp_path,monkeypatch,failure):
    from oscar_ascend.ops import loader
    config={**target(),"phase_timeout_seconds":60,"shutdown_timeout_seconds":2}
    manifest={"signature":"current-only-signature","sha256":{"binary":"digest"}}
    monkeypatch.setattr(loader,"validate_build_artifacts",lambda *_:manifest)
    monkeypatch.setattr(observe_serve,"read_npu_resources",lambda *a,**k:{})
    release_calls=[]
    def release(*a,**k):
        release_calls.append(True)
        if failure=="phase":raise RuntimeError("secondary cleanup failure")
        return {"status":"failed" if failure=="release" else "passed"}
    monkeypatch.setattr(observe_serve,"wait_for_release",release)
    phase_failure=RuntimeError("real phase failure")
    phase_failure.returncode=37
    def phase(name,command,**kwargs):
        assert name=="current-only-npu" and "tools.probe_current_only" in command
        if failure=="phase":raise phase_failure
        report=complete_report()
        report.update(artifact_signature=manifest["signature"],artifact_sha256=manifest["sha256"])
        if failure=="precision":report["precision"]="failed"
        Path(command[command.index("--output")+1]).write_text(json.dumps(report))
    monkeypatch.setattr(observe_serve,"_phase",phase)
    status={"phases":[]}
    if failure:
        with pytest.raises(RuntimeError) as caught:
            observe_serve._current_only_gate(tmp_path/"config",config,{},tmp_path,status)
        if failure=="phase":
            assert caught.value is phase_failure and caught.value.returncode==37
        assert "current_only_gate" not in status
    else:
        observe_serve._current_only_gate(tmp_path/"config",config,{},tmp_path,status)
        assert status["current_only_gate"]["status"]=="passed"
        assert status["current_only_gate"]["graph_acceptance"]=="not_established_eager_gate"
    assert release_calls==[True]


def test_observer_wires_enabled_gate_before_service_and_failure_blocks_it(tmp_path,monkeypatch):
    source=json.loads((observe_serve.ROOT/"configs/target.json").read_text())
    source["experimental_current_only"]=True
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps({'architectures':['Qwen3_5ForConditionalGeneration']}))
    source['model']=str(model)
    config=tmp_path/"config.json";config.write_text(json.dumps(source))
    # Isolate the one gate under test; unrelated newer NPU gates must not
    # launch resource observers in this host-only wiring regression.
    for name in ("_preflight",)+tuple(name for name in dir(observe_serve)
                 if name.startswith('_') and name.endswith('_gate') and name!='_current_only_gate'):
        monkeypatch.setattr(observe_serve,name,lambda *a,**k:None)
    entered=[]
    def gate(*args):
        entered.append(args[1]["experimental_current_only"])
        error=RuntimeError("NPU gate failed");error.returncode=19;error.phase="current-only-npu"
        raise error
    monkeypatch.setattr(observe_serve,"_current_only_gate",gate)
    monkeypatch.setattr(observe_serve,"managed_server",lambda *a,**k:pytest.fail("failed gate launched service"))
    monkeypatch.setattr(observe_serve,"_terminal",lambda *a,**k:None)
    assert observe_serve.run(config,tmp_path/"logs","candidate")==19
    assert entered==[True]
    saved=json.loads((tmp_path/"logs/status.json").read_text())
    assert saved["failed_phase"]=="current-only-npu" and saved["returncode"]==19


def test_bf16_current_only_requires_new_readers_without_timing():
    report=complete_report();config={**target(),'experimental_bf16_attention':True}
    for row in report['cases']:
        row['performance']='not_run_user_requested'
        calls=row['real_operator_calls']
        if 'ab_output' in row:
            row['ab_output']='separate_BF16_native_contracts_passed'
            calls['attention_cv_bf16_decode_out']=calls.pop('attention_cv_striped_decode_out')
            calls['attention_cv_bf16_decode_q1_out']=calls.pop('attention_cv_striped_q1_out')
        else:
            calls['copy_validate_current_out']=calls['rotate_clip_store_striped_out']=1
            row.pop('device_event_ms');row.pop('device_event_median_ms')
    policy={'performance':{'warmup':2,'repeats':5}}
    probe.validate_case_evidence(report,config,policy)
    report['cases'][0]['real_operator_calls'].pop('attention_cv_bf16_decode_q1_out')
    with pytest.raises(RuntimeError):probe.validate_case_evidence(report,config,policy)
