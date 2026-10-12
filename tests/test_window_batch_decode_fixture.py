"""Archive #126/#129/#140/#145/#148/#155: host fixture and bounded-run contracts.

No CANN/NPU execution here; SDK binary evidence is collected separately.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    path=ROOT/f"csrc/experiments/window_batch_decode/{name}.py"
    spec=importlib.util.spec_from_file_location(f"window_batch_{name}_test",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


fixture=load("fixture")
runner=load("run_cpu")


def test_current_shapes_cover_ring_page_stride_and_large_cases_are_opt_in():
    cases=fixture.case_plan()
    assert {case.mode for case in cases}=={"q1","q4"}
    assert {case.heads for case in cases}=={1,2}
    assert all(case.run_default for case in cases)
    large=fixture.case_plan(True)[-2:]
    assert all(len(case.contexts)==32 and case.padded==128 and not case.run_default for case in large)
    f=fixture.make_case(torch,next(case for case in cases if case.name=="q4_hkv2_c641"))
    assert f["cpu"]["wk"].shape[1:]==(323,2,256)
    a,b=fixture._location(f,511),fixture._location(f,512)
    assert a[0]!=b[0]
    ring=fixture.make_case(torch,cases[0])
    assert fixture._location(ring,258)[1]==322
    assert fixture._location(ring,259)[1]==64
    for pos in (0,63,256,258,259,500,510):
        page,row=fixture._location(ring,pos)
        assert int(ring["cpu"]["tags"][page,row])==pos%512
    assert bool(torch.isfinite(f["cpu"]["q"]).all())
    assert sorted(f["expected"])==list(range(4))


@pytest.mark.parametrize("fault,code",[("bad_tag",4),("vnan_then_tag",4),("tag_then_vnan",2),("bad_page",1)])
def test_fault_order_fixture_preserves_current_slot_and_targets_final_window_lane(fault,code):
    case=next(case for case in fixture.case_plan() if case.fault==fault)
    data=fixture.make_case(torch,case);t=data["cpu"]
    assert t["required_status_checks"][0].tolist()==[1,0,code]
    if fault=="bad_page":
        assert int(t["table"][0,2])==-1
        assert int(t["table"][0,3])==int(t["slots"][0])//128
        assert t["required_status_checks"].tolist()==[[1,0,1],[1,1,1]]
    else:
        tag_pos=500 if fault=="tag_then_vnan" else 501
        assert int(t["tags"][fixture._location(data,tag_pos)])==-1
        if fault!="bad_tag":
            nan_pos=501 if fault=="tag_then_vnan" else 500
            page,row=fixture._location(data,nan_pos)
            assert bool(torch.isnan(t["wv"][page,row,0,0]))
            assert int(t["tags"][page,row])==nan_pos


def test_export_matches_exact_binary_shape_and_frozen_tolerance(tmp_path,monkeypatch):
    definitions=fixture.case_plan()[:2]
    monkeypatch.setattr(fixture,"case_plan",lambda *_:definitions)
    manifest=fixture.export(tmp_path,cores=2)
    for case in manifest["cases"]:
        path=tmp_path/case["name"]
        shape=list(map(int,(path/"shape.txt").read_text().split()))
        assert len(shape)==16 and shape[12]==2 and shape[7]==3
        n,hq,hk,d=shape[:4]
        assert (path/"q.bin").stat().st_size==n*hq*d*2
        assert (path/"raw_striped.bin").stat().st_size==shape[10]+shape[9]*shape[11]
        assert (path/"tolerance.txt").read_text().split()==["0.005","0.005"]
        assert (path/"required_status_checks.bin").stat().st_size==0
        assert case["input_sha256"] and manifest["npu"]=="not_run"


@pytest.mark.parametrize("failure",[None,"missing_oracle","missing_guard","sdk_child_error","timeout"])
def test_serial_runner_needs_completion_and_guards_cleans_only_its_case(tmp_path,monkeypatch,failure):
    case=tmp_path/"case";case.mkdir();(case/"q.bin").write_bytes(b"golden")
    cases=tmp_path/"cases.json"
    cases.write_text(json.dumps({"cases":[{"name":"case","mode":"q1","cores":2,
        "expected_error":False,"run_default":True}]}))
    cwd_seen=[];logs=tmp_path/"logs";logs.mkdir()
    def phase(name,command,*,cwd,log_dir,**kwargs):
        cwd_seen.append(cwd)
        assert cwd!=ROOT and cwd!=case and kwargs["timeout"]==120
        (cwd/"sdk_trace.txt").write_text("A"*12000)
        evidence={"backend":"ascendc_cpu_debug","status":"passed","baseline":"current_production_simd",
            "bitwise":True,"cores":2,"fault_order":False,"oracle":True,"input_immutable":True,"gm_guards":True}
        if failure=="missing_oracle":evidence.pop("oracle")
        if failure=="missing_guard":evidence.pop("gm_guards")
        log=logs/"phase.log"
        log.write_text(json.dumps(evidence,separators=(",",":"))+"\n"+(
            "[ERROR] debugger child failed\n" if failure=="sdk_child_error" else ""))
        return SimpleNamespace(returncode=0,timed_out=failure=="timeout",cleanup_complete=True,log=str(log))
    monkeypatch.setattr(runner,"run_phase",phase)
    result=runner.run(tmp_path/"fake_binary",cases,tmp_path/"report.json",logs)
    assert result==(0 if failure is None else 1)
    assert len(cwd_seen)==1 and not cwd_seen[0].exists()
    assert (case/"q.bin").read_bytes()==b"golden"
    trace=json.loads((logs/"00-case-trace/manifest.json").read_text())
    assert trace["total_bytes"]==12000 and trace["sdk_workdir_cleaned_after_summary"]
    assert (logs/"00-case-trace/000.preview.txt").stat().st_size<8300
