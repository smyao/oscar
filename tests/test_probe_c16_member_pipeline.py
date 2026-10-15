"""Archive #126/#145/#148/#150/#154: stable-C16/member-pipeline target probe receipts, no NPU claims."""

import copy
from dataclasses import replace
import json
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from tools import probe_c16_member_pipeline as probe


def evidence():
    detail = {"context": 337, "query_tokens": 672, "recent_tokens": 256, "sink_tokens": 64,
              "head_dim": 256, "eligible_clusters": 1, "grouped_leaders": 16,
              "original_schedule_skips": 16, "source0_leaders": 32,
              "shared_history_kv_tiles": 2, "rounds_per_cluster": 2,
              "oracle": "all_672_queries_passed", "task_sha256": "initial-tasks"}
    states = [dict(detail, state=name, context=context, task_sha256=task,
                   packed_sha256=raw, rounds_per_cluster=rounds,
                   shared_history_kv_tiles=rounds)
              for name, context, task, raw, rounds in (
                  ("captured_inputs", 337, "initial-tasks", "initial-raw", 2),
                  ("changed_q_qr", 337, "initial-tasks", "initial-raw", 2),
                  ("changed_context_tasks_packed_meta", 593, "new-tasks", "new-raw", 3),
                  ("restored_context_tasks_packed_meta", 337, "initial-tasks", "initial-raw", 2))]
    graph = {"capture": "passed", "replay": "passed", "same_addresses": True,
             "q_qr_changed": True, "task_metadata_changed": True,
             "packed_meta_changed": True, "independent_oracle": "passed",
             "graph_eager_bitwise": "passed", "mature_multikv_in_every_state": True,
             "initial_inputs_restored": True, "initial_output_restored_bitwise": True,
             "states": states}
    timing = {"baseline_ms": 10., "candidate_ms": 8., "ratio": .8, "speedup": 1.25,
              "minimum_speedup": 1.0, "gate": "passed",
              "samples_ms": {probe.BASELINE: [10.] * 5, probe.CANDIDATE: [8.] * 5},
              "scope": "complete_stable_C16_vs_member_pipeline_history_window_CV_merge_BF16_copy_device_events"}
    rows = [{"case": name, "status": "passed", "precision": "partial_lse_status_clusterStats_bitwise",
             "detail": dict(detail, oracle="all_672_queries_passed" if i == 0 else "expected_invalid_metadata_status"),
             "graph": graph if i == 0 else None, "timing": timing if i == 0 else None}
            for i, name in enumerate(probe.CASE_NAMES)]
    next(row for row in rows if row['case']==probe.QR_CASE_NAME)["detail"].update(oracle="late_member_qr_isolation_passed", qr_fault={
        "query":651,"head":0,"leader":651,"cluster_anchor":336,"member":15,"lane":0,"code":2,
        "history_rows":64,"flatten_row_first":3906,"flatten_row_last":3969,
        "unaffected_rows":3968,"nonzero_task_lanes":1,"window_current_preserved":True,
        "unaffected_oracle":"passed","merge_failure":"zero_output_negative_infinity_lse_code2"})
    for row in rows:
        if row['case'] in probe.LONG_CONTEXTS:
            context=probe.LONG_CONTEXTS[row['case']]
            rounds=(context-probe.reuse.SINK+255)//256
            row['detail'].update(context=context,rounds_per_cluster=rounds,
                                 shared_history_kv_tiles=rounds,oracle='all_672_queries_passed')
            queries=list(probe.reuse._samples(672))
            row['detail']['reference_anchor_check']={'status':'bitwise_passed',
                'domain':'history_window_current','queries':queries,'expected_count':len(queries),
                'checked_count':len(queries),'bit_differences':[]}
            row['timing']=copy.deepcopy(timing)
    return {"status": "passed", "baseline_operator": probe.BASELINE,
            "candidate_operator": probe.CANDIDATE, "precision": "passed", "error_domain": "passed",
            "graph_capture": "passed", "graph_replay": "passed", "performance": "passed",
            "device_completion": "passed", "all_tasks": 2016,
            "source2": "production_prefill_suppressed", "cases": rows}


def test_receipt_rejects_c4_short_or_incomplete_graph_evidence():
    complete = evidence()
    assert probe.validate_evidence(complete) == []
    for mutation in ("c4", "missing_graph", "missing_tasks", "no_task_change",
                     "no_raw_change", "missing_oracle", "missing_round", "timing_sample", "not_restored", "recent1", "pipeline_denominator"):
        report = copy.deepcopy(complete)
        row = report["cases"][0]
        if mutation == "c4":
            row["graph"]["states"][0]["grouped_leaders"] = 4
        elif mutation == "missing_graph":
            row["graph"] = None
        elif mutation == "missing_tasks":
            report["all_tasks"] = 1008
        elif mutation == "no_task_change":
            row["graph"]["states"][2]["task_sha256"] = "initial-tasks"
        elif mutation == "no_raw_change":
            row["graph"]["states"][2]["packed_sha256"] = "initial-raw"
        elif mutation == "missing_oracle":
            report["cases"][1]["detail"].pop("oracle")
        elif mutation == "missing_round":
            row["graph"]["states"][2]["rounds_per_cluster"] = 2
        elif mutation == "not_restored":
            row["graph"]["states"][3]["task_sha256"] = "stale-new-tasks"
        elif mutation == "recent1":
            row["detail"]["recent_tokens"] = 1
        elif mutation == "pipeline_denominator":
            report["baseline_operator"] = "attention_cv_member_pipeline_cluster16_out"
        else:
            row["timing"]["samples_ms"][probe.BASELINE].pop()
        assert probe.validate_evidence(report), mutation


def test_default_pipeline_and_research_b1_have_distinct_source_bound_candidates():
    from tools import c16_probe_oracle as oracle
    from tools import probe_c16_l1_reuse as b1
    from test_probe_c16_l1_reuse import evidence as b1_evidence
    assert probe.BASELINE==b1.BASELINE=='attention_cv_window_range_cluster16_out'
    assert probe.CANDIDATE=='attention_cv_member_pipeline_cluster16_out'
    assert b1.CANDIDATE=='attention_cv_l1_reuse_cluster16_out'
    assert probe.WORKSPACE=={probe.BASELINE:4997120,probe.CANDIDATE:5652480}
    assert b1.WORKSPACE=={b1.BASELINE:4997120,b1.CANDIDATE:7487488}
    assert probe._oracle is b1._oracle is oracle._oracle
    assert probe._long_oracle is b1._long_oracle is oracle._long_oracle
    assert probe._shared_kv_attention is b1._shared_kv_attention is oracle._shared_kv_attention
    assert 'tools/c16_probe_oracle.py' in probe.SOURCE_PATHS
    assert 'tools/c16_probe_oracle.py' in b1.SOURCE_PATHS
    assert not any('l1_reuse' in path for path in probe.SOURCE_PATHS)
    assert probe.validate_evidence(b1_evidence())
    assert b1.validate_evidence(evidence())


@pytest.mark.parametrize("position,lane", [(80, 1), (96, 0), (336, 1)])
def test_fault_oracle_retains_old_interleaved_16_row_lane(position, lane):
    tasks = torch.zeros(5, 16, dtype=torch.int64)
    tasks[:, 1] = 21
    tasks[:, 3] = 64
    tasks[:, 4] = 337
    tasks[1, 1] = 0
    tasks[2, 7] = 1
    tasks[3, 4] = position
    tasks[4, 3] = position + 1
    expected = probe.expected_fault_status(torch, tasks, position)
    assert torch.nonzero(expected).tolist() == [[0, lane]]
    assert expected[0, lane] == 3


def test_fixed_capacity_changes_context_without_moving_graph_shapes():
    first = probe.make_fixture(torch, probe.case_plan()[0])
    second = probe.make_fixture(torch, replace(probe.case_plan()[0], contexts=(593,)))
    assert len(first["expected"]) == len(second["expected"]) == 672
    assert first["blocks"] == second["blocks"] == 3
    assert first["cpu"]["table"].shape == (1, 10)
    assert {key: (value.shape, value.dtype) for key, value in first["cpu"].items()} == {
        key: (value.shape, value.dtype) for key, value in second["cpu"].items()}
    assert not torch.equal(first["cpu"]["raw"], second["cpu"]["raw"])
    assert not torch.equal(first["cpu"]["lens"], second["cpu"]["lens"])
    assert torch.equal(first["cpu"]["starts"], second["cpu"]["starts"])
    assert bool((first["cpu"]["table"][:, 8:] == -1).all())
    assert bool((first["cpu"]["raw"][probe.reuse.PREFIX + 2 * first["stride"]:] == 0xA5).all())
    assert bool(torch.isnan(first["cpu"]["wk"][2:]).all())
    assert bool((first["cpu"]["tags"][2:] == -1).all())


def test_unavailable_npu_fails_without_cpu_substitute(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.reuse, "_active_device", lambda target: None)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"experimental_c16_member_pipeline": True}))
    with pytest.raises(RuntimeError, match="target NPU required"):
        probe.probe(config, probe.ROOT / "configs/acceptance.json")


def test_timed_launch_contains_merge_and_bf16_copy(monkeypatch):
    n, h, d = 2, 6, 256
    buffers = {"partial": torch.empty(n, h, 3, d), "lse": torch.empty(n, h, 3),
               "merged": torch.empty(n, h, d), "merged_lse": torch.empty(n, h),
               "merge_status": torch.empty(n, h, dtype=torch.int32),
               "bf16_output": torch.empty(n, h, d, dtype=torch.bfloat16)}
    calls = []
    def cv(*args):
        calls.append("cv")
        buffers["partial"].fill_(1.25)
    def merge(partial, lse, output, output_lse, status):
        calls.append("merge")
        output.copy_(partial[:, 0])
        output_lse.fill_(2.)
        status.zero_()
    monkeypatch.setattr(probe, "_cv_launch", cv)
    probe._launch(SimpleNamespace(merge_lse_out=merge), probe.CANDIDATE, {},
                  {"tokens": n, "spec": SimpleNamespace(heads=h, dim=d, splits=1)},
                  buffers, 20, 16)
    assert calls == ["cv", "merge"]
    assert torch.equal(buffers["bf16_output"], torch.full((n, h, d), 1.25, dtype=torch.bfloat16))


def test_source_binding_covers_stable_and_pipeline_production_inputs(tmp_path):
    import hashlib
    for name in probe.SOURCE_PATHS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    binding = probe.source_binding(tmp_path)
    assert binding == {name: hashlib.sha256(name.encode()).hexdigest() for name in probe.SOURCE_PATHS}
    assert "csrc/kernels/attention_cv_member_pipeline_cluster16.cpp" in binding
    assert "csrc/kernels/attention_cv_window_range_cluster16.cpp" in binding
    assert "csrc/include/oscar_c16_member_pipeline_launch.h" in binding
    assert "tools/probe_c16_member_pipeline.py" in binding
    (tmp_path / "csrc/kernels/attention_cv_member_pipeline_cluster16.cpp").unlink()
    with pytest.raises(FileNotFoundError):
        probe.source_binding(tmp_path)


def test_recent1_mechanism_cannot_replace_target_fixture():
    with pytest.raises(RuntimeError, match="not a mechanism fixture"):
        probe.make_fixture(torch, replace(probe.case_plan()[0], recent_tokens=1))


def _tasks_from_native_group_geometry():
    """Host-only descriptor fixture from Q672, GQA6 and M128 group boundaries."""
    spec=next(s for s in probe.case_plan() if s.name==probe.QR_CASE_NAME);tokens=spec.qlens[0];ratio=spec.heads//spec.kv_heads
    tile=128//ratio;context=spec.contexts[0]
    tasks=torch.zeros(tokens*3,16,dtype=torch.int64)
    for query in range(tokens):
        count=min(tile,tokens-query) if query%tile==0 else 0
        for source in range(3):
            row=tasks[query*3+source]
            row[0]=query;row[1]=count;row[3]=probe.reuse.SINK if source==0 else 0
            row[4]=min(context,max(probe.reuse.SINK,context+query+count-spec.recent_tokens))
            row[6]=row[7]=source;row[8]=context
    return {"tokens":tokens,"spec":spec},tasks


def test_target_last_member_qr_domain_is_derived_from_real_group_geometry():
    fixture,tasks=_tasks_from_native_group_geometry()
    statuses,affected,domain=probe.qr_fault_domain(torch,tasks,fixture)
    assert torch.nonzero(statuses).tolist()==[[651*3,0]]
    assert int(statuses[651*3,0])==2
    assert torch.nonzero(affected).tolist()==[[row,0] for row in range(3906,3970)]
    assert domain["cluster_anchor"]==336 and domain["member"]==15
    assert domain["unaffected_rows"]==3968
    damaged=tasks.clone();damaged[336*3,4]-=1
    with pytest.raises(RuntimeError,match="complete mature16-member"):
        probe.qr_fault_domain(torch,damaged,fixture)
    with pytest.raises(RuntimeError,match="query651"):
        probe.make_fixture(torch,replace(fixture['spec'],corrupt_qr_token=315))


def test_qr_fixture_changes_only_declared_last_member_qr_word():
    clean=probe.make_fixture(torch,probe.case_plan()[0])
    poisoned=probe.make_fixture(torch,next(s for s in probe.case_plan() if s.name==probe.QR_CASE_NAME))
    for name in clean['cpu']:
        if name!='qr':
            assert probe._store_bits_equal(torch,clean['cpu'][name],poisoned['cpu'][name]),name
    old=clean['cpu']['qr'].view(torch.int32);new=poisoned['cpu']['qr'].view(torch.int32)
    assert torch.nonzero(old!=new).tolist()==[[651,0,0]]
    assert int(new[651,0,0])==0x7fc10001
    assert torch.equal(poisoned['clean_qr'],clean['cpu']['qr'])
    assert not poisoned['spec'].speed_gate and not poisoned['spec'].corrupt_metadata


def _isolated_qr_host_outputs():
    fixture,tasks=_tasks_from_native_group_geometry()
    status,affected,_=probe.qr_fault_domain(torch,tasks,fixture)
    n,h,d=672,6,256;rows=n*h
    partial=torch.full((rows,3,d),.5);lse=torch.zeros(rows,3)
    partial[:,2].zero_();lse[:,2].fill_(-float('inf'))
    clean={'partial':partial.clone(),'lse':lse.clone()}
    partial[affected]=float('nan');lse[affected]=float('nan')
    bad=affected.any(-1);merged=torch.full((rows,d),.5)
    merged_lse=torch.full((rows,),float(torch.log(torch.tensor(2.))))
    fixture['expected']={i:(merged.view(n,h,d)[i].clone(),merged_lse.view(n,h)[i].clone()) for i in range(n)}
    merged[bad]=0;merged_lse[bad]=-float('inf')
    merge_status=torch.zeros(rows,dtype=torch.int32);merge_status[bad]=2
    buffers={'tasks':tasks,'status':status,'partial':partial,'lse':lse,
             'merged':merged,'merged_lse':merged_lse,'merge_status':merge_status,
             'bf16_output':merged.to(torch.bfloat16)}
    return fixture,buffers,clean


@pytest.mark.parametrize('damage',[None,'other_member','window','merge_nan','live_output'])
def test_qr_checker_rejects_cross_member_window_and_merge_contamination(damage):
    fixture,buffers,clean=_isolated_qr_host_outputs()
    if damage=='other_member':buffers['status'][336*3,0]=2
    elif damage=='window':buffers['partial'][3906,1,0]=.75
    elif damage=='merge_nan':buffers['merged'][3906,0]=float('nan')
    elif damage=='live_output':buffers['merged'][0,0]=.75
    if damage is None:
        result=probe.check_qr_isolation(torch,fixture,buffers,clean,{'atol':.005,'rtol':.005})
        assert result['history_rows']==64 and result['unaffected_rows']==3968
        assert result['window_current_preserved'] is True
    else:
        with pytest.raises((RuntimeError,AssertionError)):
            probe.check_qr_isolation(torch,fixture,buffers,clean,{'atol':.005,'rtol':.005})


@pytest.mark.parametrize('field,value',[('leader',315),('lane',1),('unaffected_rows',1952),
                                       ('window_current_preserved',False),('merge_failure','nan')])
def test_qr_receipt_rejects_mechanism_domain_or_missing_isolation(field,value):
    report=evidence();next(r for r in report['cases'] if r['case']==probe.QR_CASE_NAME)['detail']['qr_fault'][field]=value
    assert 'last_member_qr.isolation' in probe.validate_evidence(report)


@pytest.mark.parametrize('context,blocks,columns',[(32768,66,262),(65536,130,518)])
def test_long_plan_uses_actual_context_capacity_without_building_a_long_matrix(context,blocks,columns):
    spec=next(s for s in probe.case_plan() if s.contexts==(context,))
    assert spec.speed_gate and spec.qlens==(672,) and spec.recent_tokens==256
    # Tiny byte/shape stand-ins isolate capacity algebra; no large oracle is
    # built in this test. Actual full32K construction has a separate hard cap.
    fixture={'spec':spec,'blocks':blocks,'stride':8,'cpu':{
        'raw':torch.full((probe.reuse.PREFIX+blocks*8,),0xA5,dtype=torch.uint8),
        'table':torch.full((1,columns),-1,dtype=torch.int32),
        'wk':torch.full((blocks,2,1,1),float('nan')),
        'wv':torch.full((blocks,2,1,1),float('nan')),
        'tags':torch.full((blocks,2),-1,dtype=torch.int64)}}
    addresses={name:t.data_ptr() for name,t in fixture['cpu'].items()}
    probe._fixed_graph_capacity(torch,fixture)
    assert fixture['blocks']==blocks and fixture['cpu']['table'].shape==(1,columns)
    assert addresses=={name:t.data_ptr() for name,t in fixture['cpu'].items()}


@pytest.mark.parametrize('case',tuple(probe.LONG_CONTEXTS))
def test_each_long_case_requires_its_own_full_speed_samples_and_all_query_oracle(case):
    assert probe.validate_evidence(evidence())==[]
    for missing in ('timing','one_sample','oracle','context','graph','anchor','anchor_bits'):
        report=evidence();row=next(r for r in report['cases'] if r['case']==case)
        if missing=='timing':row['timing']=None
        elif missing=='one_sample':row['timing']['samples_ms'][probe.BASELINE].pop()
        elif missing=='oracle':row['detail']['oracle']='sampled_queries_only'
        elif missing=='context':row['detail']['context']=337
        elif missing=='graph':row['graph']={'capture':'passed'}
        elif missing=='anchor':row['detail']['reference_anchor_check']['queries'].pop()
        else:row['detail']['reference_anchor_check']['status']='cpu_bit_differences'
        assert probe.validate_evidence(report),missing


@pytest.mark.parametrize('context',[0,17,337])
def test_shared_zero_stride_gqa_oracle_preserves_reference_fp32_bits(context):
    from oscar_ascend.ops.reference import attention
    generator=torch.Generator().manual_seed(4551+context)
    q=torch.randn(1,6,256,generator=generator).to(torch.bfloat16)
    k=torch.randn(context,1,256,generator=generator);v=torch.randn(context,1,256,generator=generator)
    expected=attention(q,k,v,scale=.0625,causal=False)
    got=probe._shared_kv_attention(torch,q,k,v,.0625)
    assert torch.equal(expected.output.view(torch.int32),got.output.view(torch.int32))
    assert torch.equal(expected.lse.view(torch.int32),got.lse.view(torch.int32))
    if context:
        v[0,0,0]=float('nan')
        with pytest.raises(ValueError,match='non-finite'):
            probe._shared_kv_attention(torch,q,k,v,.0625)


def test_incremental_long_oracle_matches_all_short_original_queries_bitwise():
    from oscar_ascend.ops.reference import attention
    fixture=probe.make_fixture(torch,probe.case_plan()[0])
    expected=fixture['expected']
    k,v,dk,dv=fixture['logical_history'];context=fixture['spec'].contexts[0]
    anchors={}
    for row in probe.reuse._samples(672):
        cut=min(context,max(64,context+row+1-256))
        kk,vv=k.float().clone(),v.float().clone();kk[64:cut]=dk[64:cut];vv[64:cut]=dv[64:cut]
        full=attention(fixture['cpu']['q'][row:row+1],
            torch.cat((kk,fixture['cpu']['ck'][:row+1].float())),
            torch.cat((vv,fixture['cpu']['cv'][:row+1].float())),scale=fixture['scale'],causal=False)
        anchors[row]=(full.output[0],full.lse[0])
    fixture['generic_reference_anchors']=anchors
    probe._long_oracle(torch,fixture)
    assert len(fixture['expected'])==len(expected)==672
    for query in expected:
        for before,after in zip(expected[query],fixture['expected'][query]):
            assert torch.equal(before.view(torch.int32),after.view(torch.int32))
    assert fixture['reference_anchor_check']['status']=='bitwise_passed'
    assert fixture['reference_anchor_check']['queries']==list(probe.reuse._samples(672))
    anchors[0][0][0,0]+=1
    with pytest.raises(AssertionError):
        probe._long_oracle(torch,fixture)
    fixture['logical_history'][3][100,0,0]=float('nan')
    with pytest.raises(ValueError,match='non-finite'):
        probe._long_oracle(torch,fixture)


@pytest.mark.parametrize("speedup,passed", [(0.9999, False), (1.0, True), (1.041, True), (1.3, True)])
def test_complete_path_speed_gate_keeps_exact_threshold(speedup, passed):
    report = evidence()
    timing = report["cases"][0]["timing"]
    timing.update(baseline_ms=speedup, candidate_ms=1., ratio=1 / speedup, speedup=speedup,
                  samples_ms={probe.BASELINE: [speedup] * 5, probe.CANDIDATE: [1.] * 5})
    assert ("device_latency" not in probe.validate_evidence(report)) is passed
