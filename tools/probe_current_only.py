"""Archive #34/#36/#94/#95/#125/#140/#142/#148/#155: real NPU current-only gate.

D.4: this measures current FIA/copy/store, never historical dequant restore.
The old 6.5s dequant and ~215ms store are failure context, not our timings.
Only bounded sampled CPU oracles materialize diagnostic history. Small A/B
and subsequent q1 check correctness; 20K/30K events do not establish whole
model quality, graph replay or the user's eight-minute performance target.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import traceback

from . import probe_history_reuse as reuse
from .phase import atomic_json
from .probe_striped_cache import _store_bits_equal
from .striped_fixture import raw_to_striped, striped_slots_to_canonical
from .probe_bf16_attention import OPERATORS as BF16_OPERATORS
from .probe_aligned_attention import OPERATORS as ALIGNED_OPERATORS

ROOT = Path(__file__).resolve().parents[1]


def official_enabled(target):
    return target.get("experimental_official_math", target.get("official_math", False)) is True


def selected_prepare(target):
    return "prepare_official_attention_tasks_out" if official_enabled(target) else "prepare_attention_tasks_out"


def math_contract(target):
    if official_enabled(target):
        return "oscar_bf16_attention_v4"
    return "oscar_bf16_attention_v3" if target.get("experimental_aligned_math", target.get("aligned_math", False)) else "oscar_bf16_attention_v2"


def selected_operators(target):
    if official_enabled(target):
        return {family: name.replace("_aligned_", "_official_") for family, name in ALIGNED_OPERATORS.items()}
    return ALIGNED_OPERATORS if target.get("experimental_aligned_math", target.get("aligned_math", False)) else BF16_OPERATORS


def selected_writer(target):
    return "store_aligned_int2_out" if target.get("experimental_aligned_math", False) else "rotate_clip_store_striped_out"


def make_fixture(torch, shape, target, *, rotation_mode):
    if official_enabled(target):
        from .official_fixture import make_fixture as official_fixture
        return official_fixture(torch, shape, rotation_mode=rotation_mode,
            k_clip=target.get("k_clip_ratio", .96), v_clip=target.get("v_clip_ratio", .92))
    if target.get("experimental_aligned_math", False):
        from .aligned_fixture import make_fixture as aligned_fixture
        return aligned_fixture(torch, shape, rotation_mode=rotation_mode,
            k_clip=target.get("k_clip_ratio", .96), v_clip=target.get("v_clip_ratio", .92))
    return reuse.make_fixture(torch, shape, rotation_mode=rotation_mode)


def precision_fields(target):
    fields = {"bf16_attention": True, "math_contract": math_contract(target), "performance": "not_run_user_requested"}
    if target.get("experimental_aligned_math", False):
        fields.update(aligned_math=True, cache_format="striped_fp32_v2")
    if official_enabled(target):
        fields.update(official_math=True, cache_format="striped_rotated_fp32_v3")
    return fields


def device_tensors(torch, fixture, device):
    tensors = {key: value.to(device) for key, value in fixture["cpu"].items()}
    if fixture.get("cache_format") == "striped_rotated_fp32_v3":
        originals = fixture.get("original_inputs")
        if not isinstance(originals, dict) or not {"q", "ck", "cv"} <= set(originals):
            raise RuntimeError("official lifecycle fixture lacks original runtime projection inputs")
        for key in ("q", "ck", "cv"):
            tensors[key] = originals[key].to(device)
    return tensors


def aligned_source_snapshot(target):
    if official_enabled(target):
        from .probe_official_attention import source_binding
        return source_binding()
    if target.get('experimental_aligned_math',False):
        from .probe_aligned_attention import source_binding
        return source_binding()
    return None


def bind_aligned_evidence(report, target, source_before=None):
    if target.get('experimental_aligned_math',False):
        if official_enabled(target):
            from .probe_official_attention import source_binding
        else:
            from .probe_aligned_attention import source_binding
        current=source_binding()
        if source_before is not None and current!=source_before:
            raise RuntimeError('aligned lifecycle source changed during the device gate')
        report.update(aligned_source_binding=current, aligned_clip={
            'k':target.get('k_clip_ratio',.96),'v':target.get('v_clip_ratio',.92)})
    return report


def validate_aligned_calls(calls, target):
    if not target.get('experimental_aligned_math',target.get('aligned_math',False)):
        return
    allowed=set(selected_operators(target).values())
    if not isinstance(calls,dict) or calls.get('store_aligned_int2_out',0)<1:
        raise RuntimeError('aligned lifecycle lacks actual new writer witness')
    if official_enabled(target) and (calls.get(selected_prepare(target), 0) < 1 or
            calls.get("copy_validate_current_out", 0) < 1):
        raise RuntimeError("official lifecycle lacks actual prepare/final-output witnesses")
    for name,count in calls.items():
        if count and ((name.startswith('attention_cv_') and name not in allowed) or
                      name.startswith('rotate_clip_store') or name in {'rotate_out','rotate_vector_init_out'}):
            raise RuntimeError('aligned lifecycle called a legacy codec/rotation/reader: '+name)
        if official_enabled(target) and count and name == "prepare_attention_tasks_out":
            raise RuntimeError("official lifecycle called the legacy moving-tier prepare")


def case_plan():
    # Four short A/B pairs; two long cases never launch long FP32 CV baselines.
    return (
        (17, False, "int64", True), (17, True, "int32", True),
        (385, False, "int32", True), (385, True, "int64", True),
        (20000, False, "int64", False), (30000, True, "int32", False),
    )


def _case_name(case):
    fresh, draft, slot_type, paired = case
    return f"{'mixed' if paired else 'fresh'}_{fresh}_{'first_mtp' if draft else 'main'}_{slot_type}"


def _rotation_mode(target, draft):
    # Runtime._prepare_rotations assigns MTP the explicit PR identity. It
    # must not accidentally be timed as a main-layer Hadamard transform.
    return "identity" if draft else ("hadamard" if target.get("rotation_method") == "hadamard" else "nontrivial")


def _rotation_label(mode):
    return {"identity": "draft_identity", "hadamard": "target_hadamard",
            "nontrivial": "nontrivial_dense_fixture"}[mode]


def validate_case_evidence(report, target, acceptance):
    """Reject partial/stale reports even when their top-level status says pass."""
    rows = report.get("cases")
    expected_cases = {_case_name(case): case for case in case_plan()}
    if (not isinstance(rows, list) or len(rows) != len(expected_cases)
            or any(not isinstance(row, dict) for row in rows)):
        raise RuntimeError("current-only evidence requires all six case records")
    names = [row.get("case") for row in rows]
    if any(not isinstance(name, str) for name in names) or set(names) != set(expected_cases):
        raise RuntimeError("current-only evidence has missing, duplicated or unexpected cases")
    policy = acceptance["performance"]
    bf16=target.get("experimental_bf16_attention",False)
    from oscar_ascend.ops.cv_dispatch import OPTIMIZED_STRIPED_OPS
    def reader(name):
        if bf16:return selected_operators(target)[{"attention_cv_striped_decode_out":"q4",
                         "attention_cv_striped_q1_out":"q1"}[name]]
        return OPTIMIZED_STRIPED_OPS[name] if target.get("experimental_decode_bundle", False) else name
    for row in rows:
        name = row["case"]
        fresh, draft, slot_type, paired = expected_cases[name]
        required = {"status": "passed", "independent_oracle": "passed", "copy_lse_oracle": "passed",
            "slot_dtype": slot_type,
            "rotation": _rotation_label(_rotation_mode(target,draft))}
        if (any(row.get(key) != value for key, value in required.items())
                or row.get("first_draft") is not draft
                or row.get("sampled_queries") != len(reuse._samples(fresh)) + (4 if paired else 0)):
            raise RuntimeError(f"current-only case {name} lacks exact oracle/shape/rotation evidence")
        error = row.get("max_output_abs")
        if type(error) not in (float, int) or not math.isfinite(error) or error < 0:
            raise RuntimeError(f"current-only case {name} lacks a finite oracle error")
        calls = row.get("real_operator_calls")
        validate_aligned_calls(calls,target)
        expected_copy = ((3 if paired else 1) if official_enabled(target) else
                         1 if paired or bf16 else 1 + policy["warmup"] + policy["repeats"])
        expected_store = 3 if paired else expected_copy
        if (not isinstance(calls, dict)
                or type(calls.get("copy_validate_current_out")) is not int
                or calls["copy_validate_current_out"] != expected_copy
                or type(calls.get(selected_writer(target))) is not int
                or calls[selected_writer(target)] != expected_store):
            raise RuntimeError(f"current-only case {name} lacks real copy/writer call witnesses")
        if paired:
            required_pair = {"ab_output": "separate_BF16_native_contracts_passed" if bf16 else "frozen_tolerance_passed", "store_bitwise": "passed",
                "baseline": "complete_rotated_current_CV" if official_enabled(target) else "complete_current_CV" if draft else "existing_main_current_FIA_merge"}
            decode = row.get("decode_after_store", {})
            required_decode = {"status": "passed", "reader": "production_subsequent_q1",
                "output_bitwise": "passed", "independent_oracle": "passed",
                "subsequent_store_bitwise": "passed", "sampled_queries": 2}
            if (any(row.get(key) != value for key, value in required_pair.items())
                    or not isinstance(decode, dict)
                    or any(decode.get(key) != value for key, value in required_decode.items())
                    or calls.get(reader("attention_cv_striped_decode_out")) != 1
                    or calls.get(reader("attention_cv_striped_q1_out")) != 1):
                raise RuntimeError(f"current-only case {name} lacks A/B, store or existing q1 reader evidence")
        else:
            if bf16:
                if row.get("performance")!="not_run_user_requested":raise RuntimeError("current-only BF16 must not fabricate timing")
                continue
            events, median = row.get("device_event_ms"), row.get("device_event_median_ms")
            if (not isinstance(events, list) or len(events) != policy["repeats"]
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                           for value in events)
                    or type(median) not in (int, float) or not math.isfinite(median)
                    or not math.isclose(median, statistics.median(events), rel_tol=1e-12)
                    or row.get("timing_scope") != "production_fresh_FIA_copy_validate_INT2_store_status_guard"
                    or row.get("latency_gate") != "not_established_no_long_CV_baseline"):
                raise RuntimeError(f"current-only case {name} lacks valid long-combination event statistics")


class CountedOps:
    """Call witnesses only: every invocation delegates unchanged to real ops."""
    def __init__(self, ops):
        self.ops, self.calls = ops, Counter()

    def __getattr__(self, name):
        operation = getattr(self.ops, name)
        def invoke(*args, **kwargs):
            self.calls[name] += 1
            return operation(*args, **kwargs)
        return invoke


def _configuration(target, enabled, *, probe_tokens=None):
    # The short gate is specifically for the current production striped bundle.
    for key in ("experimental_history_reuse", "experimental_fast_unpack",
                "experimental_mixed_cv", "experimental_striped_cache"):
        if target.get(key) is not True:
            raise RuntimeError(f"current-only gate requires explicit {key}=true")
    config = {**target, "experimental_current_only": enabled}
    if target.get("experimental_bf16_attention",False):
        if official_enabled(target):
            from .official_reference import load_acceptance
        elif target.get("experimental_aligned_math", False):
            from .aligned_reference import load_acceptance
        else:
            from .bf16_reference import load_acceptance
        load_acceptance()
    if probe_tokens is not None:
        if type(probe_tokens) is not int or probe_tokens <= 0:
            raise ValueError("independent probe token capacity must be a positive integer")
        # This private model-free provider already receives a fixture-sized
        # GraphWorkspace. Do not inherit/guess the service scheduler or graph
        # capacity, and never write these diagnostic values to target config.
        config.update(max_num_batched_tokens=probe_tokens, use_native_token_budget=False)
        config["compilation_config"] = {
            **config.get("compilation_config", {}), "cudagraph_capture_sizes": []}
    return config


def _runtime(torch, fixture, tensors, device, cores, target, enabled, *, hadamard=False):
    from oscar_ascend.runtime import AscendRuntimeProvider, GraphWorkspace, LayerState, WorkspaceGeometry
    from oscar_ascend.integration.impl import OscarAttentionImpl
    spec = fixture["spec"]
    workspace = GraphWorkspace(WorkspaceGeometry(
        fixture["tokens"], spec.heads, spec.kv_heads, spec.dim,
        splits=1, cube_cores=cores, history_cluster_size=16,
        c4_member_pipeline=target.get("experimental_c4_member_pipeline",False) or target.get("experimental_bf16_attention",False),
        c16_member_pipeline=target.get("experimental_c16_member_pipeline",False),
        c16_l1_reuse=target.get("experimental_c16_l1_reuse",False) or target.get("experimental_bf16_attention",False),
        **({"aligned_math": True, "official_math": official_enabled(target)} if target.get("experimental_aligned_math", False) else {})), device)
    # The independent fixture has a 64-byte diagnostic prefix. This FULL-only
    # runtime uses conv_bytes=0 and strips that prefix from both A/B buffers.
    aligned = target.get("experimental_aligned_math", False)
    raw = (fixture['cpu']['raw'] if aligned else raw_to_striped(fixture))[reuse.PREFIX:].to(device)
    rk = (fixture["rotation_k"].float() if official_enabled(target) else
          torch.eye(spec.dim,dtype=torch.float32) if fixture["rotation_mode"] == "identity"
          else reuse._hadamard(torch,spec.dim)).to(device)
    rv = tensors["rv"].float() if aligned else tensors["rv"]
    state = LayerState(raw, raw, SimpleNamespace(block_size=reuse.BLOCK_TOKENS,
        conv_bytes=0, ssm_bytes=fixture["stride"]),
        SimpleNamespace(sink_tokens=reuse.SINK, recent_tokens=spec.recent_tokens,
            speculative_tokens=reuse.SPECULATIVE, ring_tokens=spec.recent_tokens+reuse.SPECULATIVE),
        tensors["wk"].clone(), tensors["wv"].clone(), tensors["tags"].clone(),
        rk, rv, rk.T.contiguous(), rv.T.contiguous(), hadamard, fixture["blocks"], workspace,
        "striped_rotated_fp32_v3" if official_enabled(target) else "striped_fp32_v2" if aligned else "striped_v1",
        rotation_v_nz=(rv.reshape(spec.dim,spec.dim//8,8).permute(1,0,2).contiguous().view(spec.dim,spec.dim)
                       if target.get("experimental_hybrid_l1_decode",False) and not target.get("experimental_bf16_attention",False) else None),
        rotation_v_bf16=rv.to(torch.bfloat16).contiguous() if target.get("experimental_bf16_attention",False) else None,
        **({"rotation_k_bf16": rk.to(torch.bfloat16).contiguous()} if aligned else {}))
    provider = AscendRuntimeProvider(_configuration(target, enabled, probe_tokens=fixture["tokens"]))
    provider.layers["probe.full"] = state
    provider.ops = CountedOps(torch.ops.oscar_ascend_ops)
    # Instantiate the real forward without model/distributed initialization.
    # Workspace, layer state, every op and native FIA remain the production code.
    impl = object.__new__(OscarAttentionImpl)
    impl.provider, impl.num_heads, impl.num_kv_heads = provider, spec.heads, spec.kv_heads
    impl.head_size, impl.scale = spec.dim, fixture["scale"]
    return impl, state


def _metadata(torch, fixture, tensors, *, draft, enabled):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from oscar_ascend.integration.metadata import from_common
    common = SimpleNamespace(causal=True, query_start_loc=tensors["starts"],
        query_start_loc_cpu=fixture["cpu"]["starts"], seq_lens=tensors["lens"],
        seq_lens_cpu_upper_bound=fixture["cpu"]["lens"],
        block_table_tensor=tensors["table"], slot_mapping=tensors["slots"],
        num_reqs=len(fixture["spec"].qlens), num_actual_tokens=fixture["actual_tokens"],
        num_input_tokens=fixture["tokens"], max_query_len=max(fixture["spec"].qlens),
        max_seq_len=max(fixture["cpu"]["lens"].tolist()),
        attn_state=AscendAttentionState.ChunkedPrefill)
    # Explicit first-pass test metadata; actual builder/config qualification
    # is checked by the separate pinned-native CPU contracts.
    return from_common(common, is_draft=draft, current_only=enabled,
                       first_draft_current_only=enabled and draft)


def _forward(torch, impl, state, tensors, metadata, output=None):
    if output is None:
        output = torch.empty_like(tensors["q"])
    impl.forward(SimpleNamespace(layer_name="probe.full"), tensors["q"],
        tensors["ck"], tensors["cv"], state.packed, metadata, output=output)
    return output


def _assert_output(torch, output, expected, tolerance):
    indexes = sorted(expected)
    observed = output[indexes].float().cpu()
    reference = torch.stack([expected[index][0] for index in indexes])
    torch.testing.assert_close(observed, reference, **tolerance)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("current-only full output has nonfinite rows")
    return {"sampled_queries": len(indexes),
            "max_output_abs": float((observed-reference).abs().max())}


def _expected(fixture,target,*,native_current):
    if not target.get("experimental_bf16_attention",False):return fixture["expected"]
    if official_enabled(target):
        from .official_runtime_oracle import sampled_fixture_expected
        return sampled_fixture_expected(fixture, native_current=native_current, dequant_mode=0)
    if target.get("experimental_aligned_math", False):
        from .aligned_runtime_oracle import sampled_fixture_expected
        # All initial lifecycle metadata below explicitly uses ChunkedPrefill.
        return sampled_fixture_expected(fixture, native_current=native_current, dequant_mode=0)
    from .bf16_runtime_oracle import sampled_fixture_expected
    return sampled_fixture_expected(fixture,native_current=native_current)


def _next_bf16_decode_oracle(torch,fixture,state,q,k,v):
    from dataclasses import replace
    official=fixture.get('cache_format')=='striped_rotated_fp32_v3'
    aligned=official or fixture.get('cache_format')=='striped_fp32_v2'
    if official:
        from .official_runtime_oracle import sampled_fixture_expected
        from .official_reference import rotate_bf16
    elif aligned:
        from .aligned_runtime_oracle import sampled_fixture_expected
        from .aligned_reference import rotate_bf16
    else:
        from .bf16_runtime_oracle import sampled_fixture_expected
    spec=fixture['spec'];count=len(spec.qlens)
    canonical=(state.raw.cpu().view(fixture['blocks'],reuse.BLOCK_TOKENS,spec.kv_heads,144) if aligned else
        striped_slots_to_canonical(state.raw.cpu().view(fixture['blocks'],reuse.BLOCK_TOKENS,spec.kv_heads,136)))
    contexts=tuple(a+b for a,b in zip(spec.contexts,spec.qlens))
    cpu={**fixture['cpu'],'q':q,'qr':rotate_bf16(q,state.rotation_k_bf16.cpu()) if aligned else q.float()@state.rotation_k.cpu(),'ck':k,'cv':v,
         'rv':state.rotation_v.cpu(),'raw':torch.cat((torch.zeros(reuse.PREFIX,dtype=torch.uint8),canonical.reshape(-1))),
         'wk':state.window_key.cpu(),'wv':state.window_value.cpu(),'tags':state.window_tags.cpu()}
    sample={**fixture,'cpu':cpu,'spec':replace(spec,qlens=(1,)*count,contexts=contexts,padded_tokens=count,slot_context=True),
            'tokens':count,'actual_tokens':count,'expected':{i:None for i in range(count)}}
    if official:
        qr=rotate_bf16(q,state.rotation_k_bf16.cpu())
        sample['cpu'].update(q=qr,qr=qr,ck=rotate_bf16(k,state.rotation_k_bf16.cpu()),
            cv=rotate_bf16(v,state.rotation_v_bf16.cpu()))
        sample['cached_quant_ends']=[max(min(reuse.SINK,c),c+1-spec.recent_tokens) for c in contexts]
    return sampled_fixture_expected(sample,native_current=False,**({'dequant_mode':1} if aligned else {}))


def _assert_stores(torch, old, new):
    for name in ("raw", "window_key", "window_value", "window_tags"):
        if not _store_bits_equal(torch, getattr(old, name), getattr(new, name)):
            raise RuntimeError(f"current-only store changed {name} bits")


def _next_decode_oracle(torch, fixture, state, q, k, v):
    """Independent dense oracle for TWO single-query reads, before next store."""
    from oscar_ascend.ops.reference import attention, decode_kv
    spec = fixture["spec"]
    canonical = striped_slots_to_canonical(state.raw.cpu().view(
        fixture["blocks"], reuse.BLOCK_TOKENS, spec.kv_heads, 136))
    wk, wv, tags = state.window_key.cpu(), state.window_value.cpu(), state.window_tags.cpu()
    rk, rv = state.rotation_k.cpu(), state.rotation_v.cpu()
    expected = {}
    for req, (old, length) in enumerate(zip(spec.contexts, spec.qlens)):
        context = old + length
        positions = list(range(context))
        pages = fixture["page_assignments"][req]
        physical = torch.tensor([pages[p//reuse.BLOCK_TOKENS] for p in positions])
        offsets = torch.arange(context) % reuse.BLOCK_TOKENS
        keys, values = decode_kv(canonical[physical, offsets], spec.dim)
        keys, values = keys @ rk.T, values @ rv.T
        cut = min(context, max(reuse.SINK, context+1-spec.recent_tokens))
        exact = list(range(min(reuse.SINK, context))) + list(range(cut, context))
        for p in exact:
            page, off = pages[p//reuse.BLOCK_TOKENS], p % reuse.BLOCK_TOKENS
            row = p if p < reuse.SINK else reuse.SINK + off % (spec.recent_tokens+reuse.SPECULATIVE)
            if int(tags[page,row]) != off:
                raise RuntimeError("new store lacks the exact window needed by next decode")
            keys[p], values[p] = wk[page,row], wv[page,row]
        result = attention(q[req:req+1], torch.cat((keys,k[req:req+1].float())),
            torch.cat((values,v[req:req+1].float())), scale=fixture["scale"], causal=False)
        expected[req] = result.output[0], result.lse[0]
    return expected


def _decode_after_store(torch, fixture, tensors, old_impl, old, new_impl, new, device, tolerance):
    from oscar_ascend.integration.metadata import OscarMetadata
    spec = fixture["spec"]
    count = len(spec.qlens)
    generator = torch.Generator().manual_seed(47001)
    q = torch.randn(count,spec.heads,spec.dim,generator=generator).to(torch.bfloat16)
    k = torch.randn(count,spec.kv_heads,spec.dim,generator=generator).to(torch.bfloat16)
    v = torch.randn(count,spec.kv_heads,spec.dim,generator=generator).to(torch.bfloat16)
    bf16=new_impl.provider.config.get("experimental_bf16_attention",False)
    if bf16!=old_impl.provider.config.get("experimental_bf16_attention",False):raise RuntimeError("lifecycle A/B readers use different precision contracts")
    expected = (_next_bf16_decode_oracle(torch,fixture,new,q,k,v) if bf16 else
                _next_decode_oracle(torch,fixture,new,q,k,v))
    contexts = [old+length for old,length in zip(spec.contexts,spec.qlens)]
    slots = [fixture["page_assignments"][r][p//reuse.BLOCK_TOKENS]*reuse.BLOCK_TOKENS+p%reuse.BLOCK_TOKENS
             for r,p in enumerate(contexts)]
    inputs = {**tensors,"q":q.to(device),"ck":k.to(device),"cv":v.to(device)}
    metadata = OscarMetadata(torch.arange(count+1,dtype=torch.int32,device=device),
        torch.tensor([p+1 for p in contexts],dtype=torch.int32,device=device), tensors["table"],
        torch.tensor(slots,dtype=tensors["slots"].dtype,device=device),count,count,1,max(contexts)+1,
        num_input_tokens=count,is_draft=True,draft_index=1)
    if new_impl.provider.config.get("experimental_aligned_math", False):
        from dataclasses import replace
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        metadata = replace(metadata, attn_state=AscendAttentionState.SpecDecoding)
    q1 = selected_operators(new_impl.provider.config)['q1']
    outputs=[]
    for impl,state in ((old_impl,old),(new_impl,new)):
        before=impl.provider.ops.calls["copy_validate_current_out"]
        bf16_before=impl.provider.ops.calls[q1] if bf16 else None
        outputs.append(_forward(torch,impl,state,inputs,metadata))
        expected_copy_delta=1 if official_enabled(impl.provider.config) else 0
        if impl.provider.ops.calls["copy_validate_current_out"] != before+expected_copy_delta:
            raise RuntimeError("subsequent q1 wrongly selected current-only")
        if bf16 and impl.provider.ops.calls[q1]!=bf16_before+1:
            raise RuntimeError("subsequent q1 did not execute the BF16 reader")
    torch.npu.synchronize()
    if not _store_bits_equal(torch,*outputs):
        raise RuntimeError("existing q1 reader changed after current-only store")
    errors=_assert_output(torch,outputs[1],expected,tolerance)
    _assert_stores(torch,old,new)
    return {"status":"passed","reader":"production_subsequent_q1", "output_bitwise":"passed",
            "operator":q1 if bf16 else "legacy_selected_q1",
            "independent_oracle":"passed","subsequent_store_bitwise":"passed",**errors}


def run_case(torch, target, acceptance, device, cores, case):
    fresh,draft,slot_type,paired=case
    name=_case_name(case)
    print(f"[oscar] CURRENT_ONLY_CASE_BEGIN case={name}",flush=True)
    shape=reuse.Shape(name,(4,fresh) if paired else (fresh,),
        (511,0) if paired else (0,),256,1,1,False)
    rotation_mode = _rotation_mode(target,draft)
    fixture=make_fixture(torch,shape,target,rotation_mode=rotation_mode)
    bf16=target.get("experimental_bf16_attention",False)
    expected=_expected(fixture,target,native_current=tuple(not draft or context==0 for context in shape.contexts))
    hadamard = rotation_mode == "hadamard"
    tensors=device_tensors(torch,fixture,device)
    tensors["slots"]=tensors["slots"].to(getattr(torch,slot_type))
    impl,state=_runtime(torch,fixture,tensors,device,cores,target,True,hadamard=hadamard)
    metadata=_metadata(torch,fixture,tensors,draft=draft,enabled=True)
    if metadata.current_only_plan is None or metadata.current_only_plan.fresh_tokens!=fresh:
        raise RuntimeError("explicit current-only NPU case did not produce the expected plan")
    original_raw=state.raw.clone()
    output=_forward(torch,impl,state,tensors,metadata)
    torch.npu.synchronize()
    tolerance=acceptance["fused_attention"]
    errors=_assert_output(torch,output,expected,tolerance)
    expected_initial_copies=(2 if paired else 1) if official_enabled(target) else 1
    if impl.provider.ops.calls["copy_validate_current_out"]!=expected_initial_copies or impl.provider.ops.calls[selected_writer(target)]!=(2 if paired else 1):
        raise RuntimeError("production current-only/copy/striped-writer route witness mismatch")
    if _store_bits_equal(torch,original_raw,state.raw):
        raise RuntimeError("current-only writer did not publish compressed bytes")
    cut=4 if paired else 0
    sample=[i for i in sorted(expected) if i>=cut]
    observed_lse=state.workspace.lse[[i-cut for i in sample]].cpu()
    expected_lse=torch.stack([expected[i][1] for i in sample])
    torch.testing.assert_close(observed_lse,expected_lse,**tolerance)
    row={"case":name,"status":"passed","first_draft":draft,"slot_dtype":slot_type,
         "workspace_token_capacity":fixture["tokens"],
         "capacity_source":"independent_fixture_not_service_scheduler",
         "rotation":_rotation_label(rotation_mode),
         "independent_oracle":"passed","copy_lse_oracle":"passed",**errors}
    if paired:
        old_impl,old=_runtime(torch,fixture,tensors,device,cores,target,False,hadamard=hadamard)
        old_meta=_metadata(torch,fixture,tensors,draft=draft,enabled=False)
        old_output=_forward(torch,old_impl,old,tensors,old_meta)
        torch.npu.synchronize()
        if old_impl.provider.ops.calls["copy_validate_current_out"] != (1 if official_enabled(target) else 0):
            raise RuntimeError("disabled A/B baseline selected current-only")
        old_expected=_expected(fixture,target,native_current=False if official_enabled(target) else not draft)
        _assert_output(torch,old_output,old_expected,tolerance)
        if not bf16:torch.testing.assert_close(output.float(),old_output.float(),**tolerance)
        _assert_stores(torch,old,state)
        row.update(ab_output="separate_BF16_native_contracts_passed" if bf16 else "frozen_tolerance_passed",store_bitwise="passed",
            ab_max_abs_diagnostic=float((output.float()-old_output.float()).abs().max()) if bf16 else None,
            baseline="complete_rotated_current_CV" if official_enabled(target) else "complete_current_CV" if draft else "existing_main_current_FIA_merge",
            decode_after_store=_decode_after_store(torch,fixture,tensors,old_impl,old,impl,state,device,tolerance))
    elif bf16:
        row.update(performance="not_run_user_requested",store_bitwise="small_pairs_only")
    else:
        measurements=[]
        policy=acceptance["performance"]
        for iteration in range(policy["warmup"]+policy["repeats"]):
            begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            begin.record()
            output=_forward(torch,impl,state,tensors,metadata,output)
            end.record();torch.npu.synchronize()
            ms=float(begin.elapsed_time(end))
            if not math.isfinite(ms) or ms<=0:raise RuntimeError("invalid current-only device duration")
            if iteration>=policy["warmup"]:measurements.append(ms)
        _assert_output(torch,output,fixture["expected"],tolerance)
        row.update(device_event_ms=measurements,device_event_median_ms=statistics.median(measurements),
            timing_scope="production_fresh_FIA_copy_validate_INT2_store_status_guard",
            latency_gate="not_established_no_long_CV_baseline",store_bitwise="small_pairs_only")
    row["real_operator_calls"]=dict(impl.provider.ops.calls)
    if bf16:row.update(precision_fields(target))
    print("[oscar] CURRENT_ONLY_CASE_DONE "+json.dumps(row,sort_keys=True),flush=True)
    return row


def probe(config_path, acceptance_path):
    target=json.loads(config_path.read_text());acceptance=json.loads(acceptance_path.read_text())
    if target.get("experimental_current_only") is not True:
        raise RuntimeError("current-only NPU gate requires an explicit experimental_current_only=true")
    _configuration(target,True)
    bindings=aligned_source_snapshot(target)
    if acceptance.get("frozen_before_measurement") is not True:
        raise RuntimeError("frozen acceptance is required")
    from .probe_native_current_fia import _select_target_npu, _target_geometry
    _select_target_npu(target)
    import torch
    import torch_npu  # noqa: F401 -- no CPU operator substitute
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if not torch.npu.is_available():raise RuntimeError("current-only gate requires a real NPU")
    torch.npu.set_device(0)
    from .build_ops import normalize_soc
    if normalize_soc(torch.npu.get_device_name(0))!=target["soc_version"]:
        raise RuntimeError("selected NPU SoC differs from explicit target")
    if _target_geometry(target)!=(6,1,256):raise RuntimeError("current-only gate requires Hq6/Hkv1/D256")
    import vllm_ascend.ops  # noqa: F401 -- initialize before native attention imports
    from oscar_ascend.ops.loader import require_capabilities,validate_build_artifacts
    from oscar_ascend.ops.cv_dispatch import STRIPED_CV_OPS
    manifest_path=ROOT/"build/ascendc/build_manifest.json"
    manifest=validate_build_artifacts(manifest_path)
    require_capabilities(STRIPED_CV_OPS|{"copy_validate_current_out",selected_prepare(target),
        "rotate_out","merge_lse_out","status_guard",selected_writer(target)}|
        (set(selected_operators(target).values()) | ({'copy_validate_bf16_query_out'} if target.get('experimental_aligned_math',False) else set())
         if target.get("experimental_bf16_attention",False) else set()),manifest_path)
    cores=reuse._core_count(torch,target);device=torch.device("npu:0")
    rows=[]
    for case in case_plan():
        rows.append(run_case(torch,target,acceptance,device,cores,case))
        gc.collect();torch.npu.empty_cache()
    report={"status":"passed","precision":"passed","store_bitwise":"passed",
        "decode_after_store":"passed","device_completion":"passed","cases":rows,
        "artifact_signature":manifest["signature"],"artifact_sha256":manifest["sha256"],
        "graph_capture":"not_run_eager_scope","graph_replay":"not_run_eager_scope",
        "model_quality":"not_established","full_model_performance":"not_established"}
    if target.get("experimental_bf16_attention",False):report.update(precision_fields(target))
    bind_aligned_evidence(report,target,bindings)
    validate_case_evidence(report,target,acceptance)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/target.json")
    parser.add_argument("--acceptance",type=Path,default=ROOT/"configs/acceptance.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:report=probe(args.config,args.acceptance)
    except Exception as exc:
        traceback.print_exc();report={"status":"failed","first_error":str(exc)}
    atomic_json(args.output,report)
    print("[oscar] CURRENT_ONLY_RESULT "+json.dumps({"status":report["status"],"report":str(args.output)}),flush=True)
    return 0 if report["status"]=="passed" else 2


if __name__=="__main__":raise SystemExit(main())
