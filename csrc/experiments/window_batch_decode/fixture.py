"""Archive #126/#129/#140/#145/#148/#155: bounded window DMA goldens.

D.4 four questions: source1 of fused decode FIA; prior history restoration
cost 6499.8-6655.1ms/card while FIA was 18.5-18.9ms. This fixture reuses the
independent PR oracle offline and never restores serving history. Current
production SIMD is the A/B denominator; no CPU time is called NPU speed.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import probe_history_reuse as reuse
from tools.striped_fixture import raw_to_striped


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    contexts: tuple[int, ...]
    heads: int = 1
    padded: int = 8
    splits: int = 1
    fault: str | None = None
    run_default: bool = True
    cores: int | None = None


def case_plan(include_large=False, include_long_single=False):
    cases = [
        Case("q4_c511_ring259", "q4", (511,)),
        Case("q1_c511_ring259", "q1", (511,)),
        Case("q4_c641_page512", "q4", (641,)),
        Case("q1_c773_ring259", "q1", (773,)),
        Case("q4_hkv2_c641", "q4", (641,), heads=2),
        Case("q1_hkv2_c511", "q1", (511,), heads=2),
        Case("q4_bad_tag", "q4", (511,), fault="bad_tag"),
        Case("q4_vnan_then_tag", "q4", (511,), fault="vnan_then_tag"),
        Case("q4_tag_then_vnan", "q4", (511,), fault="tag_then_vnan"),
        Case("q1_bad_page", "q1", (511,), fault="bad_page"),
    ]
    if include_large:
        cases += [Case("q4_32req_n128", "q4", (511,641,773,1025)*8, padded=128, splits=3, run_default=False),
                  Case("q1_32req_n128", "q1", (511,641,773,1025)*8, padded=128, splits=3, run_default=False)]
    if include_long_single:
        cases += [Case("q4_1req_c20000", "q4", (20000,), padded=4, splits=3, run_default=False, cores=20),
                  Case("q1_1req_c20000_n128", "q1", (20000,), padded=128, splits=3, run_default=False, cores=20)]
    return tuple(cases)


def source_work(case):
    """Logical executed KV256 units from exact source bounds, not latency."""
    qlen=4 if case.mode=="q4" else 1
    counts={"history":0,"window":0,"current":0}
    nonempty={key:0 for key in counts}
    for context in case.contexts:
        widths={"history":max(0,min(context,max(reuse.SINK,context+qlen-reuse.RECENT))-min(reuse.SINK,context)),
            "window":min(reuse.SINK,context)+context-min(context,max(reuse.SINK,context+1-reuse.RECENT)),
            "current":qlen}
        for source,width in widths.items():
            split_width=math.ceil(width/case.splits)
            for split in range(case.splits):
                active=max(0,min(width,(split+1)*split_width)-min(width,split*split_width))
                counts[source]+=case.heads*math.ceil(active/256)
                nonempty[source]+=case.heads*int(active>0)
    return {"kv256_units":counts,"nonempty_tasks":nonempty,
        "task_slots":case.padded*case.heads*3*case.splits,
        "scope":"exact_source_bounds_and_query_owners_not_device_time"}


def _location(fixture, position):
    physical = fixture["page_assignments"][0][position // reuse.BLOCK_TOKENS]
    inpage = position % reuse.BLOCK_TOKENS
    row = position if position < reuse.SINK else reuse.SINK + inpage % (reuse.RECENT + reuse.SPECULATIVE)
    return physical, row


def make_case(torch, case):
    qlen = 4 if case.mode == "q4" else 1
    shape = reuse.Shape(case.name, (qlen,)*len(case.contexts), case.contexts, 256, case.heads,
        case.splits, False, padded_tokens=case.padded, slot_context=case.mode=="q1")
    fixture = reuse.make_fixture(torch, shape)
    tensors = fixture["cpu"]
    checks = []
    if case.fault:
        if case.contexts != (511,) or case.heads != 1 or case.splits != 1:
            raise ValueError("fault order goldens use one request/head at context511, S1")
        if case.fault == "bad_page":
            # Column2 covers old window positions256..383, while q1's actual
            # current slot511 remains in column3, preserving slot-context prep.
            tensors["table"][0,2] = -1
            checks = [(1,0,1),(1,1,1)]
        else:
            tag_position = 500 if case.fault=="tag_then_vnan" else 501
            tensors["tags"][_location(fixture,tag_position)] = -1
            if case.fault in {"vnan_then_tag","tag_then_vnan"}:
                nan_position = 501 if case.fault=="tag_then_vnan" else 500
                physical, row = _location(fixture,nan_position)
                tensors["wv"][physical,row,0,0] = float("nan")
            elif case.fault != "bad_tag":
                raise ValueError("unknown window fault")
            # Source1 task1, final KV tile/lane0. V-only NaN leaves QK/softmax
            # finite, so native row error ordering gives exactly 4 or 2.
            checks = [(1,0,2 if case.fault=="tag_then_vnan" else 4)]
    tensors["raw_striped"] = raw_to_striped(fixture)
    n, hq, d = fixture["tokens"], shape.heads, shape.dim
    output=torch.zeros(n,hq,d,dtype=torch.float32)
    lse=torch.full((n,hq),-float("inf"),dtype=torch.float32)
    for token,(values,logs) in fixture["expected"].items():
        output[token],lse[token] = values,logs
    positions=torch.full((n,),-1,dtype=torch.int64)
    begin=0
    for context in case.contexts:
        positions[begin:begin+qlen] = torch.arange(context,context+qlen)
        begin += qlen
    tensors.update(expected_output=output,expected_lse=lse,expected_positions=positions,
        required_status_checks=torch.tensor(checks,dtype=torch.int64).reshape(-1,3))
    fixture["case"] = case
    return fixture


def _bytes(tensor):
    value=tensor.contiguous()
    return ctypes.string_at(value.data_ptr(),value.numel()*value.element_size())


def export(directory, *, cores=2, include_large=False, include_long_single=False):
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if not 1<=cores<=32:raise ValueError("invalid actual CPU/CAModel core count")
    acceptance=json.loads((ROOT/"configs/acceptance.json").read_text())
    if acceptance.get("frozen_before_measurement") is not True:raise ValueError("frozen oracle policy required")
    rows=[]
    for case in case_plan(include_large,include_long_single):
        fixture=make_case(torch,case);spec=fixture["spec"]
        case_cores=case.cores if case.cores is not None else cores
        path=directory/case.name;path.mkdir(parents=True,exist_ok=True)
        tensors=fixture["cpu"];digest=hashlib.sha256()
        for key,tensor in tensors.items():
            content=_bytes(tensor);(path/f"{key}.bin").write_bytes(content)
            if key not in {"expected_output","expected_lse","expected_positions","required_status_checks"}:
                digest.update(key.encode()+b"\0"+content)
        shape=(fixture["tokens"],spec.heads,spec.kv_heads,256,case.contexts[0],reuse.SINK,reuse.RECENT,
            reuse.SPECULATIVE,reuse.BLOCK_TOKENS,fixture["blocks"],reuse.PREFIX,fixture["stride"],
            case_cores,len(case.contexts),case.splits,tensors["table"].shape[1])
        (path/"shape.txt").write_text(" ".join(map(str,shape))+"\n")
        (path/"case_flags.txt").write_text(f"{int(case.fault is not None)}\n")
        tolerance=acceptance["fused_attention"]
        (path/"tolerance.txt").write_text(f"{tolerance['atol']:.10g} {tolerance['rtol']:.10g}\n")
        rows.append({"name":case.name,"mode":case.mode,"path":str(path.resolve()),"cores":case_cores,
            "run_default":case.run_default,"expected_error":case.fault is not None,
            "fault":case.fault,"required_status_checks":tensors["required_status_checks"].tolist(),
            "input_sha256":digest.hexdigest(),"actual_tokens":fixture["actual_tokens"],
            "padded_tokens":fixture["tokens"],"hkv":spec.kv_heads,"ring_tokens":reuse.RECENT+reuse.SPECULATIVE,
            "source_work":source_work(case),
            "input_bytes":{key:tensor.numel()*tensor.element_size() for key,tensor in tensors.items()
                if key not in {"expected_output","expected_lse","expected_positions","required_status_checks"}},
            "measurement_scope":"single_request_mechanism_not_decode32_or_whole_run"
                if len(case.contexts)==1 and max(case.contexts)>=20000 else "bounded_fixture"})
        print(f"WINDOW_BATCH_FIXTURE case={case.name} core_count={case_cores}",flush=True)
    manifest={"status":"generated_cpu_only","schema":1,"cases":rows,
        "baseline":"current_production_attention_cv_striped_decode_simd.cpp",
        "comparison":"same_input_same_cores_all_partial_lse_status_bits",
        "oracle":"independent_PR_reference_frozen_tolerance_for_valid_cases",
        "npu":"not_run","full_model_performance":"not_established"}
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"cases.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/window_batch_decode")
    parser.add_argument("--cores",type=int,default=2)
    parser.add_argument("--include-large",action="store_true")
    parser.add_argument("--include-long-single",action="store_true")
    args=parser.parse_args()
    export(args.output,cores=args.cores,include_large=args.include_large,include_long_single=args.include_long_single)
