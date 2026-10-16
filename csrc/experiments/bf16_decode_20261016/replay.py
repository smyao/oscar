"""Archive #126/#145/#148/#154: complete same-library FP32/BF16 decode arms.

D.4: time the full fused CV including final Rv, not only BF16 MMAD. Retain
all source tasks, input guards, independent declared oracle and full profiles.
The only intentional input-format difference is initialization-only Rv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "csrc/experiments/window_batch_decode/replay_camodel.py"
TEMPLATE_SHA = "2b435e186ad6aa8112d620fa83299cacc7c9b7a60d02f77b59a1b43e9447883c"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from csrc.experiments.c16_l1_reuse_20261015.safe_logging_runner import transform


def sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()


def adapted():
    raw=TEMPLATE.read_bytes()
    if sha(raw)!=TEMPLATE_SHA:
        raise ValueError("complete qualified replay changed")
    text=transform(raw.decode())
    def edit(old,new):
        nonlocal text
        if text.count(old)!=1:
            raise ValueError("BF16 replay anchor changed: "+old[:100])
        text=text.replace(old,new)
    edit("'rv':'rv.bin'", "'rv':('rv_nz.bin' if mode=='q4' and args.variant=='baseline' else 'rv.bin')")
    edit("    inputs['tasks']=(exported/'tasks.bin').read_bytes()",
        "    if args.variant=='candidate':inputs['rv']=(exported/'rv_bf16.bin').read_bytes()\n"
        "    inputs['tasks']=(exported/'tasks.bin').read_bytes()")
    edit("'rv':d*d*4,", "'rv':d*d*(2 if args.variant=='candidate' else 4),")
    edit("    cpu_paths={name:exported/(name+'.bin') for name in ('partial','lse','status')}",
        "    if args.variant=='candidate':\n"
        "        oracle={name:(exported/path).read_bytes() for name,path in {\n"
        "            'output':'bf16_expected_output.bin','lse':'bf16_expected_output_lse.bin'}.items()}\n"
        "    cpu_paths={name:exported/(name+'.bin') for name in ('partial','lse','status')}")
    edit("    for key,want in cpu_outputs.items():\n        if outputs[key]!=want:failures.append(key+'_cpu_baseline_bitwise')",
        "    for key,want in cpu_outputs.items():\n"
        "        if key=='status':\n"
        "            if outputs[key]!=want:failures.append(key+'_cpu_exact')\n"
        "        elif args.variant=='baseline':\n"
        "            if outputs[key]!=want:failures.append(key+'_cpu_baseline_bitwise')\n"
        "        else:\n"
        "            for actual,expected in zip(floats(outputs[key]),floats(want)):\n"
        "                if actual==expected:continue\n"
        "                if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected)>.005+.005*abs(expected):\n"
        "                    failures.append(key+'_cpu_numerical');break\n"
        "    if args.variant=='candidate':\n"
        "        for key,path in [('partial','bf16_expected_partial.bin'),('lse','bf16_expected_lse.bin')]:\n"
        "            expected=(exported/path).read_bytes()\n"
        "            if len(expected)!=len(outputs[key]):raise ValueError('independent BF16 partial size')\n"
        "            for actual,want in zip(floats(outputs[key]),floats(expected)):\n"
        "                if actual==want:continue\n"
        "                if not math.isfinite(actual) or not math.isfinite(want) or abs(actual-want)>.005+.005*abs(want):\n"
        "                    failures.append(key+'_independent_BF16_oracle');break")
    if "os.dup2" in text or "/proc/self/fd" in text:
        raise ValueError("unsafe historical logger survived")
    return text


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--family",choices=("q4","q1"),required=True)
    parser.add_argument("--arm",choices=("baseline","candidate"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--fixture-root",type=Path,required=True)
    parser.add_argument("--export-root",type=Path,required=True)
    args=parser.parse_args();source=adapted()
    config=args.config.resolve()
    if os.environ.get("CAMODEL_CONFIG_PATH")!=str(config):
        raise ValueError("official simulator config must be set before runtime load")
    if json.loads((config/"config.json").read_text())["LOG"]["core_enable_mask"]!=["0x0"]:
        raise ValueError("official zero-dump LOG mask required")
    plan=args.output.with_name(args.output.name+"-plan.json")
    plan.parent.mkdir(parents=True,exist_ok=True)
    if plan.exists() or args.output.exists():raise ValueError("fresh measurement arm required")
    binding=json.loads((ROOT/"pair_source_binding.json").read_text())
    for path,want in binding["sources"].items():
        if sha((ROOT/path).read_bytes())!=want:raise ValueError("compiled source drift: "+path)
    if sha(args.library.read_bytes())!=binding["library_sha256"]:raise ValueError("library changed")
    plan.write_text(json.dumps({"source_binding":binding,"original_replay_sha256":TEMPLATE_SHA,
        "adapted_replay_sha256":sha(source),"family":args.family,"arm":args.arm,
        "deadline_s":600,"scope":"all24 tasks; complete CV including same-kernel finalRv; excludes merge/copy/service",
        "intentional_contract_change":"BF16 Q/K/V/P/normalized history/Rv operands with FP32 accumulation",
        "old_FP32_reference":"baseline only; candidate uses independently frozen BF16v2 task oracle",
        "descriptor_mutation":False},indent=2)+"\n")
    previous=sys.argv
    sys.argv=[str(__file__),"--root",str(ROOT),"--output",str(args.output),"--case",args.family+"_finite_cpu",
        "--variant",args.arm,"--fixture-root",str(args.fixture_root),"--export-root",str(args.export_root),
        "--library",str(args.library),"--simulator-config-dir",str(config),
        "--baseline-q4","oscar_attention_cv_hybrid_l1_decode_kernel",
        "--baseline-q1","oscar_attention_cv_mdl_decode_q1_kernel",
        "--candidate-q4","oscar_attention_cv_bf16_decode_kernel",
        "--candidate-q1","oscar_attention_cv_bf16_decode_q1_kernel",
        "--baseline-workspace-bytes-per-core","1146880","--candidate-workspace-bytes-per-core","1146880",
        "--cpu-reference","required","--instruction-budget-bytes","0"]
    try:
        namespace={"__name__":"bf16_decode_complete_replay","__file__":str(__file__)}
        exec(compile(source,str(TEMPLATE),"exec"),namespace)
        result=namespace["main"]()
        if result:raise SystemExit(result)
    finally:
        sys.argv=previous


if __name__=="__main__":
    main()
