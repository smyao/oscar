"""Archive #126/#145/#148/#164: paired complete C16 CPU function validation.

D.4: bounded fixtures, never performance or full-history production restore.
Uses frozen v4 oracle and unchanged actual prepare. No relaxed thresholds.
Run from a staged source root containing baseline.cpp from the recorded base.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("OSCAR_C16_REPORT", str(ROOT / "reports/long-context-c16")))


def main():
    from tools.phase import run_phase, atomic_json
    import torch
    from tools import probe_history_reuse as reuse
    from tools.aligned_fixture import make_fixture
    from tools.official_reference import CACHE_FORMAT, rotate_bf16
    from csrc.experiments.official_math_20261021.fixture import dump, golden
    torch.set_num_threads(1)
    sdk = Path('/usr/local/Ascend/cann-9.1.0')
    os.environ['LD_LIBRARY_PATH'] = ':'.join(str(sdk / p) for p in
        ('lib64', 'tools/tikicpulib/lib', 'tools/tikicpulib/lib/Ascend910B4')) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    OUT.mkdir(parents=True, exist_ok=False)
    report = {"backend": "official_CANN_CPU_debug", "performance": "not_run", "NPU": "not_run", "cases": []}

    def phase(name, cmd, timeout=180, env=None):
        p = run_phase(name, cmd, cwd=ROOT, log_dir=OUT / "logs", timeout=timeout, env=env)
        if p.returncode:
            report.update(status="failed", failed_phase=name, returncode=p.returncode, log=p.log)
            atomic_json(OUT / "summary.json", report)
            raise RuntimeError(f"{name}: rc={p.returncode}")
        # SDK successful-operation traces can be gigabytes. Checks, oracle
        # logs and outputs remain enabled and retained. Only after rc0 and
        # owned-process completion remove that phase's disposable trace.
        for trace in (ROOT / 'npuchk', ROOT / 'cceprint'):
            if trace.is_dir(): shutil.rmtree(trace)

    phase("configure", ["cmake", "-S", "csrc/experiments/official_c16_20261027", "-B", "build/long-context-c16", "-DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0"])
    phase("build", ["cmake", "--build", "build/long-context-c16", "--parallel", "2"], 900)
    binary = str(ROOT / "build/long-context-c16/oscar_long_context_c16_cpu")
    # The initial q337/context577 control exhausted its bounded SDK trace
    # resources. Retain that attempt; separate mature/current and multi-KV
    # coverage without extending deadlines or changing the numerical gate.
    cases = [("mature_history_current", 337, 320, 1, 1, 0, False),
             ("independent_multiple_kv", 22, 577, 16, 1, 0, False),
             ("full_window_two_heads", 22, 320, 256, 2, 0, False),
             ("legacy_mode1_current", 337, 96, 16, 1, 1, False),
             ("metadata_fault", 22, 577, 16, 1, 0, True),
             ("all_masked_history", 22, 96, 256, 1, 0, False)]
    for name, active, context, recent, hk, mode, fault in cases:
        folder = OUT / name; folder.mkdir()
        clusters=active//336
        shape = reuse.Shape(name, (active,), (context,), 256, hk, 1, bool(clusters),
            speed_gate=False, recent_tokens=recent, padded_tokens=active+1,
            corrupt_metadata=fault, corrupt_position=80)
        fixture = make_fixture(torch, shape); data = fixture["cpu"]
        data["q"] = data["qr"].clone()
        for key, matrix in (("ck", fixture["rotation_k"]), ("cv", data["rv"])):
            data[key] = rotate_bf16(data[key], matrix)
        valid = data["tags"] >= 0
        for key, matrix in (("wk", fixture["rotation_k"]), ("wv", data["rv"])):
            data[key][valid] = rotate_bf16(data[key][valid], matrix)
        for key in ("q", "qr", "ck", "cv"):
            data[key][active:] = float("nan")
        fixture.update(cache_format=CACHE_FORMAT, cached_quant_ends=[max(min(64, context), context+1-recent)])
        positions = torch.full((active+1,), -1, dtype=torch.int64)
        positions[:active] = torch.arange(context, context+active)
        data["expected_positions"] = positions
        checks = [((qb*hk+hk-1)*3, 1, 3) for qb in range(0,active,21)] if fault else []
        data["required_status_checks"] = torch.tensor(checks,dtype=torch.int64).reshape(-1,3)
        torch.save(fixture, folder / "fixture.pt")
        for key, value in data.items(): dump(folder / (key+".bin"), value)
        dump(folder / "raw_striped.bin", data["raw"])
        dims=(active+1,6*hk,hk,256,context,64,recent,3,reuse.BLOCK_TOKENS,fixture["blocks"],reuse.PREFIX,fixture["stride"],2,1,1,data["table"].shape[1])
        (folder / "shape.txt").write_text(" ".join(map(str,dims))+"\n")
        (folder / "case_flags.txt").write_text(str(int(fault))+"\n")
        (folder / "cluster_flags.txt").write_text(f"0 {hk*clusters} {80 if fault else -1} -1\n")
        (folder / "expected_counts.txt").write_text(str(((active+20)//21-16*clusters)*hk)+'\n')
        outputs=[]
        for arm in ("baseline", "candidate"):
            dest=folder / arm; dest.mkdir(); outputs.append(dest)
            env={**os.environ, "OSCAR_C16_CONTROL":str(int(arm=="baseline"))}
            command=[binary,"c16",str(folder),str(dest),str(mode)]
            phase(name+"-"+arm+"-prepare",command+["prepare-only"],env=env)
            golden(folder,dest,mode)
            phase(name+"-"+arm,command,env=env)
        fields=["tasks", "partial", "lse", "status", "cluster_stats"]
        if not fault: fields += ["output", "output_lse"]
        hashes={}
        for field in fields:
            a,b=(p/(field+".bin") for p in outputs)
            if a.read_bytes()!=b.read_bytes(): raise RuntimeError(name+": bitwise mismatch "+field)
            hashes[field]=hashlib.sha256(a.read_bytes()).hexdigest()
        report["cases"].append({"name":name,"queries":active,"context":context,"mode":mode,"heads":hk,"recent":recent,"bitwise":"passed","hashes":hashes,"oracle":"exact_error_domain" if fault else "v4_frozen_0.005"})
        atomic_json(OUT / "summary.json",report)
    report["status"]="passed"; atomic_json(OUT/"summary.json",report)
    print(json.dumps({"status":"passed","cases":len(cases),"performance":"not_run"}))


if __name__ == "__main__": main()
