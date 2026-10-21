"""Archive #94/#95/#125/#148: collect only this local VM task's real evidence.

Original failed phases remain in the bundle. Host/CPU/CANN findings cannot
be promoted to NPU, graph, LongBench or performance acceptance.
"""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GUEST = "/tmp/oscar-official-math-20261021"
PROGRAM = r'''
from pathlib import Path
import sys, tarfile, hashlib, json, io
root=Path('/tmp/oscar-official-math-20261021')
paths=[]
for name in ('logs/official-full-build','logs/official-full-build-v2'):
 paths.extend(p for p in (root/name).rglob('*') if p.is_file())
evidence_hashes={}
for directory in (root/'reports').glob('official-cpu-*'):
 paths.extend(p for p in directory.rglob('*') if p.is_file() and (p.suffix in ('.json','.log','.txt')))
 for p in directory.rglob('*'):
  if p.is_file() and p.suffix in ('.bin','.pt'):
   evidence_hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('reports/build.json','build/ascendc/build_manifest.json','build/ascendc/oscar_build_signature.json'):
 p=root/name
 if p.is_file():paths.append(p)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as bundle:
 for p in sorted(set(paths)):bundle.add(p,arcname=str(p.relative_to(root)),recursive=False)
 payload=json.dumps(evidence_hashes,indent=2).encode()
 info=tarfile.TarInfo('reports/official-cpu-files.json');info.size=len(payload)
 bundle.addfile(info,io.BytesIO(payload))
'''


def main():
    completed = subprocess.run(["limactl", "shell", "--workdir", GUEST, "oscar", "--", "python3", "-c", PROGRAM],
                               check=True, capture_output=True, timeout=60)
    archive = HERE / "vm_evidence.tar.gz"
    archive.write_bytes(completed.stdout)
    folder = HERE / "vm_evidence"; folder.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:gz") as bundle:
        bundle.extractall(folder, filter="data")
    build = json.loads((folder / "reports/build.json").read_text())
    numeric = {}
    summaries = []
    for summary in sorted((folder / "reports").glob("official-cpu-*/summary.json")):
        value = json.loads(summary.read_text())
        summaries.append({"path": str(summary.relative_to(HERE)), "status": value["status"], "failed_phase": value.get("failed_phase")})
        for case in value["cases"]:
            logfile = summary.parent / "logs" / (case["case"] + "-execute.log")
            rows = []
            for line in logfile.read_text(errors="replace").splitlines():
                if line.startswith('{"backend"'):
                    rows.append(json.loads(line))
            if not rows:
                raise ValueError("passed case lacks the kernel harness result: " + str(logfile))
            numeric[case["case"]] = {**rows[-1], **{k: v for k, v in case.items() if k.startswith("empty_history_")}, "log": str(logfile.relative_to(HERE))}
    wanted = {"q4_mode0", "q4_mode1", "q1_mode1", "c4_mode0", "c4_mode1", "c16_mode0", "c16_mode1",
              "q4_mode1_fault", "c4_mode1_fault", "c16_mode1_fault", "c16_mode0_empty_history"}
    compiled = build["configuration"]["source"]
    relevant = [p for p in (ROOT / "csrc/kernels").glob("*official*.cpp")]
    relevant += list((ROOT / "csrc/include").glob("oscar_official*"))
    relevant += [ROOT / "csrc/official_attention_bindings.cpp", ROOT / "csrc/CMakeLists.txt", ROOT / "csrc/torch_bindings.cpp"]
    hashes = {}
    for path in relevant:
        relative = str(path.relative_to(ROOT / "csrc"))
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded = compiled.get(relative)
        hashes[str(path.relative_to(ROOT))] = {"sha256": current, "matches_completed_build": current == recorded}
    report = {"scope": "local Lima CANN full extension and official SDK CPU-debug functional validation",
              "build": build["build"], "build_signature": build["signature"], "binary_sha256": build["sha256"],
              "production_source_binding": hashes, "cpu_cases": numeric,
              "cpu_matrix_complete": set(numeric) == wanted,
              "q4_fault_full_status": json.loads((folder / "reports/official-cpu-v3/q4_fault_full_status.json").read_text())
                  if (folder / "reports/official-cpu-v3/q4_fault_full_status.json").exists() else None,
              "original_runs": summaries,
              "first_failure_explanations": [
                  "First CANN source stage omitted the existing current_only include dependency; rc2 logs retained, complete v2 build passed after staging it.",
                  "CPU v1/v2 C4 harness retained the ancestor 337-token fixture's x4 count conversion; the new 85-token double-KV fixture declares two clusters. Actual2/members8/skips8 matched geometry. Removed only that harness conversion; original failed outputs/logs retained."],
              "target_npu": "not_run", "graph_capture": "not_run", "graph_replay": "not_run",
              "model_quality": "not_run", "performance": "not_run_user_prohibited",
              "archive_sha256": hashlib.sha256(completed.stdout).hexdigest()}
    report["status"] = "passed" if report["build"] == "passed" and report["cpu_matrix_complete"] and all(v["matches_completed_build"] for v in hashes.values()) else "incomplete"
    (HERE / "kernel_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(numeric), "report": str(HERE / "kernel_validation.json")}))


if __name__ == "__main__":
    main()
