"""Bounded startup excerpts and optional read-only Linux/NPU process snapshot.

Archive #94/#95/#118/#123/#125/#158: keep the first error, selected devices,
source lines and real command failures. Never infer ownership or kill/reset a
process, import torch, scan native source, or alter the serving configuration.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
FAULT = re.compile(r"Traceback|\bERROR\b|\bFAILED\b|\bFATAL\b|Error:|Exception:|507035|family=")
KEY = re.compile(r"Free memory on device|\b\w*(?:Error|Exception):|family=|"
                 r"PERF_RUNTIME_CONFIG|\[oscar\] devices=|SERVE_MODE|"
                 r"OSCAR_STARTUP|OSCAR_GUARD|Address already in use|out of memory", re.I)
MEMORY = re.compile(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\).*?"
                    r"utilization \(([\d.]+), ([\d.]+) GiB\)")
# #142/#158: a completed but slow warmup may contain no ERROR at all.
# Native monitor.py / gpu_model_runner.py / core.py log nested durations;
# retain them per emitter, never sum ranks or treat compile as disjoint time.
STARTUP = re.compile(r"torch\.compile took|Initial profiling/warmup run took|"
                     r"Graph capturing finished|Capturing (?:CUDA|ACL|NPU) graphs|"
                     r"init engine \(|Starting to load model|Loading model weights took|"
                     r"Model loading took|Start warmup|Warmup finished|"
                     r"Application startup complete|Uvicorn running|Starting vLLM API server", re.I)
DURATIONS = (
    ("torch_compile", re.compile(r"torch\.compile took ([\d.]+) s in total")),
    ("initial_profile_warmup", re.compile(r"Initial profiling/warmup run took ([\d.]+) s")),
    ("graph_capture", re.compile(r"Graph capturing finished in ([\d.]+) secs")),
    ("engine_init_including_compile", re.compile(r"init engine \(.*?\) took ([\d.]+) s")),
)
PID = re.compile(r"\bpid[=:]\s*(\d+)", re.I)
ENV_KEYS = ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
            "OSCAR_TARGET_CONFIG", "OSCAR_ENABLED", "VLLM_PLUGINS",
            "OSCAR_GUARD_HOST_SYNC", "OSCAR_DEBUG_SYNC", "OSCAR_DEBUG_MIN_TOKENS",
            "OSCAR_TIMING", "OSCAR_TIMING_STDERR", "OSCAR_PROFILER",
            "OSCAR_TRACE_DIR", "OSCAR_PROGRESS_INTERVAL_SECONDS",
            "OSCAR_DEVICE_TIMING_CONTROL", "OSCAR_PASSIVE_TIMING_CONTROL",
            "ASCEND_LAUNCH_BLOCKING", "CUDA_LAUNCH_BLOCKING")
CONFIG_KEYS = ("devices", "port", "soc_version", "tensor_parallel_size",
               "data_parallel_size", "gpu_memory_utilization", "max_model_len",
               "max_num_seqs", "enable_prefix_caching", "use_native_token_budget",
               "speculative_config", "compilation_config")


def clipped(text, limit):
    data = text.encode("utf-8")
    marker = b"\n[TRUNCATED: output limit; not complete evidence]\n"
    if len(data) <= limit:
        return text
    return data[:limit - len(marker)].decode("utf-8", errors="ignore") + marker.decode()


def scan_log(path):
    """Stream the file, retaining bounded head/first-fault/key/tail evidence."""
    head, context, keys, memory, startup_first, durations = [], [], [], [], [], []
    startup_last = deque(maxlen=24)
    before, tail = deque(maxlen=12), deque(maxlen=16)
    pids, digest = set(), hashlib.sha256()
    first_fault = None
    count = byte_count = key_count = memory_count = long_lines = startup_count = duration_count = 0
    with path.open("rb") as source:
        while raw := source.readline(1024 * 1024):
            digest.update(raw)
            byte_count += len(raw)
            # Drain oversized logical lines without allocating them in full.
            oversized = len(raw) == 1024 * 1024 and not raw.endswith(b"\n")
            piece = raw
            while not piece.endswith(b"\n") and len(piece) == 1024 * 1024:
                piece = source.readline(1024 * 1024)
                digest.update(piece)
                byte_count += len(piece)
            count += 1
            long_lines += int(oversized)
            line = ANSI.sub("", raw.decode("utf-8", errors="replace")).rstrip("\r\n")
            pids.update(int(pid) for pid in PID.findall(line) if len(pids) < 128)
            row = f"{count}: {clipped(line, 5000)}"
            if len(head) < 8:
                head.append(row)
            tail.append(row)
            if STARTUP.search(line):
                startup_count += 1
                if len(startup_first) < 24:
                    startup_first.append(row)
                else:
                    startup_last.append(row)
            for phase, pattern in DURATIONS:
                if match := pattern.search(line):
                    duration_count += 1
                    if len(durations) < 32:
                        durations.append({"phase": phase, "seconds": float(match[1]), "line": count,
                                          "pid": next(iter(PID.findall(line)), None)})
            if KEY.search(line):
                key_count += 1
                if len(keys) < 40:
                    keys.append(row)
            if match := MEMORY.search(line):
                memory_count += 1
                if len(memory) < 16:
                    memory.append({"line": count, "pid": next(iter(PID.findall(line)), None),
                                   **dict(zip(("free_gib", "total_gib", "utilization", "required_gib"),
                                              map(float, match.groups())))})
            if first_fault is None and FAULT.search(line):
                first_fault = count
                context.extend(before)
            if first_fault is not None and count < first_fault + 100:
                context.append(row)
            before.append(row)
    return {"file": str(path.resolve()), "lines": count, "bytes_scanned": byte_count,
            "sha256": digest.hexdigest(), "oversized_lines_prefix_only": long_lines,
            "first_fault_line": first_fault, "pids_seen": sorted(pids),
            "memory_gate_count": memory_count, "memory_gates": memory,
            "startup_markers_matched": startup_count,
            "startup_markers_omitted": max(0, startup_count - len(startup_first) - len(startup_last)),
            "startup_duration_count": duration_count, "startup_durations": durations,
            "duration_scope": "native logged durations, per PID; may overlap/nest; no cross-rank or phase summation",
            "startup_markers": startup_first + list(startup_last),
            "key_matches": key_count, "keys_retained": len(keys),
            "head": head, "keys": keys, "first_fault_context": context, "tail": list(tail)}


def command_snapshot(argv, timeout=10):
    # Fixed read-only commands; no shell. Spool output to avoid an unbounded pipe.
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
            status = {"returncode": result.returncode}
        except (OSError, subprocess.TimeoutExpired) as error:
            status = {"error": str(error)}
        output.seek(0)
        raw = output.read(32769)
    return {"argv": argv, **status, "truncated": len(raw) > 32768,
            "text": raw[:32768].decode("utf-8", errors="replace")}


def proc_identity(pid, proc_root=Path("/proc")):
    result = {"pid": pid}
    directory = proc_root / str(pid)
    for name in ("comm", "status", "environ"):
        try:
            with (directory / name).open("rb") as source:
                raw = source.read(65536)
            if name == "environ":
                env = dict(item.split("=", 1) for item in raw.decode(errors="replace").split("\0") if "=" in item)
                result["environment_allowlist"] = {k: env[k] for k in ENV_KEYS if k in env}
                result["environment_read_limit_reached"] = len(raw) == 65536
            elif name == "status":
                result["namespace_status"] = [line for line in raw.decode(errors="replace").splitlines()
                                              if line.startswith(("Name:", "Pid:", "PPid:", "NSpid:", "Uid:",
                                                                  "Cpus_allowed_list:", "Mems_allowed_list:"))]
            else:
                result[name] = raw.decode(errors="replace").strip()
        except OSError as error:
            result[name + "_error"] = str(error)
    for label, link in (("cwd", "cwd"), ("stdout", "fd/1"), ("stderr", "fd/2")):
        try:
            result[label] = os.readlink(directory / link)
        except OSError as error:
            result[label + "_error"] = str(error)
    return result


def snapshot(pids):
    smi = command_snapshot(["npu-smi", "info"])
    processes = command_snapshot(["ps", "-eo", "pid,ppid,pgid,etimes,comm"])
    selected_rows = []
    selected_pids = set(pids)
    for line in processes["text"].splitlines():
        fields = line.split(None, 4)
        if len(fields) != 5 or not fields[0].isdigit():
            continue
        if int(fields[0]) in pids or re.search(r"python|vllm|worker|engine", fields[4], re.I):
            selected_rows.append(line)
            if len(selected_pids) < 64:
                selected_pids.add(int(fields[0]))
    processes["text"] = "PID PPID PGID ELAPSED_SECONDS COMMAND_NAME\n" + "\n".join(selected_rows[:64])
    processes["filtered_rows_omitted"] = max(0, len(selected_rows) - 64)
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "current snapshot, NOT historical occupancy; PIDs may have exited/reused; container PID namespaces may differ",
            "npu_smi": smi, "processes": processes,
            "identities": [proc_identity(pid) for pid in sorted(selected_pids)[:64]]}


def selected_json(path, keys):
    try:
        with path.open("rb") as source:
            raw = source.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("metadata exceeds 1 MiB; not read")
        data = json.loads(raw)
        return {"path": str(path.resolve()), "selected": {k: data[k] for k in keys if k in data}}
    except (OSError, ValueError, TypeError) as error:
        return {"path": str(path), "error": str(error)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", default=[], help="explicit log file, repeatable (max 8); no directory scan")
    parser.add_argument("--run-dir", type=Path, help="only read effective-target.json and status.json here")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--snapshot", action="store_true", help="read npu-smi, ps and allowlisted /proc identity; does not initialize NPU")
    parser.add_argument("--pid", type=int, action="append", default=[], help="extra live PID for identity; never killed")
    parser.add_argument("--output", type=Path, required=True, help="new compact text file, never overwrite")
    args = parser.parse_args(argv)
    if len(args.log) > 8 or len(args.pid) > 32 or any(pid <= 0 for pid in args.pid):
        parser.error("at most 8 logs and 32 positive PIDs")
    if not args.log and not args.snapshot:
        parser.error("provide --log and/or --snapshot")
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    sections = ["OSCAR STARTUP EVIDENCE v1\n" + datetime.now(timezone.utc).isoformat(),
                "Read-only diagnostics; excerpts are bounded; missing evidence is unknown.\n"
                "Log PID labels do not prove physical-card mapping or process ownership.\n"
                "Direct install_serve exec leaves status=serving even when service later fails; use console log/exit code."]
    metadata = [selected_json(args.config, CONFIG_KEYS)]
    if args.run_dir:
        metadata += [selected_json(args.run_dir / "effective-target.json", CONFIG_KEYS),
                     selected_json(args.run_dir / "status.json", ("status", "failed_phase", "error", "target_devices", "port", "effective_config"))]
    sections.append("CONFIG (files now on disk, not proof of worker environment)\n" + json.dumps(metadata, ensure_ascii=False, indent=2))
    sections.append("COLLECTOR SHELL ENV (not worker env)\n" + json.dumps({k: os.environ[k] for k in ENV_KEYS if k in os.environ}, ensure_ascii=False))
    reports, failed = [], False
    pids = set(args.pid)
    for path in args.log:
        try:
            report = scan_log(path)
            pids.update(report["pids_seen"])
        except OSError as error:
            failed = True
            report = {"file": str(path), "read_error": str(error)}
        reports.append(report)
    for report in reports:
        summary = {k: v for k, v in report.items() if k not in ("head", "keys", "startup_markers", "first_fault_context", "tail")}
        sections.append("LOG SUMMARY\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    if args.snapshot:
        sections.append("LIVE SNAPSHOT\n" + clipped(json.dumps(snapshot(pids), ensure_ascii=False, indent=2), 65536))
    for report in reports:
        lines = ["EXCERPTS " + report["file"]]
        for key in ("keys", "startup_markers", "head", "first_fault_context", "tail"):
            lines += ["--- " + key + " ---", *report.get(key, [])]
        sections.append(clipped("\n".join(lines), 20000))
    content = clipped("\n\n".join(sections) + "\n", 256 * 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(content)
    print(f"[oscar] startup evidence: {args.output} ({len(content.encode('utf-8'))} bytes); send this file only")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
