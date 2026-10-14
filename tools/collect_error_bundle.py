#!/usr/bin/env python3
"""Archive #94/#95/#118/#123/#125/#157: bounded error evidence, not a verdict.

Standalone Python 3; no project imports, NPU initialization, subprocesses or
service changes. Read saved terminal text (or pasted stdin) and existing logs.
The default output is at most 48 KiB, below the observed 64 KiB paste cutoff.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DEVICE_FAULT = re.compile(r"aive[cf] error|timeout or trap|OSCAR.*(?:Trap|trapped)|"
                          r"\b(?:error\s*code|errcode|error)\b[\"']?\s*[:=]?\s*(?:is\s+)?5070\d{2}\b|"
                          r"\b\w*(?:Error|Exception):[^\n]*?(?<![\w.])5070\d{2}(?![\w.])", re.I)
FAULT = re.compile(r"\b(?:ERROR|FATAL|FAILED|Segfault)\b|\bTraceback\s+\(most recent call last\):|"
                   r"\b\w*(?:Error|Exception):|aive[cf] error|"
                   r"timeout or trap|site=.*family=|\"status\"\s*:\s*\"failed\"", re.I)
DETAIL = re.compile(r"\b\w*(?:Error|Exception):|aive[cf] error|"
                    r"timeout or trap|site=.*family=|fault kernel", re.I)
INFO = re.compile(r"PERF_RUNTIME_CONFIG|SERVE_MODE|devices=.*port=|"
                  r"max_num_batched_tokens|Avg prompt throughput|SpecDecoding metrics|"
                  r"OSCAR_GUARD_HOST_SYNC|OSCAR_DEBUG_SYNC|"
                  r'"event"\s*:\s*"(?:prefix_cache_contract|prefix_chunk|cache_layout|'
                  r'attention_progress|graph_replay_progress)"')
STAMP = re.compile(r"\b(?:20\d\d[-/])?\d\d[-/]\d\d[ T]\d\d:\d\d:\d\d\b")
PREFIX = re.compile(r"^\([^)]*\)\s+(?:ERROR|WARNING|INFO|DEBUG)\s+"
                    r"\S+\s+\S+\s+\[[^]]+\]\s?")
ROLES = ("worker_or_device", "engine", "other", "api")
SUFFIXES = {".log", ".txt", ".jsonl", ".json", ".out", ".err"}
MAX_LINE = 64 * 1024


def error_flags(line):
    """Inspect JSON error fields, never arbitrary timestamp/PID/hash digits.

    Also handle pretty-printed JSON fields and run_phase's RESULT records.
    The returned flags select excerpts; the original text is always retained.
    """
    text = line.strip()
    for prefix in ("RESULT ", "SERVICE_RESULT ", "SERVICE_OUTCOME "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith('"'):
        text = "{" + text.rstrip(",") + "}"
    try:
        record = json.loads(text) if text.startswith("{") else None
    except (ValueError, RecursionError):
        record = None
    if not isinstance(record, dict):
        device = bool(DEVICE_FAULT.search(line))
        return bool(FAULT.search(line)) or device, bool(DETAIL.search(line)) or device
    fault = detail = False
    pending = [record]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            for key, value in item.items():
                key = key.lower()
                if key in {"error", "error_type", "exception", "traceback", "failed_phase"} and value:
                    fault = detail = True
                elif key in {"returncode", "exit_code", "error_code", "errcode"}:
                    try:
                        failed = int(value) != 0 if value is not None else False
                    except (ValueError, TypeError, OverflowError):
                        failed = False
                    fault |= failed
                    detail |= failed
                elif key in {"status", "level", "severity"} and str(value).lower() in {"failed", "error", "fatal"}:
                    fault = detail = True
                elif key in {"message", "msg"} and isinstance(value, str):
                    device = bool(DEVICE_FAULT.search(value))
                    fault |= bool(FAULT.search(value)) or device
                    detail |= bool(DETAIL.search(value)) or device
                if isinstance(value, (dict, list)):
                    pending.append(value)
    return fault, detail


def clip(text, size):
    raw = text.encode("utf-8", errors="replace")
    marker = b"\n[... omitted by byte limit ...]\n"
    if len(raw) <= size:
        return text
    left = (size - len(marker)) * 3 // 4
    right = size - len(marker) - left
    return (raw[:left].decode("utf-8", errors="ignore") + marker.decode()
            + raw[-right:].decode("utf-8", errors="ignore"))


def role(line):
    emitter = re.match(r"^\([^)]*\)", line)
    label = emitter[0] if emitter else ""
    # An API exception often *mentions* EngineCore/Worker. Attribute the
    # emitting process, not that message text, or a missing primary looks found.
    if "APIServer" in label:
        return "api"
    if "Worker" in label:
        return "worker_or_device"
    if "EngineCore" in label:
        return "engine"
    if "EngineDeadError" in line:
        return "api"
    if DEVICE_FAULT.search(line):
        return "worker_or_device"
    return "other"


def signature(line):
    # Keep PID/rank in the signature; never merge independent rank failures.
    emitter = re.match(r"^\([^)]*\)", line)
    return (emitter[0] if emitter else "") + STAMP.sub("<time>", PREFIX.sub("", line))


def dedup(rows):
    unique = {}
    for number, text in rows:
        key = signature(text)
        if key in unique:
            unique[key][2] = number
            unique[key][3] += 1
        else:
            unique[key] = [number, text, number, 1]
    return "\n".join(f"L{first}: {text}" +
                     (f" [repeated {count}x; last L{last}]" if count > 1 else "")
                     for first, text, last, count in unique.values())


def scan(stream, *, size=None):
    """Bounded memory; regular files stop at their size when opened."""
    before, tail, info = deque(maxlen=12), deque(maxlen=6), deque(maxlen=10)
    windows, details, detail_keys = {}, [], set()
    detail_tail = deque(maxlen=8)
    counts = Counter()
    digest = hashlib.sha256()
    total = lines = oversized = detail_matches = 0
    first_stamp = last_stamp = None
    unterminated = False
    while size is None or total < size:
        raw = stream.readline(MAX_LINE if size is None else min(MAX_LINE, size - total))
        if not raw:
            break
        total += len(raw)
        digest.update(raw)
        prefix = raw
        long_line = len(raw) == MAX_LINE and not raw.endswith(b"\n")
        while long_line and raw and not raw.endswith(b"\n") and (size is None or total < size):
            raw = stream.readline(MAX_LINE if size is None else min(MAX_LINE, size - total))
            total += len(raw)
            digest.update(raw)
        oversized += int(long_line)
        unterminated = not raw.endswith(b"\n")
        lines += 1
        line = ANSI.sub("", prefix.decode("utf-8", errors="replace")).rstrip("\r\n")
        if match := STAMP.search(line):
            first_stamp = first_stamp or match[0]
            last_stamp = match[0]
        row = (lines, clip(line, 1800))
        kind = role(line)
        fault, detail = error_flags(line)
        if fault:
            counts[kind] += 1
            if kind not in windows:
                windows[kind] = {"first_error_line": lines, "rows": list(before), "left": 120}
        for window in windows.values():
            if window["left"]:
                window["rows"].append(row)
                window["left"] -= 1
        if detail:
            detail_matches += 1
            key = signature(line)
            if key not in detail_keys and len(details) < 24:
                details.append((kind, row))
                detail_keys.add(key)
            else:
                detail_tail.append((kind, row))
        if INFO.search(line):
            info.append(row)
        tail.append(row)
        before.append(row)
    return {"lines": lines, "bytes_scanned": total, "sha256_of_scanned_bytes": digest.hexdigest(),
            "oversized_lines_prefix_only": oversized, "last_line_without_newline": unterminated,
            "first_timestamp_text": first_stamp, "last_timestamp_text": last_stamp,
            "error_line_counts": dict(counts), "detail_matches": detail_matches,
            "windows": windows, "details": details + list(detail_tail),
            "info": list(info), "tail": list(tail)}


def discover(root, output, maximum):
    """Select recent files; do not follow directory or file symlinks."""
    candidates, errors, visited = [], [], 0
    def onerror(error):
        errors.append(str(error))
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        dirs[:] = sorted(d for d in dirs if not Path(directory, d).is_symlink())
        for name in sorted(files):
            visited += 1
            if visited > 20000:
                errors.append("discovery stopped at 20000 files; use --logs-dir for one run")
                break
            path = Path(directory, name)
            if path.suffix.lower() not in SUFFIXES or path.is_symlink() or path.resolve() == output:
                continue
            # Do not recursively re-ingest this tool's old output bundles.
            if name.startswith("error-bundle-"):
                continue
            try:
                meta = path.stat()
                if stat.S_ISREG(meta.st_mode):
                    candidates.append((meta.st_mtime_ns, str(path.resolve())))
            except OSError as error:
                errors.append(str(error))
        if visited > 20000:
            break
    candidates.sort(reverse=True)
    return [Path(p) for _, p in candidates[:maximum]], max(0, len(candidates) - maximum), errors


def make_report(reports, *, root, omitted, notices, max_bytes):
    manifest = []
    blocks = []
    ranks = Counter()
    for index, (name, report) in enumerate(reports, 1):
        source = f"[{index}] {name}"
        if "read_error" in report:
            manifest.append(source + " READ_ERROR: " + report["read_error"])
            continue
        ranks.update(report["error_line_counts"])
        summary = {k: v for k, v in report.items() if k not in ("windows", "details", "info", "tail")}
        manifest.append(source + "\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
        diagnostic = any(part in {"diagnostic", "diagnostics"} for part in Path(name).parts)
        note = " [diagnostic source: expected synthetic failures possible]" if diagnostic else ""
        penalty = 4 if diagnostic else 0
        for kind in ROLES:
            detail_rows = [r for k, r in report["details"] if k == kind]
            if detail_rows:
                blocks.append((ROLES.index(kind) + penalty, index, "EXCEPTION DETAILS " + source + note,
                               dedup(detail_rows), 2600))
            if window := report["windows"].get(kind):
                blocks.append((ROLES.index(kind) + penalty, index, "FIRST " + kind + " CONTEXT " + source + note,
                               dedup(window["rows"]), 6200))
        if report["info"]:
            blocks.append((8, index, "CONFIG / LAST SAMPLED PROGRESS " + source, dedup(report["info"]), 1600))
        if report["tail"]:
            blocks.append((9, index, "TAIL " + source, dedup(report["tail"]), 600))
    header = ("OSCAR ERROR BUNDLE v2\nUTC=" + datetime.now(timezone.utc).isoformat()
              + f"\nlogs_root={root}\nselected_sources={len(reports)} omitted_older_files={omitted}\n"
              + "Read-only excerpts; no NPU/test/restart. Missing errors do NOT mean success.\n"
              + "Different paths/timestamps may be different runs. File mtime is not the error time.\n"
              + "Repeated lines are collapsed within excerpts, not repaired into a complete traceback.\n"
              + "Terminal scrollback cannot be recovered unless supplied as a file or pasted stdin.\n"
              + ("No Worker/device or EngineCore error found; API errors alone cannot identify the root cause.\n"
                 if not (ranks["worker_or_device"] or ranks["engine"]) else "")
              + "\n".join("NOTICE: " + n for n in notices) + "\n\nSOURCES\n"
              + "\n".join(manifest))
    # Keep source provenance plus room for high-priority worker/device errors.
    header = clip(header, min(12000, max_bytes // 3))
    chunks = [header]
    used = len(header.encode())
    skipped = 0
    for _, _, title, body, limit in sorted(blocks, key=lambda b: (b[0], b[1])):
        remaining = max_bytes - used - 180
        if remaining < 400:
            skipped += 1
            continue
        section = clip("\n\n" + title + "\n" + body, min(limit, remaining))
        chunks.append(section)
        used += len(section.encode())
    chunks.append(f"\n\nEND: excerpt_sections_omitted={skipped}; size_limit_bytes={max_bytes}; original logs unchanged.\n")
    return "".join(chunks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"), help="logs root or, preferably, one run directory")
    parser.add_argument("--terminal", action="append", default=[], help="saved terminal log; '-' pastes from stdin until Ctrl-D; repeatable")
    parser.add_argument("--output", type=Path, help="new file, default error-bundle-<local-time>.txt in current directory")
    parser.add_argument("--max-kb", type=int, default=48, help="output cap in KiB, 16..256; default 48")
    parser.add_argument("--max-files", type=int, default=32, help="most recently modified log files, 1..128; default 32")
    args = parser.parse_args(argv)
    if not 16 <= args.max_kb <= 256 or not 1 <= args.max_files <= 128 or len(args.terminal) > 8 or args.terminal.count("-") > 1:
        parser.error("max-kb=16..256, max-files=1..128, at most 8 terminal inputs and one stdin")
    output = (args.output or Path("error-bundle-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".txt")).resolve()
    if output.exists():
        parser.error("output exists; choose a new filename (never overwrite)")
    root = args.logs_dir.resolve()
    paths, omitted, notices = discover(root, output, args.max_files)
    if not root.is_dir():
        notices.append(f"logs directory unavailable: {root}")
    if not args.terminal:
        notices.append("no --terminal supplied; unsaved terminal output is NOT included")
    sources = [("<pasted-terminal>", None)] if "-" in args.terminal else []
    explicit = [Path(p).resolve() for p in args.terminal if p != "-"]
    seen = set()
    for path in explicit + paths:
        if path != output and path not in seen:
            seen.add(path)
            sources.append((str(path), path))
    if not sources:
        parser.error("no readable input candidates; supply --terminal or the correct --logs-dir")
    reports = []
    failures = 0
    for name, path in sources:
        try:
            if path is None:
                print("粘贴本次服务端 terminal 日志，完成后在新的一行按 Ctrl-D：", file=sys.stderr, flush=True)
                report = scan(sys.stdin.buffer)
                if report["bytes_scanned"] == 0:
                    notices.append("pasted terminal was empty")
            else:
                # Check before open so FIFOs and /proc fd pipes cannot block
                # or consume output belonging to a running service.
                meta = path.stat()
                if not stat.S_ISREG(meta.st_mode):
                    raise ValueError("input must be a regular saved file, not a device/pipe")
                with path.open("rb") as stream:
                    report = scan(stream, size=os.fstat(stream.fileno()).st_size)
                report["mtime_utc"] = datetime.fromtimestamp(meta.st_mtime, timezone.utc).isoformat()
            reports.append((name, report))
        except (OSError, ValueError) as error:
            reports.append((name, {"read_error": str(error)}))
            print(f"READ_ERROR: {name}: {error}", file=sys.stderr, flush=True)
            if path in explicit:
                notices.append(f"terminal file NOT collected: {name}; use --terminal - to paste existing scrollback")
            failures += 1
    content = make_report(reports, root=root, omitted=omitted, notices=notices, max_bytes=args.max_kb * 1024)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    primary = any(report.get("error_line_counts", {}).get(kind, 0)
                  for _, report in reports for kind in ("worker_or_device", "engine"))
    if not primary:
        print("未找到 Worker/device 或 EngineCore 首错；运行轨迹和 API 连带报错不足以定位根因。"
              "\n若服务输出尚未保存，请用 --terminal - 粘贴当前服务终端的异常栈。", file=sys.stderr, flush=True)
    print(f"汇总文件：{output}\n大小：{len(content.encode())} 字节（上限 {args.max_kb} KiB）"
          f"\n来源：{len(reports)} 个；未选入的旧文件：{omitted}；读取失败：{failures}"
          "\n只需发送这个汇总文件；截断/遗漏均在文件内标注。", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
