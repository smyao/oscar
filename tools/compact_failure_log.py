"""Small, read-only diagnostic export; archive #94/#95/#125/#157.

Standalone stdlib script: no project imports, device operations or log deletion.
Only scans the selected run directory; never reads request/response payloads.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import re


SIGNAL = re.compile(r"OSCAR_FIRST_ERROR|OSCAR_STATUS_DIAGNOSTICS_ACTIVE|"
                    r"Traceback|Error:|Exception:|FAILED|fatal|507035|"
                    r"status_guard|\[oscar-plog\]|timed? ?out", re.I)
FIELDS = {"status", "error", "failed_phase", "returncode", "exit_code",
          "resource_release", "cleanup_complete", "cleanup_error", "passed",
          "timed_out", "interrupted", "group", "physical_device"}


def compact(run: Path, limit: int = 6000) -> str:
    metadata, first, signals, tails, layout = [], [], [], [], []
    seen = set()

    def add(bucket, source, line, width=700):
        line = line.strip()
        identity = (source, line) if bucket is metadata else line
        if not line or identity in seen:
            return
        seen.add(identity)
        bucket.append(f"{source}: {line[:width]}" + (" [line truncated]" if len(line) > width else ""))

    def walk(value, source, depth=0):
        if depth > 7:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                label = f"{source}.{key}"
                if key in FIELDS and not isinstance(item, (dict, list)):
                    add(metadata, label, json.dumps(item, ensure_ascii=False), 250)
                elif key == "excerpts" and isinstance(item, list):
                    for entry in item[:12]:
                        if isinstance(entry, dict):
                            text = str(entry.get("text", ""))
                            if not re.search(r"\[INFO\]|\[DEBUG\]", text):
                                add(signals, str(entry.get("path", label)), text)
                elif key not in {"response", "payload", "prompt", "requests", "command", "env"}:
                    walk(item, label, depth + 1)
        elif isinstance(value, list):
            for index, item in enumerate(value[:16]):
                walk(item, f"{source}[{index}]", depth + 1)

    for name in ("exit-code.txt", "service-report.json", "diagnostic-report.json", "build/status.json"):
        path = run / name
        if not path.is_file():
            metadata.append(f"{name}: missing (stage may not have run)")
            continue
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                metadata.append(f"{name}: too large; see original")
            elif path.suffix == ".json":
                walk(json.loads(path.read_text()), name)
            else:
                add(metadata, name, path.read_text())
        except (OSError, ValueError) as error:
            metadata.append(f"{name}: cannot read: {error}")

    trace_files = sorted((run / "service").glob("trace-*/worker-*.jsonl"))
    layout.append(f"worker trace files: {len(trace_files)}")
    for path in trace_files[:12]:
        events = {}
        rows = []
        try:
            with path.open(errors="replace") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                    except ValueError:
                        rows.append(f"invalid JSON at line {number}")
                        continue
                    if not isinstance(record, dict):
                        continue
                    event = str(record.get("event"))
                    events[event] = events.get(event, 0) + 1
                    if event == "cache_layout":
                        row = {k: record.get(k) for k in (
                            "rank", "pid", "prefix_caching", "scheduler_lcm_block_size",
                            "mamba_block_size", "physical_block_tokens")}
                        value = json.dumps(row, ensure_ascii=False)
                        if value not in rows:
                            rows.append(value)
            layout.append(f"{path.name}: events={json.dumps(events, sort_keys=True)[:300]} "
                          f"layouts={'; '.join(rows[:4]) or 'NONE'}")
        except OSError as error:
            layout.append(f"{path.name}: {error}")
    paths = [run / "service/server.log", run / "terminal.log"]
    paths += sorted((run / "build").rglob("*.log"))
    # Synthetic diagnostic children intentionally Trap: label separately.
    paths += sorted((run / "diagnostic").glob("*.log"))
    for path in paths[:40]:
        if not path.is_file() or path.is_symlink():
            continue
        source = str(path.relative_to(run))
        tail = deque(maxlen=5)
        kept = 0
        try:
            with path.open(errors="replace") as stream:
                for number, line in enumerate(stream, 1):
                    # Stream from the beginning to retain the first fault.
                    if '"synthetic"' in line and "OSCAR_FIRST_ERROR" in line:
                        continue
                    tail.append((number, line))
                    if "OSCAR_FIRST_ERROR" in line and len(first) < 12:
                        add(first, f"{source}:{number}", line, 1100)
                    elif SIGNAL.search(line) and kept < 14:
                        add(signals, f"{source}:{number}", line)
                        kept += 1
            for number, line in tail:
                add(tails, f"{source}:{number}", line, 250)
        except OSError as error:
            metadata.append(f"{source}: cannot read: {error}")

    sections = [
        ("RESULT", metadata, 1800), ("FIRST DEVICE STATUS", first, 1800),
        ("WORKER LAYOUT", layout, 1800),
        ("ERROR EXCERPTS", signals, 1800), ("TAIL", tails, 400),
    ]
    output = (f"OSCAR compact diagnostic\nrun={run.resolve()}\n"
              "Excerpt only; absence of an error is NOT a pass.\n"
              "diagnostic/ contains expected synthetic Trap failures; consult its report.\n")
    footer = "\n[Bounded excerpt; original logs retained. Omitted details may be needed.]\n"
    for title, lines, budget in sections:
        if not lines:
            continue
        available = min(budget, limit - len(output) - len(footer) - len(title) - 5)
        if available < 80:
            break
        text = "\n".join(lines)
        if len(text) > available:
            text = text[:available - 16] + "\n[truncated]"
        output += f"\n[{title}]\n{text}\n"
    return (output + footer)[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()
    if not 2000 <= args.max_chars <= 20000:
        parser.error("--max-chars must be 2000..20000")
    run = args.run
    if run is None:
        marker = Path("logs/latest-npu-check.txt")
        if not marker.is_file():
            parser.error("provide the run directory; logs/latest-npu-check.txt is missing")
        run = Path(marker.read_text().strip())
    if not run.is_dir():
        parser.error(f"run directory does not exist: {run}")
    output = args.output or run / "key-errors.txt"
    if output.exists():
        parser.error(f"refusing to overwrite existing file: {output}; choose --output")
    output.write_text(compact(run, args.max_chars), encoding="utf-8")
    print(f"Written: {output} ({output.stat().st_size} bytes; <= {args.max_chars} characters)")


if __name__ == "__main__":
    main()
