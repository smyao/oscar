"""Archive #94/#95/#125/#159: engine death can coexist with API exit code 0.

Read the completed service log, keep the raw process code separately, and
recognize explicit fatal lifecycle messages rather than every ERROR/warning.
No request, readiness check, native patch or device operation is performed.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR_RECORD = re.compile(r"^(?:\([^)]*\)\s*)*ERROR\s+\d\d-\d\d\s+\d\d:\d\d:\d\d\s+\[[^]]+\]")
FATAL = re.compile(r"EngineCore encountered a fatal error\.|WorkerProc hit an exception\.|"
                   r"\bEngineDeadError:")
OUTPUT_THREAD = re.compile(r"^(?:\([^)]*\)\s*)*Exception in thread WorkerAsyncOutputCopy:")


def fatal_service_evidence(path: Path):
    """Bounded memory and evidence; stop at the file's initial size."""
    matches = []
    seen = set()
    total = lines = 0
    with path.open("rb") as stream:
        remaining = os.fstat(stream.fileno()).st_size
        while remaining:
            raw = stream.readline(min(65536, remaining))
            if not raw:
                break
            remaining -= len(raw)
            prefix = raw
            while not raw.endswith(b"\n") and remaining:
                raw = stream.readline(min(65536, remaining))
                if not raw:
                    break
                remaining -= len(raw)
            lines += 1
            line = ANSI.sub("", prefix.decode("utf-8", errors="replace")).strip()
            if not ((ERROR_RECORD.search(line) and FATAL.search(line)) or OUTPUT_THREAD.search(line)):
                continue
            total += 1
            if line not in seen and len(matches) < 8:
                matches.append({"line": lines, "text": line[:2000]})
                seen.add(line)
    return {"fatal_line_count": total, "excerpts": matches,
            "scope": "explicit fatal lifecycle messages; absence is not a service acceptance test"}
