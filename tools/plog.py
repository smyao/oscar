"""Bounded, task-owned CANN plog evidence; archive #118/#122.

Generic SoC/package messages are not a root-cause diagnosis. Preserve actual
CANN lines from this run and these PIDs only. NPU status_guard has no printf;
this collector does not invent device status values after a Trap.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


@dataclass(frozen=True)
class PlogLimits:
    files: int = 12
    entries: int = 2048
    depth: int = 4
    bytes_per_file: int = 256 * 1024
    total_bytes: int = 1024 * 1024
    lines: int = 64
    excerpt_chars: int = 32768
    line_chars: int = 2048

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in self.__dict__.values()):
            raise ValueError("plog limits must be positive integers")


_FILE_PID = re.compile(r"(?:^|[-_.])(?:plog[-_](?:pid[-_]?)?|pid[-_=]?)(\d+)(?=$|[-_.])", re.I)
_LINE_PID = re.compile(r"\b(?:pid|process_id)\s*[:=]\s*(\d+)\b", re.I)
_CANN_PID = re.compile(r"^\s*(?:\[[A-Z]+\]\s*)?[A-Za-z_][\w]*\((\d+),[^)]*\)")
_SIGNAL = re.compile(r"oscar|\bop\b|operator|optype|kernel|error|failed|fatal|exception|aicore|aivec|\bEZ\d+", re.I)
_LOG_NAME = re.compile(r"(?:plog[-_][^.]+|.*\.log(?:[._-]\d+)*)$", re.I)


def _roots(environ, home):
    values = []
    configured = environ.get("ASCEND_PROCESS_LOG_PATH")
    if configured:
        if str(configured).startswith("~/"):
            values.append(Path(home) / str(configured)[2:])
        elif not str(configured).startswith("~"):
            values.append(Path(configured))
    values.append(Path(home) / "ascend" / "log")
    roots = []
    for path in values:
        # Never turn a mistaken log-path setting into a whole-home/disk scan.
        if not path.is_absolute():
            continue
        root = path.resolve()
        if root in {Path(root.anchor), Path(home).resolve()}:
            continue
        if root not in roots:
            roots.append(root)
    return roots


def _line_owner(line):
    # A CANN header PID is the emitter; a later message-body pid= may refer
    # to an IPC peer and must never relabel another process's log as ours.
    match = _CANN_PID.search(line) or _LINE_PID.search(line)
    return None if match is None else int(match[1])


def collect_plog(*, started_at, owned_pids, environ=None, home=None, limits=None):
    """Read bounded recent CANN logs; never traverse projects or symlinks.

    PID-bearing foreign files are not opened. An aggregate log without a PID
    in its name contributes only lines with an explicit owned PID. Filename
    ownership permits PID-less continuation lines, but never foreign PID lines.
    """
    if type(started_at) not in (float, int) or not math.isfinite(started_at) or started_at <= 0:
        raise ValueError("plog start must be a finite wall-clock timestamp")
    pids = {pid for pid in owned_pids if type(pid) is int and pid > 0}
    if not pids:
        raise ValueError("plog requires at least one explicitly owned PID")
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    limits = PlogLimits() if limits is None else limits
    roots = _roots(environ, home)
    report = {"status": "no_matching_evidence", "started_at": started_at,
              "owned_pids": sorted(pids), "roots": [str(path) for path in roots],
              "files_read": 0, "bytes_read": 0, "entries_scanned": 0,
              "truncated": False, "excerpts": [], "diagnostic_errors": [],
              "scope": "recent task-owned CANN log excerpts; not a device-completion or root-cause certificate"}
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        pending = [(root, 0)]
        while pending and report["entries_scanned"] < limits.entries:
            directory, depth = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if report["entries_scanned"] >= limits.entries:
                            report["truncated"] = True
                            break
                        report["entries_scanned"] += 1
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if depth < limits.depth:
                                pending.append((Path(entry.path), depth + 1))
                            else:
                                report["truncated"] = True
                            continue
                        if not _LOG_NAME.fullmatch(entry.name) or not entry.is_file(follow_symlinks=False):
                            continue
                        info = entry.stat(follow_symlinks=False)
                        if info.st_mtime < started_at:
                            continue
                        match = _FILE_PID.search(entry.name)
                        pid = None if match is None else int(match[1])
                        if pid is not None and pid not in pids:
                            continue
                        candidates.append((pid is not None, info.st_mtime, Path(entry.path), pid))
            except OSError as error:
                if len(report["diagnostic_errors"]) < 4:
                    report["diagnostic_errors"].append(f"scan {directory}: {type(error).__name__}")
        if pending:
            report["truncated"] = True
    scored = []
    for _, mtime, path, file_pid in sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True):
        if report["files_read"] >= limits.files or report["bytes_read"] >= limits.total_bytes:
            report["truncated"] = True
            break
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_mtime < started_at:
                    continue
                allowance = min(limits.bytes_per_file, limits.total_bytes - report["bytes_read"])
                # Keep both the first cause and the final failure tail, without
                # reading an arbitrarily large log or inventing line numbers.
                ranges = [(0, min(info.st_size, allowance))]
                if info.st_size > allowance:
                    half = allowance // 2
                    ranges = [(0, half), (info.st_size - (allowance - half), allowance - half)]
                    report["truncated"] = True
                report["files_read"] += 1
                for offset, count in ranges:
                    stream.seek(offset)
                    data = stream.read(count)
                    report["bytes_read"] += len(data)
                    for raw_line in data.splitlines(keepends=True):
                        line_offset = offset
                        offset += len(raw_line)
                        text = raw_line.decode("utf-8", errors="replace").strip()
                        owner = _line_owner(text)
                        if owner is not None and owner not in pids:
                            continue
                        if owner is None and file_pid is None:
                            continue
                        if not _SIGNAL.search(text):
                            continue
                        text = text[:limits.line_chars]
                        failed = re.search("error|failed|fatal|exception|\\bEZ\\d+", text, re.I) is not None
                        named = re.search("oscar", text, re.I) is not None
                        kernel = re.search("kernel|optype|operator", text, re.I) is not None
                        score = (0 if named else 1 if kernel else 2) if failed else (3 if named else 4 if kernel else 5)
                        scored.append((score, -mtime, line_offset, {
                            "path": str(path), "pid": file_pid if owner is None else owner,
                            "ownership": "filename" if owner is None else "line_pid",
                            "byte_offset": line_offset, "text": text}))
        except OSError as error:
            if len(report["diagnostic_errors"]) < 4:
                report["diagnostic_errors"].append(f"read {path}: {type(error).__name__}")
    chars = 0
    for _, _, _, entry in sorted(scored, key=lambda item: item[:3]):
        remaining = limits.excerpt_chars - chars
        if len(report["excerpts"]) >= limits.lines or remaining <= 0:
            report["truncated"] = True
            break
        entry["text"] = entry["text"][:remaining]
        chars += len(entry["text"])
        report["excerpts"].append(entry)
    if report["excerpts"]:
        report["status"] = "evidence_collected"
    return report


def attach_plog(report, *, started_at, owned_pids, environ=None, home=None, limits=None):
    """Optional diagnostics must never replace the probe's original failure."""
    try:
        evidence = collect_plog(started_at=started_at, owned_pids=owned_pids,
                                environ=environ, home=home, limits=limits)
        report["cann_plog"] = evidence
        for entry in evidence["excerpts"]:
            print(f"[oscar-plog] {entry['path']} byte={entry['byte_offset']} pid={entry['pid']}: {entry['text']}", file=sys.stderr)
    except BaseException as error:
        report["cann_plog"] = {"status": "diagnostic_failed", "error_type": type(error).__name__,
                                "error": str(error)[:1024], "original_failure_preserved": True}


class OwnedProcessGroup:
    """Record IDs from an explicitly start_new_session-owned service group.

    ps is used only for PID/PGID metadata, never command lines or environments.
    Keep observed members so logs remain attributable after a worker exits.
    """
    def __init__(self, leader):
        if type(leader) is not int or leader <= 0:
            raise ValueError("owned group leader must be a positive PID")
        self.leader, self.pids = leader, {leader}
        self.next_poll = 0.0
        self.error = None

    def refresh(self, *, force=False):
        if not force and time.monotonic() < self.next_poll:
            return
        self.next_poll = time.monotonic() + 1.0
        try:
            result = subprocess.run(["ps", "-axo", "pid=,pgid="], capture_output=True,
                                    text=True, timeout=1.0, check=True)
            for line in result.stdout[:1024 * 1024].splitlines():
                fields = line.split()
                if len(fields) == 2 and all(value.isdecimal() for value in fields):
                    pid, group = map(int, fields)
                    if group == self.leader:
                        self.pids.add(pid)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"[:1024]
