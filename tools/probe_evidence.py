"""Consume bounded host scheduler evidence when native stats are disabled.

Archive #27/#94/#125/#157/#163/#176: strict run/window provenance and real
native results; never manufacture Prometheus counters or accept missing proof.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from .phase import atomic_json


def configure(environment, lifecycle, directory):
    directory = Path(directory) / f"native-evidence-{uuid.uuid4().hex}"
    directory.mkdir()
    run_id = uuid.uuid4().hex
    environment.update(OSCAR_PROBE_EVIDENCE_DIR=str(directory), OSCAR_PROBE_RUN_ID=run_id)
    lifecycle["probe_evidence"] = {"directory": str(directory), "run_id": run_id}


def stop_run(info):
    directory = Path(info["directory"])
    atomic_json(directory / "closed.json", {"run_id": info["run_id"]})
    (directory / "active.json").unlink(missing_ok=True)


class ProbeEvidence:
    def __init__(self, server):
        info = server.lifecycle["probe_evidence"]
        self.directory = Path(info["directory"])
        self.run_id = info["run_id"]
        self.window_id = None

    def start(self):
        if self.window_id is not None or (self.directory / "closed.json").exists():
            raise RuntimeError("native probe window already started")
        self.window_id = uuid.uuid4().hex
        atomic_json(self.directory / "active.json",
                    {"run_id": self.run_id, "window_id": self.window_id})
        return self

    def stop(self):
        stop_run({"directory": self.directory, "run_id": self.run_id})

    def read(self):
        result, identities = [], set()
        for path in sorted(self.directory.glob("scheduler-*.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("run_id") != self.run_id or row.get("schema") != 1:
                    raise ValueError("native evidence has foreign run or schema")
                if row.get("window_id") != self.window_id:
                    continue
                identity = (row.get("pid"), row.get("sequence"))
                if (any(type(value) is not int or value <= 0 for value in identity)
                        or identity in identities or not isinstance(row.get("request_id"), str)
                        or not row["request_id"] or row.get("event") not in {"mtp_verified", "preempted"}):
                    raise ValueError("invalid or duplicated native evidence event")
                identities.add(identity)
                result.append(row)
        return result


def evaluate_mtp_events(records):
    rows = [row for row in records if row["event"] == "mtp_verified"]
    counts = ("scheduled_draft_slots", "num_accepted_tokens", "invalid_draft_slots")
    invalid = [row for row in rows if any(type(row.get(name)) is not int for name in counts)
               or not 0 < row["scheduled_draft_slots"] <= 3
               or not 0 <= row["num_accepted_tokens"] <= row["scheduled_draft_slots"]
               or row["invalid_draft_slots"] < 0]
    return {"status": "passed" if rows and not invalid else "failed",
            "source": "native_scheduler_probe_events", "verified_steps": len(rows),
            "scheduled_draft_slots": sum(row["scheduled_draft_slots"] for row in rows) if not invalid else None,
            "accepted_tokens": sum(row["num_accepted_tokens"] for row in rows) if not invalid else None,
            "invalid_draft_slots": sum(row["invalid_draft_slots"] for row in rows) if not invalid else None,
            "reason": "missing or invalid native MTP events" if not rows or invalid else None,
            "scope": "raw scheduled slots and accepted output tokens; not Prometheus effective drafts or quality"}


def preempted_requests(records):
    result = []
    for row in records:
        if row["event"] != "preempted":
            continue
        before, after = row.get("preemptions_before"), row.get("preemptions_after")
        if type(before) is not int or before < 0 or type(after) is not int or after != before + 1:
            raise ValueError("native preemption did not complete exactly once")
        result.append(row["request_id"])
    return result
