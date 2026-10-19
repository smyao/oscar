"""Explicit C16 paired diagnosis on real service inputs; archive #133/#145/#164.

Only the one-key breakdown experiment sets these environment variables. The
production operator publishes the outputs consumed by inference. A separate
diagnostic clone writes separate outputs, is compared, and never publishes KV.
No events, allocations, clocks or file reads occur on an unselected call.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from .dummy_context import is_native_dummy_run

_seen = set()
_loaded = None
_serial = 0
_MAX_CALLS = 20
_OUTPUTS = {11: "partial", 12: "lse", 13: "status", 15: "cluster_stats"}


def eligible(layer, metadata, *, tokens):
    if not os.environ.get("OSCAR_C16_BREAKDOWN_DIR") or is_native_dummy_run():
        return False
    if getattr(metadata, "capture_origin", False) or getattr(metadata, "dummy_origin", False):
        return False
    return (tokens == 16384 and getattr(metadata, "max_query_len", 0) >= 16384
            and getattr(metadata, "max_seq_len", 0) >= 32768
            and (layer, bool(getattr(metadata, "is_draft", False)), getattr(metadata, "draft_index", 0)) not in _seen
            and len(_seen) < _MAX_CALLS)


def _atomic(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _load(torch):
    global _loaded
    library = Path(os.environ["OSCAR_C16_TIMER_LIBRARY"]).resolve()
    manifest_path = Path(os.environ["OSCAR_C16_TIMER_MANIFEST"]).resolve()
    if _loaded is not None:
        if _loaded[0] != str(library):
            raise RuntimeError("C16 diagnostic library changed inside one worker")
        return _loaded[1]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "built":
        raise RuntimeError("C16 diagnostic build is not complete")
    if Path(manifest["library"]).resolve() != library or _digest(library) != manifest["library_sha256"]:
        raise RuntimeError("C16 diagnostic library identity differs from its manifest")
    kernel = Path(manifest["kernel_library"]).resolve()
    if _digest(kernel) != manifest["kernel_library_sha256"]:
        raise RuntimeError("C16 diagnostic kernel identity differs from its manifest")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in manifest["source_sha256"].items():
        if _digest(root / relative) != expected:
            raise RuntimeError(f"C16 diagnostic source changed after build: {relative}")
    # Preload the exact signed dependency before the binding's DT_NEEDED lookup.
    torch.ops.load_library(str(kernel))
    torch.ops.load_library(str(library))
    _loaded = (str(library), manifest)
    return manifest


def _measure(torch, operator, args):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    stream = torch.npu.current_stream()
    start.record(stream)
    result = operator(*args)
    end.record(stream)
    end.synchronize()
    return result, float(start.elapsed_time(end))


def _compare(torch, original_args, diagnostic_args):
    checks = {}
    for index, name in _OUTPUTS.items():
        left, right = original_args[index], diagnostic_args[index]
        # Reinterpret, not numerically convert: preserve zero signs and NaN
        # payloads in the exact same kernel's partial/LSE comparison.
        checks[name] = bool(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))
    checks["baseline_status_zero"] = bool(torch.all(original_args[13] == 0).item())
    return {"status": "passed" if all(checks.values()) else "failed", "bitwise": checks}


def wrap(original, *, layer, metadata, tokens):
    if not eligible(layer, metadata, tokens=tokens):
        return original

    def paired(*args):
        global _serial
        import torch
        if torch.npu.is_current_stream_capturing():
            return original(*args)  # Graph capture remains the production path.
        if len(args) != 27 or args[25] != 0:
            raise RuntimeError("C16 breakdown requires the aligned prefill ABI")
        manifest = _load(torch)
        from ..c16_timer_schema import counter_shape, decode_ticks, SCHEMA_ID, CLOCK_HZ, CLOCK_SOURCE
        if manifest["phase_schema_id"] != SCHEMA_ID:
            raise RuntimeError("C16 phase schema differs from the compiled library")
        key = (layer, bool(getattr(metadata, "is_draft", False)), getattr(metadata, "draft_index", 0))
        _seen.add(key)
        _serial += 1
        rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else int(os.environ.get("RANK", 0))
        directory = Path(os.environ["OSCAR_C16_BREAKDOWN_DIR"]).resolve()
        path = directory / f"call-rank{rank}-{_serial:04d}.json"
        cores = int(args[26])
        row = {"schema_version": 1, "phase_schema_id": SCHEMA_ID, "rank": rank,
               "call_index": _serial, "layer": layer, "is_draft": key[1], "draft_index": key[2],
               "status": "starting", "instrumentation": "local_completion_fences",
               "baseline_operator": "attention_cv_aligned_cluster16_out",
               "diagnostic_operator": "oscar_c16_timer.attention_out", "build": manifest,
               "clock_hz": CLOCK_HZ, "clock_source": CLOCK_SOURCE,
               "geometry": {"tokens": int(args[0].shape[0]), "hq": int(args[0].shape[1]),
                    "hk": int(args[2].shape[1]), "dim": int(args[0].shape[2]), "splits": int(args[23]),
                    "requests": getattr(metadata, "num_reqs", None), "cores": cores,
                    "max_seq_len": getattr(metadata, "max_seq_len", None),
                    "max_query_len": getattr(metadata, "max_query_len", None), "dequant_mode": int(args[25])},
               "inference_uses": "original_production_outputs", "wall_start_unix": time.time()}
        _atomic(path, row)
        try:
            diagnostic = list(args)
            for index in _OUTPUTS:
                diagnostic[index] = torch.empty_like(args[index])
            ticks = torch.full(counter_shape(cores), -1, dtype=torch.int64, device=args[0].device)
            # Inputs and workspace are stable. Only scratch is reused, after
            # baseline completion; outputs remain separate for the final merge.
            result, row["baseline_ms"] = _measure(torch, original, args)
            _, row["diagnostic_ms"] = _measure(torch, torch.ops.oscar_c16_timer.attention_out,
                                               (*diagnostic, ticks))
            row["validation"] = _compare(torch, args, diagnostic)
            raw_ticks = ticks.cpu().tolist()  # Tiny counters only; no full KV/partial D2H.
            tick_path = path.with_name(path.stem + "-ticks.json")
            _atomic(tick_path, raw_ticks)
            decoded = decode_ticks(raw_ticks, cores)
            row.update(actors=decoded["actors"], counters_valid=decoded["valid"],
                       counter_issues=decoded["issues"], counter_path=str(tick_path))
            max_actor_ms = max((actor["total_cycles"] * 1000.0 / CLOCK_HZ for actor in decoded["actors"]), default=0.0)
            # Actor clocks exclude the final counter flush, whereas NPU Events
            # surround the whole op. A material overrun rejects the timebase;
            # do not silently convert arbitrary cycle units to milliseconds.
            row["timebase_check"] = {"status": "passed" if 0 < max_actor_ms <= row["diagnostic_ms"] * 1.05 + .1 else "failed",
                                      "max_actor_ms": max_actor_ms, "event_ms": row["diagnostic_ms"],
                                      "scope": "upper-bound consistency, not a frequency calibration"}
            row["diagnostic_over_baseline"] = row["diagnostic_ms"] / row["baseline_ms"] if row["baseline_ms"] > 0 else None
            row["extra_output_bytes"] = sum(t.numel() * t.element_size() for i, t in enumerate(diagnostic) if i in _OUTPUTS)
            row["status"] = "complete" if (row["validation"]["status"] == "passed" and decoded["valid"]
                                             and row["timebase_check"]["status"] == "passed") else "failed"
            if row["status"] != "complete":
                raise RuntimeError("C16 paired output/counter validation failed; see saved call record")
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["wall_finish_unix"] = time.time()
            _atomic(path, row)

    return paired
