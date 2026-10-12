"""Parse sampled vLLM service logs; archive #130/#131/#137/#152/#155.

The Engine rates are ten-second samples, not request-level completions.  Prompt
throughput counts local-compute prompt tokens when first output is reported,
not the tokens computed within that ten-second interval.  The first sample has
an unknown left boundary; HTTP 200 can precede stream finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median


ENGINE = re.compile(
    r"INFO (?P<time>\d\d-\d\d \d\d:\d\d:\d\d).*?Engine 000: "
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[\d.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[\d.]+)%, "
    r"Prefix cache hit rate: (?P<prefix>[\d.]+)%"
)
SPEC = re.compile(
    r"INFO (?P<time>\d\d-\d\d \d\d:\d\d:\d\d).*?SpecDecoding metrics: "
    r"Mean acceptance length: (?P<length>[\d.]+), "
    r"Accepted throughput: (?P<accepted_rate>[\d.]+) tokens/s, "
    r"Drafted throughput: (?P<drafted_rate>[\d.]+) tokens/s, "
    r"Accepted: (?P<accepted>\d+) tokens, Drafted: (?P<drafted>\d+) tokens, "
    r".*?Avg Draft acceptance rate: (?P<acceptance>[\d.]+)%"
)


def _second(stamp: str) -> int:
    hh, mm, ss = map(int, stamp[-8:].split(":"))
    return hh * 3600 + mm * 60 + ss


def _window(samples: list[dict]) -> dict:
    """Nominal 10s integration; final idle sample is included but contributes zero."""
    if not samples:
        return {"samples": 0}
    accepted = sum(x["spec"]["accepted"] for x in samples if x.get("spec"))
    drafted = sum(x["spec"]["drafted"] for x in samples if x.get("spec"))
    generation = sum(x["generation_tps"] * 10 for x in samples)
    return {
        "samples": len(samples),
        "nominal_sample_seconds": len(samples) * 10,
        "prompt_tokens_rate_integral_approx": round(sum(x["prompt_tps"] * 10 for x in samples), 1),
        "generation_tokens_rate_integral_approx": round(generation, 1),
        "accepted_tokens_spec_sum": accepted,
        "drafted_tokens_spec_sum": drafted,
        "acceptance_weighted": round(accepted / drafted, 6) if drafted else None,
        "mtp_verification_rounds_inferred": drafted / 3,
        "generation_from_mtp_counts": accepted + drafted / 3,
        "generation_rate_minus_mtp_count": round(generation - (accepted + drafted / 3), 1),
        "running_sample_mean": round(sum(x["running"] for x in samples) / len(samples), 3),
        "waiting_sample_mean": round(sum(x["waiting"] for x in samples) / len(samples), 3),
        "running_sample_max": max(x["running"] for x in samples),
        "waiting_sample_max": max(x["waiting"] for x in samples),
        "kv_usage_sample_max_percent": max(x["kv_percent"] for x in samples),
        "generation_tps_sample_median": median(x["generation_tps"] for x in samples),
    }


def analyze(path: Path, source: str | None = None) -> dict:
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    samples: list[dict] = []
    for line_number, line in enumerate(lines, 1):
        matched = ENGINE.search(line)
        if matched:
            sample = {
                "line": line_number,
                "time": matched["time"],
                "prompt_tps": float(matched["prompt"]),
                "generation_tps": float(matched["generation"]),
                "running": int(matched["running"]),
                "waiting": int(matched["waiting"]),
                "kv_percent": float(matched["kv"]),
                "prefix_hit_percent": float(matched["prefix"]),
            }
            if samples:
                sample["seconds_since_previous"] = _second(sample["time"]) - _second(samples[-1]["time"])
                if sample["seconds_since_previous"] <= 0:
                    raise ValueError("nonmonotonic Engine samples")
            samples.append(sample)
            continue
        matched = SPEC.search(line)
        if matched:
            if not samples or "spec" in samples[-1]:
                raise ValueError(f"unpaired SpecDecoding sample at line {line_number}")
            if abs(_second(matched["time"]) - _second(samples[-1]["time"])) > 1:
                raise ValueError(f"SpecDecoding sample too far from Engine at line {line_number}")
            samples[-1]["spec"] = {
                "line": line_number,
                "time": matched["time"],
                "mean_acceptance_length": float(matched["length"]),
                "accepted_rate": float(matched["accepted_rate"]),
                "drafted_rate": float(matched["drafted_rate"]),
                "accepted": int(matched["accepted"]),
                "drafted": int(matched["drafted"]),
                "printed_acceptance_percent": float(matched["acceptance"]),
            }
    if not samples:
        raise ValueError("no Engine samples")
    if any("spec" not in sample for sample in samples[:-1]):
        raise ValueError("missing SpecDecoding sample before final Engine sample")
    active = [x for x in samples if x["prompt_tps"] > 0 or x["generation_tps"] > 0]
    first_queued = next((i for i, x in enumerate(samples) if x["waiting"] > 0), None)
    first_drained = next((i for i in range(first_queued + 1, len(samples)) if samples[i]["waiting"] == 0), None) if first_queued is not None else None
    last_prompt = max(i for i, x in enumerate(samples) if x["prompt_tps"] > 0)
    last_active = max(i for i, x in enumerate(samples) if x in active)
    first_full32 = next((x for x in samples if x["running"] == 32 and x["waiting"] == 0), None)
    spans = {
        "first_engine": {"time": samples[0]["time"], "line": samples[0]["line"]},
        "last_engine": {"time": samples[-1]["time"], "line": samples[-1]["line"]},
        "sample_timestamp_span_seconds": _second(samples[-1]["time"]) - _second(samples[0]["time"]),
        "first_queued": {"time": samples[first_queued]["time"], "line": samples[first_queued]["line"]} if first_queued is not None else None,
        "first_waiting_drained": {"time": samples[first_drained]["time"], "line": samples[first_drained]["line"]} if first_drained is not None else None,
        "first_full32_waiting0": {"time": first_full32["time"], "line": first_full32["line"]} if first_full32 else None,
        "last_prompt_positive": {"time": samples[last_prompt]["time"], "line": samples[last_prompt]["line"]},
        "last_active": {"time": samples[last_active]["time"], "line": samples[last_active]["line"]},
    }
    groups = {
        "all": samples,
        "active": active,
        "prompt_visible": [x for x in samples if x["prompt_tps"] > 0],
        "prompt_off_generation": [x for x in samples if x["prompt_tps"] == 0 and x["generation_tps"] > 0],
        "queue_pending": [x for x in samples if x["waiting"] > 0],
        "full32_waiting0": [x for x in samples if x["running"] == 32 and x["waiting"] == 0],
        "full32_prompt_visible": [x for x in samples if x["running"] == 32 and x["waiting"] == 0 and x["prompt_tps"] > 0],
        "full32_prompt_off": [x for x in samples if x["running"] == 32 and x["waiting"] == 0 and x["prompt_tps"] == 0 and x["generation_tps"] > 0],
        "initial_single_request": samples[:first_queued] if first_queued is not None else [],
        "admission_until_waiting_drains": samples[first_queued:first_drained] if first_queued is not None and first_drained is not None else [],
        "after_last_prompt_until_last_active": samples[last_prompt + 1:last_active + 1],
    }
    return {
        "source": source or str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_lines": len(lines),
        "http_200_lines": sum('HTTP/1.1" 200 OK' in line for line in lines),
        "measurement_boundary": "sampled Engine/Spec windows; first interval unknown; HTTP 200 is not stream completion",
        "prompt_accounting": "vLLM loggers.py:145 sums IterationStats.prompt_token_stats.computed; PrefillStats is emitted with first EngineCoreOutput, so a positive prompt rate is delayed accounting and a zero rate does not prove no prefill execution",
        "spans": spans,
        "timestamp_step_seconds": sorted({x.get("seconds_since_previous") for x in samples[1:]}),
        "groups": {name: _window(rows) for name, rows in groups.items()},
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.source, args.source_label)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
