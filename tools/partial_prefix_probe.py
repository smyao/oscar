"""Archive #26/#37-49/#94/#95/#125/#127/#128/#157: bounded partial APC probe.

Two isolated salts distinguish sparse checkpoint availability from a controlled
positive-hit witness. Does not reset caches, change budgets, deploy a service,
or treat HTTP completion / a zero partial hit as a coordinator diagnosis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from uuid import uuid4

from .prefix_cache_probe import isolated_http, reference_hit_bound, save

FULL_TOKENS = 2304
HASH_TOKENS = 768


def plan(seed_tokens=32768, shared_tokens=13824):
    if (type(seed_tokens) is not int or type(shared_tokens) is not int or
            not 0 < shared_tokens < seed_tokens <= 262144):
        raise ValueError("require 0 < shared-tokens < seed-tokens <= 262144")
    witness = reference_hit_bound(shared_tokens)
    if not witness:
        raise ValueError("shared prefix must contain a FULL page and its lookahead hash")
    anchor = witness + HASH_TOKENS
    return {"seed_tokens": seed_tokens, "shared_tokens": shared_tokens,
            "partial_upper_bound": witness, "anchor_tokens": anchor,
            "expected_controlled_hit": witness,
            "sequence": ["long_seed", "partial_after_long_seed", "checkpoint_anchor", "partial_after_anchor"],
            "prompt_token_total": 3 * seed_tokens + anchor,
            "interpretation": "uncontrolled zero/unknown does not establish a coordinator bug; controlled hit is required"}


def validate(report):
    expected = report["plan"]["expected_controlled_hit"]
    requests = report["requests"]
    for label, result in requests.items():
        hit = result.get("cached_tokens")
        upper = 0 if label in {"long_seed", "checkpoint_anchor"} else expected
        if hit is not None and (type(hit) is not int or hit < 0 or hit % FULL_TOKENS or hit > upper):
            raise ValueError(f"unsafe or unexpected cached_tokens: arm={label} actual={hit} bound={upper}")
    controlled = requests["partial_after_anchor"].get("cached_tokens")
    if type(controlled) is not int or controlled != expected:
        raise ValueError(f"controlled APC witness missing: expected={expected}, actual={controlled}")
    observed = requests["partial_after_long_seed"].get("cached_tokens")
    return {"controlled_hit": "passed", "observed_partial_hit": observed,
            "observed_interpretation": ("unknown" if observed is None else
                                        "no_joint_checkpoint_hit" if observed == 0 else "positive_partial_hit"),
            "coordinator_bug_established": False,
            "scope": "positive reuse witness only; not numerical quality or throughput acceptance"}


def run(args):
    contract = plan(args.seed_tokens, args.shared_tokens)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "plan": contract, "requests": {},
              "base_url": args.base_url, "model": args.model,
              "geometry_assumption": "OSCAR TP4/MTP physical FULL2304 native hash/GDN768; caller verifies runtime configuration",
              "model_quality": "not_established", "performance_acceptance": "not_established"}
    save(directory / "summary.json", report)
    deadline = time.monotonic() + args.total_timeout
    def timeout(limit):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("partial prefix probe total deadline exceeded")
        return min(limit, remaining)
    base = args.base_url.rstrip("/")
    try:
        encoded = isolated_http(directory, "tokenize", {"url": base + "/tokenize",
            "timeout": timeout(min(args.timeout, 15)), "payload": {"model": args.model,
            "prompt": "A short synthetic cache test. Apples rivers books mountains. ",
            "add_special_tokens": False}})["body"]["tokens"]
        if len(set(encoded)) < 2:
            raise ValueError("tokenizer must return at least two distinct valid IDs")
        seed = (encoded * ((args.seed_tokens + len(encoded) - 1) // len(encoded)))[:args.seed_tokens]
        fork = seed.copy()
        fork[args.shared_tokens] = next(token for token in encoded if token != seed[args.shared_tokens])
        salt_long, salt_anchor = uuid4().hex, uuid4().hex
        report["salts"] = {"long_seed_pair": salt_long, "checkpoint_pair": salt_anchor}
        for label, ids, salt in (
            ("long_seed", seed, salt_long),
            ("partial_after_long_seed", fork, salt_long),
            ("checkpoint_anchor", seed[:contract["anchor_tokens"]], salt_anchor),
            ("partial_after_anchor", fork, salt_anchor),
        ):
            result = isolated_http(directory, label, {"url": base + "/v1/completions",
                "timeout": timeout(args.timeout), "stream": True,
                "payload": {"model": args.model, "prompt": ids, "max_tokens": args.output_tokens,
                            "temperature": 0, "seed": 7, "ignore_eos": True,
                            "stream": True, "stream_options": {"include_usage": True},
                            "return_token_ids": True, "cache_salt": salt}})
            report["requests"][label] = result
            save(directory / "summary.json", report)
            print(json.dumps({"t": "partial-prefix-probe", "arm": label,
                              "prompt_tokens": len(ids), "cached_tokens": result.get("cached_tokens"),
                              "status": result["status"]}), flush=True)
        report["assessment"] = validate(report)
        report["status"] = "completed"
        save(directory / "summary.json", report)
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(directory / "summary.json", report)
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8989")
    parser.add_argument("--model", default="qwen3.5")
    parser.add_argument("--seed-tokens", type=int, default=32768)
    parser.add_argument("--shared-tokens", type=int, default=13824)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--total-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("reports/partial-prefix-probe"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.output_tokens <= 128 or not 0 < args.timeout <= 300 or not 0 < args.total_timeout <= 1800:
        parser.error("output-tokens=1..128, timeout=0..300s, total-timeout=0..1800s")
    try:
        contract = plan(args.seed_tokens, args.shared_tokens)
    except ValueError as error:
        parser.error(str(error))
    if not args.run:
        print(json.dumps({"status": "plan_only", "requests": 4, **contract}, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
