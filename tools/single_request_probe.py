"""Archive #130/#131/#157/#159: one bounded synthetic request for trap attribution.

Uses the running service's tokenizer and exact token-ID input. APC, MTP and
graphs remain configured by the service; this is not AISBench or acceptance.
Default is plan-only. No retry, model loading, cache clearing or server restart.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from uuid import uuid4

from .prefix_cache_probe import isolated_http, save


def run(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "base_url": args.base_url, "model": args.model,
              "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
              "generation_requests": 1, "synthetic": True,
              "scope": "shape-oriented diagnostic; neither original payload reproduction nor quality/performance acceptance"}
    save(directory / "summary.json", report)
    deadline = time.monotonic() + args.timeout
    def remaining():
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError("single-request probe total deadline exceeded")
        return budget
    base = args.base_url.rstrip("/")
    try:
        encoded = isolated_http(directory, "tokenize", {"url": base + "/tokenize",
            "timeout": min(15, remaining()), "payload": {"model": args.model,
            "prompt": "A short synthetic diagnostic. Apples rivers books mountains. ",
            "add_special_tokens": False}})["body"]["tokens"]
        if (not isinstance(encoded, list) or not encoded or
                any(type(token) is not int or token < 0 for token in encoded)):
            raise ValueError("tokenizer must return nonempty valid integer token IDs")
        ids = (encoded * ((args.prompt_tokens + len(encoded) - 1) // len(encoded)))[:args.prompt_tokens]
        print(f"[oscar] SINGLE_REQUEST_START prompt={len(ids)} output={args.output_tokens} "
              f"report={directory / 'summary.json'}", flush=True)
        report["response"] = isolated_http(directory, "request", {"url": base + "/v1/completions",
            "timeout": remaining(), "stream": True, "payload": {"model": args.model,
            "prompt": ids, "max_tokens": args.output_tokens, "temperature": 0,
            "seed": 7, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True,
            "cache_salt": uuid4().hex}})
        report["status"] = "completed"
        save(directory / "summary.json", report)
        print(f"[oscar] SINGLE_REQUEST_COMPLETED report={directory / 'summary.json'}", flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(directory / "summary.json", report)
        traceback.print_exc()
        print(f"[oscar] SINGLE_REQUEST_FAILED report={directory / 'summary.json'}; "
              "inspect the server.log from the same run", flush=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7878")
    parser.add_argument("--model", default="qwen3.5")
    # Shape suggested by scheduler computed=24626/output=11; original prompt
    # and previous asynchronous batch are not present in the failure dump.
    parser.add_argument("--prompt-tokens", type=int, default=24616)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("logs") / (
        "single-request-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not 2 <= args.prompt_tokens <= 65536 or not 1 <= args.output_tokens <= 128 or not 0 < args.timeout <= 300:
        parser.error("prompt-tokens=2..65536, output-tokens=1..128, timeout=0..300s")
    if not args.run:
        print(json.dumps({"status": "plan_only", "generation_requests": 1,
                          "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
                          "base_url": args.base_url, "timeout_seconds": args.timeout,
                          "output": str(args.output)}, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
