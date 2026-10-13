"""Short synthetic APC service probe; archive #37-49/#94/#95/#125/#130/#131/#157.

Native evidence: completion/protocol.py (token IDs, cache_salt, SSE usage).
Isolated HTTP children give every request a real wall deadline. No deployment,
cache reset, native-source changes, tolerance changes, or AISBench invocation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.request import Request, urlopen
from uuid import uuid4


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_sse(response, expected_prompt, expected_output, raw_log=None):
    started = time.monotonic()
    first = None
    chunks, tokens, texts = [], [], []
    usage = finish = None
    done = False
    for raw in response:
        if raw_log is not None:
            raw_log.write(raw)
            raw_log.flush()
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        item = json.loads(payload)
        if "error" in item:
            raise RuntimeError(f"service stream error: {item['error']}")
        if item.get("usage") is not None:
            usage = item["usage"]
        for choice in item.get("choices", []):
            ids = choice.get("token_ids") or []
            text = choice.get("text") or ""
            if text or ids:
                now = time.monotonic() - started
                first = now if first is None else first
                chunks.append(now)
                texts.append(text)
                tokens.extend(ids)
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if not done or finish is None or usage is None or first is None:
        raise ValueError("incomplete SSE: need content, finish_reason, usage and [DONE]")
    if usage.get("prompt_tokens") != expected_prompt or usage.get("completion_tokens") != expected_output:
        raise ValueError(f"server token usage mismatch: {usage}")
    if tokens and len(tokens) != expected_output:
        raise ValueError("returned token IDs disagree with completion token usage")
    details = usage.get("prompt_tokens_details") or {}
    return {"usage": usage, "cached_tokens": details.get("cached_tokens"),
            "cached_tokens_scope": "missing_is_unknown_not_zero",
            "finish_reason": finish, "sse_done": done, "content_chunks": len(chunks),
            "first_content_after_headers_s": first, "chunk_times_after_headers_s": chunks,
            "token_ids": tokens or None,
            "text_sha256": hashlib.sha256("".join(texts).encode()).hexdigest()}


def child(request_path, result_path):
    started = time.monotonic()
    try:
        task = json.loads(request_path.read_text())
        payload = task.get("payload")
        request = Request(task["url"], data=json.dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=task["timeout"]) as response:
            header_s = time.monotonic() - started
            if task.get("stream"):
                with result_path.with_suffix(".sse.txt").open("wb") as raw_log:
                    result = read_sse(response, len(payload["prompt"]), payload["max_tokens"], raw_log)
                result["ttft_s"] = header_s + result["first_content_after_headers_s"]
            elif task.get("text"):
                result = {"body": response.read().decode("utf-8")}
            else:
                result = {"body": json.load(response)}
        result.update(status="completed", elapsed_s=time.monotonic() - started)
        save(result_path, result)
        return 0
    except Exception as error:
        traceback.print_exc()
        save(result_path, {"status": "failed", "error": f"{type(error).__name__}: {error}",
                           "elapsed_s": time.monotonic() - started})
        return 1


def isolated_http(directory, name, task):
    request, result = directory / f"{name}.request.json", directory / f"{name}.response.json"
    save(request, task)
    command = [sys.executable, "-u", "-m", "tools.prefix_cache_probe", "_request", str(request), str(result)]
    try:
        process = subprocess.run(command, timeout=task["timeout"], check=False)
    except subprocess.TimeoutExpired:
        save(result, {"status": "failed", "error": "absolute HTTP child deadline exceeded",
                      "deadline_s": task["timeout"]})
        raise RuntimeError(f"{name}: absolute deadline {task['timeout']}s exceeded") from None
    if process.returncode != 0:
        raise RuntimeError(f"{name}: HTTP child rc={process.returncode}; result={result}")
    return json.loads(result.read_text())


def cases(lengths):
    if not lengths or any(n < 2 for n in lengths):
        raise ValueError("lengths must contain integers >= 2")
    return [{"length": n, "sequence": ["cold_seed", "warm_repeat", "fork_left", "fork_right",
                                      "fork_left_cold", "fork_right_cold"]}
            for n in lengths]


def validate_hits(record, full_tokens=2304, hash_tokens=768):
    """Strict positive-hit contract for this pinned OSCAR MTP geometry.

    Zero-hit usage is omitted by native completion serving. Missing positive
    evidence remains unknown and fails; HTTP success alone never passes APC.
    """
    from oscar_ascend.integration.prefix_cache import prefix_hit_bound
    n = record["input_tokens"]
    expected = prefix_hit_bound(n, full_tokens, hash_tokens, guarded=True, mtp=True)
    warm = record["requests"]["warm_repeat"].get("cached_tokens")
    if expected and (type(warm) is not int or warm != expected):
        raise ValueError(f"APC warm hit mismatch: prompt={n} expected={expected}, actual={warm}")
    for label, response in record["requests"].items():
        bound = prefix_hit_bound(n + int(label.startswith("fork")), full_tokens, hash_tokens,
                                 guarded=True, mtp=True)
        hit = response.get("cached_tokens")
        if hit is not None and (type(hit) is not int or hit < 0 or hit % full_tokens or hit > bound):
            raise ValueError(f"APC unsafe/non-page hit: prompt={n} arm={label} hit={hit} bound={bound}")
    return {"status": "passed" if expected else "no_reusable_page",
            "expected_warm_tokens": expected, "actual_warm_tokens": warm,
            "full_page_tokens": full_tokens, "hash_tokens": hash_tokens}


def run(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "short_synthetic_APC_completion_probe",
              "base_url": args.base_url, "model": args.model, "cases": [],
              "model_quality": "not_established", "performance_acceptance": "not_established",
              "checkpoint_visibility": "server_metrics_or_usage_required",
              "config_assumption": "caller verifies effective APC/MTP/native-budget configuration"}
    save(directory / "summary.json", report)
    base = args.base_url.rstrip("/")
    deadline = time.monotonic() + getattr(args, "total_timeout", 600)
    def timeout():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("prefix probe total deadline exceeded")
        return min(args.timeout, remaining)
    try:
        encoded = isolated_http(directory, "tokenize", {"url": base + "/tokenize",
            "timeout": min(args.timeout, 15), "payload": {"model": args.model,
            "prompt": "A short synthetic cache test. Apples rivers books mountains. ",
            "add_special_tokens": False}})["body"]["tokens"]
        if len(set(encoded)) < 2:
            raise ValueError("tokenizer must return at least two distinct valid token IDs")
        for case_index, plan in enumerate(cases(args.lengths)):
            n = plan["length"]
            prompt = (encoded * ((n + len(encoded) - 1) // len(encoded)))[:n]
            salt = uuid4().hex
            record = {"input_tokens": n, "cache_salt": salt, "requests": {}}
            report["cases"].append(record)
            before = isolated_http(directory, f"{case_index:02d}-metrics-before", {
                "url": base + "/metrics", "timeout": min(args.timeout, 15), "text": True})
            (directory / f"{case_index:02d}-metrics-before.txt").write_text(before["body"])

            def call(label, ids, cache_salt=salt):
                result = isolated_http(directory, f"{case_index:02d}-{label}", {
                    "url": base + "/v1/completions", "timeout": timeout(), "stream": True,
                    "payload": {"model": args.model, "prompt": ids, "max_tokens": args.output_tokens,
                                "temperature": 0, "seed": 7, "ignore_eos": True,
                                "stream": True, "stream_options": {"include_usage": True},
                                "return_token_ids": True, "cache_salt": cache_salt}})
                print(json.dumps({"t": "prefix-probe", "case": n, "arm": label,
                                  **{k: result[k] for k in ("status", "ttft_s", "elapsed_s", "cached_tokens")}},
                                 ensure_ascii=False), flush=True)
                return result

            for label in ("cold_seed", "warm_repeat"):
                record["requests"][label] = call(label, prompt)
                save(directory / "summary.json", report)
            # Two suffixes fork after the exact same token prefix. No reset or
            # global state mutation; separate lengths have unique cache salts.
            tail_ids = list(dict.fromkeys(encoded))[:2]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [(label, executor.submit(call, label, prompt + [tail]))
                           for label, tail in zip(("fork_left", "fork_right"), tail_ids)]
                for label, future in futures:
                    record["requests"][label] = future.result()
            for label, tail in zip(("fork_left_cold", "fork_right_cold"), tail_ids):
                record["requests"][label] = call(label, prompt + [tail], uuid4().hex)
                warm_fork = record["requests"][label.removesuffix("_cold")]
                record[label + "_output_identical"] = (
                    warm_fork["token_ids"] == record["requests"][label]["token_ids"]
                    if warm_fork["token_ids"] is not None and record["requests"][label]["token_ids"] is not None
                    else warm_fork["text_sha256"] == record["requests"][label]["text_sha256"])
            after = isolated_http(directory, f"{case_index:02d}-metrics-after", {
                "url": base + "/metrics", "timeout": min(args.timeout, 15), "text": True})
            (directory / f"{case_index:02d}-metrics-after.txt").write_text(after["body"])
            cold, warm = (record["requests"][name] for name in ("cold_seed", "warm_repeat"))
            record["warm_over_cold_ttft"] = warm["ttft_s"] / cold["ttft_s"]
            record["cold_warm_output_identical"] = (
                cold["token_ids"] == warm["token_ids"] if cold["token_ids"] is not None and warm["token_ids"] is not None
                else cold["text_sha256"] == warm["text_sha256"])
            record["output_comparison_scope"] = "sensitivity_signal_not_frozen_quality_gate"
            if getattr(args, "require_hit", False):
                record["cache_gate"] = validate_hits(record)
            save(directory / "summary.json", report)
        if getattr(args, "require_hit", False):
            if not any(r["cache_gate"]["status"] == "passed" for r in report["cases"]):
                raise ValueError("APC gate requires at least one positive page hit")
            report["cache_gate"] = "passed"
        report["status"] = "completed"
        report["request_count"] = len(report["cases"]) * 6
        save(directory / "summary.json", report)
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(directory / "summary.json", report)
        traceback.print_exc()
        return 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_request":
        return child(Path(argv[1]), Path(argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8989")
    parser.add_argument("--model", default="qwen3.5")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2304, 3072, 4608, 5376, 8192])
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--total-timeout", type=float, default=600)
    parser.add_argument("--require-hit", action="store_true", help="require measured warm hits for OSCAR TP4/MTP geometry 2304/768")
    parser.add_argument("--output", type=Path, default=Path("reports/prefix-service-probe"))
    parser.add_argument("--run", action="store_true", help="send the planned requests to an already running service")
    args = parser.parse_args(argv)
    if not 1 <= args.output_tokens <= 128 or not 0 < args.timeout <= 300 or not 0 < args.total_timeout <= 1800:
        parser.error("output-tokens must be 1..128 and timeout must be 0..300 seconds")
    plan = cases(args.lengths)
    if not args.run:
        print(json.dumps({"status": "plan_only", "requests": len(plan) * 6,
                          "cases": plan, "output_tokens": args.output_tokens,
                          "scope": "completion_and_APC_probe_not_performance_acceptance"}, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
