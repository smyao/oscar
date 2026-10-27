"""Archive #84/#86/#94-97/#123/#165/#166: native OptR trace subprocess.

Fresh spawned native workers collect complete request histories and sparse
post-RoPE Q / sigmoid gate / pre-gate attention teachers. Prompt packing is
calibration-only and is not a GPQA or long-context quality evaluation. No NPU
sampling is substituted by CPU execution. The later service argv is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from tools.calibrate_model_rotations import calibration_process_env, load_messages, prompt_ids_sha256
from tools.calibration_data import calibration_prompt_source


def build_request_plan(tokenizer, messages, policy):
    """Partition complete source records before packing any sequence/chunk."""
    unique = {}
    for message in messages:
        encoded = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        identity = hashlib.sha256(encoded).hexdigest()
        unique.setdefault(identity, message)
    splits = ("train", "validation", "test")
    if len(unique) < len(splits):
        raise ValueError("OptR needs at least three distinct complete source records for disjoint splits")
    partitions = {split: [] for split in splits}
    for index, item in enumerate(sorted(unique.items())):
        partitions[splits[index % len(splits)]].append(item)
    plan = []
    for split in splits:
        records, stream = [], []
        for identity, message in partitions[split]:
            ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=False)
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                if len(ids) != 1:
                    raise ValueError("OptR source record rendered multiple prompts")
                ids = ids[0]
            if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)):
                raise ValueError("OptR source record must render nonempty integer token IDs")
            records.append(identity)
            stream.extend(ids)
        for index, length in enumerate(policy["capture_lengths"][split]):
            # Exactly reproduce the target bucket without allocating an
            # unbounded concatenation. Repetition is explicit provenance.
            whole, tail = divmod(length, len(stream))
            ids = stream * whole + stream[:tail]
            plan.append({"request_id": f"{split}_{length}_{index}", "split": split,
                         "length_bucket": length, "prompt_token_ids": ids,
                         "source_record_sha256": records, "source_stream_tokens": len(stream),
                         "source_repeated": length > len(stream),
                         "prompt_sha256": prompt_ids_sha256([ids])})
    return plan


def _check_rpc(rows, tp_size, predicate, stage):
    if (not isinstance(rows, list) or len(rows) != tp_size or
            sorted(row.get("rank", -1) for row in rows) != list(range(tp_size)) or
            any(not predicate(row) for row in rows)):
        raise RuntimeError(f"OptR {stage} did not complete on every native TP rank")


def collect(config_path, output_dir):
    from tools.model_config import apply_model_options, resolve_mamba_cache_dtypes
    from oscar_ascend.rotations import calibration_cache_identity
    from oscar_ascend.optr_policy import optr_policy, validate_refinement_config
    from oscar_ascend.integration.optr_capture import file_sha256
    from oscar_ascend.optr_traces import _workspace_geometry
    from tools.target_cli import target_env
    from tools.prepare_rotations import model_geometry
    from tools.phase import atomic_json

    config = apply_model_options(json.loads(Path(config_path).read_text()), require_model=True)
    validate_refinement_config(config)
    if config.get("rotation_refinement") != "optr":
        raise ValueError("native OptR capture requires rotation_refinement=optr")
    policy = optr_policy(config)
    # Must precede tokenizer/Torch/vLLM import, including a direct -m launch.
    os.environ.update(calibration_process_env(target_env(config)))
    os.environ.pop("OSCAR_PASSIVE_TIMING_CONTROL", None)
    os.environ.pop("OSCAR_TRACE_DIR", None)
    print(json.dumps({"optr_capture": "process_environment", "worker_multiproc_method": "spawn",
                      "scope": "isolated_native_calibration", "oscar_enabled": "0"}), flush=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("OptR capture requires a fresh empty output directory")
    source = calibration_prompt_source(config, output_dir)
    source_hash = file_sha256(source)
    print(json.dumps({"optr_capture": "calibration_data_ready", "source": str(source), "sha256": source_hash}), flush=True)
    messages = load_messages(source)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    plan = build_request_plan(tokenizer, messages, policy)
    max_prompt = max(row["length_bucket"] for row in plan)
    max_length = max_prompt + policy["decode_tokens"] + 1
    if max_length > config["max_model_len"]:
        raise ValueError("OptR prompt plus generated tokens exceeds the configured model context")
    chunk = config.get("max_num_batched_tokens", 16384)
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("OptR native token budget must be a positive integer")
    layers, dim, fingerprint = model_geometry(Path(config["model"]))
    state_dtypes = resolve_mamba_cache_dtypes(config, require_model=True)
    profile = calibration_cache_identity(config, fingerprint)
    raw = json.loads((Path(config["model"]) / "config.json").read_text())
    model = raw.get("text_config", raw)
    spec = {"layer_ids": [int(name.split(".")[2]) for name in layers], "head_dim": dim,
            "tp_size": config["tensor_parallel_size"], "global_q_heads": model["num_attention_heads"],
            "global_kv_heads": model["num_key_value_heads"], "model_fingerprint": fingerprint,
            "output_dir": str(output_dir), "samples_per_frame": policy["query_samples_per_frame"],
            "decode_tokens": policy["decode_tokens"], "max_length": max_length,
            "request_ids": [row["request_id"] for row in plan],
            "prompt_sha256": prompt_ids_sha256([row["prompt_token_ids"] for row in plan])}
    spec["workspace_geometry"] = _workspace_geometry(config)
    if not model.get("attn_output_gate", True) or spec["global_kv_heads"] != spec["tp_size"]:
        raise ValueError("OptR capture requires gated FULL attention with one disjoint KV head per rank")
    print(json.dumps({"optr_capture": "input_prepared", "requests": len(plan),
                      "capture_lengths": policy["capture_lengths"], "prompt_sha256": spec["prompt_sha256"],
                      "source_plan": policy["source_plan"], "performance": "not_measured"}), flush=True)
    import torch
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    if threads <= 0:
        raise ValueError("OMP_NUM_THREADS must be positive")
    torch.set_num_threads(threads)
    from vllm import LLM, SamplingParams
    options = dict(model=config["model"], tensor_parallel_size=spec["tp_size"], dtype="bfloat16",
                   quantization=None, trust_remote_code=True, enforce_eager=True,
                   enable_prefix_caching=False, enable_chunked_prefill=True,
                   max_model_len=max_length, max_num_batched_tokens=chunk, max_num_seqs=1,
                   gpu_memory_utilization=config["gpu_memory_utilization"], seed=0,
                   hf_overrides=config.get("hf_overrides"), additional_config=config.get("additional_config"),
                   worker_extension_cls="oscar_ascend.integration.optr_capture.OptRCaptureWorker", **state_dtypes)
    # Constructor warmup completes before any hook is installed. There is no
    # speculative_config: MTP identity and all normal service options survive.
    llm = LLM(**options)
    timeout = config.get("phase_timeout_seconds", 1800)
    try:
        started = llm.collective_rpc("oscar_optr_capture_start", timeout=timeout, args=(spec,))
        _check_rpc(started, spec["tp_size"], lambda r: r.get("layers") == sorted(spec["layer_ids"])
                   and r.get("state") == "armed_after_warmup", "arming")
        params = SamplingParams(temperature=0, max_tokens=policy["decode_tokens"] + 1, ignore_eos=True)
        for index, request in enumerate(plan):
            begun = llm.collective_rpc("oscar_optr_capture_begin", timeout=timeout, args=(request,))
            _check_rpc(begun, spec["tp_size"], lambda r: r.get("request_id") == request["request_id"]
                       and r.get("status") == "request_armed", "request begin")
            print(json.dumps({"optr_capture": "native_request_started", "request": request["request_id"],
                              "index": index + 1, "total": len(plan)}), flush=True)
            outputs = llm.generate([{"prompt_token_ids": request["prompt_token_ids"]}], params, use_tqdm=False)
            if (len(outputs) != 1 or not outputs[0].finished or len(outputs[0].outputs) != 1
                    or len(outputs[0].outputs[0].token_ids) != policy["decode_tokens"] + 1
                    or outputs[0].prompt_token_ids != request["prompt_token_ids"]):
                raise RuntimeError("OptR native request did not finish its complete prefill/decode trajectory")
            result = {"request_id": request["request_id"], "output_token_ids": list(outputs[0].outputs[0].token_ids)}
            request["generated_output_token_ids"] = result["output_token_ids"]
            saved = llm.collective_rpc("oscar_optr_capture_end", timeout=timeout, args=(result,))
            _check_rpc(saved, spec["tp_size"], lambda r: r.get("request_id") == request["request_id"]
                       and r.get("status") == "request_saved"
                       and r.get("tokens") == request["length_bucket"] + policy["decode_tokens"], "request save")
            print(json.dumps({"optr_capture": "native_request_saved", "request": request["request_id"],
                              "index": index + 1, "total": len(plan)}), flush=True)
        finished = llm.collective_rpc("oscar_optr_capture_finish", timeout=timeout)
        _check_rpc(finished, spec["tp_size"], lambda r: r.get("layers") == sorted(spec["layer_ids"])
                   and r.get("status") == "complete", "finish")
    finally:
        primary = sys.exc_info()[0] is not None
        errors = []
        try:
            llm.collective_rpc("oscar_optr_capture_abort", timeout=min(timeout, 30))
        except Exception as error:
            errors.append("restore: " + str(error))
        try:
            llm.llm_engine.engine_core.shutdown(timeout=30)
        except Exception as error:
            errors.append("shutdown: " + str(error))
        if errors:
            message = "native OptR cleanup failed: " + "; ".join(errors)
            if primary:
                print(message, file=sys.stderr, flush=True)
            else:
                raise RuntimeError(message)
    rank_paths, hashes = [], {}
    for row in sorted(finished, key=lambda row: row["rank"]):
        path = Path(row["path"]).resolve()
        if path != output_dir / f"rank{row['rank']}-manifest.json" or not path.is_file():
            raise RuntimeError("OptR rank manifest path differs or is missing")
        digest = file_sha256(path)
        if digest != row["sha256"]:
            raise RuntimeError("OptR rank manifest changed after native worker publication")
        rank_paths.append(str(path))
        hashes[str(path)] = digest
    report = {**spec, "schema_version": 1, "status": "native_optr_capture_complete",
              "rank_paths": rank_paths, "rank_sha256": hashes, "calibration_profile": profile,
              "optr_policy": policy, "prompt_source": str(source), "prompt_source_sha256": source_hash,
              "request_plan": [{k: v for k, v in row.items() if k != "prompt_token_ids"} for row in plan],
              "worker_multiproc_method": "spawn", "mamba_cache_dtypes": state_dtypes,
              "weights_quantization": None, "performance": "not_measured",
              "quality": "calibration_only_not_benchmark"}
    atomic_json(output_dir / "capture-manifest.json", report)
    print(json.dumps({"optr_capture": report["status"], "requests": len(plan), "ranks": spec["tp_size"],
                      "layers": len(layers)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.config, args.output_dir)


if __name__ == "__main__":
    main()
