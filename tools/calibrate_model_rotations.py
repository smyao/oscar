"""Archive #23-25/#28/#84/#86/#94-97/#166: native post-RoPE QKV sampling.

Official recipe pinned to OSCAR339368e: seed0 GPQA choice shuffle, chat-template
prompt IDs, whole-prompt budget30000. Sampling uses native vLLM Ascend in an
isolated process; no OSCAR cache, performance tests, model weight edits or
production CPU fallback. The parent fits FP64 moments after this model exits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from tools.calibration_data import calibration_prompt_source

ROOT = Path(__file__).resolve().parents[1]
GPQA_FIELDS = ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
GPQA_TEMPLATE = ("Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")


def load_messages(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not set(GPQA_FIELDS) <= rows[0].keys():
            raise ValueError("calibration CSV must have the original GPQA fields")
        rng = random.Random(0)
        result = []
        for row in rows:
            choices = [row[name] for name in GPQA_FIELDS[1:]]
            shuffled = [choices[i] for i in rng.sample(range(4), 4)]
            text = GPQA_TEMPLATE.format(Question=row["Question"], **dict(zip("ABCD", shuffled)))
            result.append([{"role": "user", "content": text}])
        return result
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        messages = ([{"role": "user", "content": item}] if isinstance(item, str) else
                    item.get("messages") if isinstance(item,dict) else None)
        if not isinstance(messages, list) or not messages:
            raise ValueError("calibration JSONL requires nonempty messages")
        result.append(messages)
    if not result:
        raise ValueError("empty calibration prompts")
    return result


def render_prompt_ids(tokenizer, messages, token_budget, chunk_limit):
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError("calibration_tokens must be positive")
    if type(chunk_limit) is not int or chunk_limit<=0:
        raise ValueError('calibration prefill chunk limit must be positive')
    prompts, count = [], 0
    for message in messages:
        ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("one calibration record must render to one prompt")
            ids = ids[0]
        ids = list(map(int, ids))
        if not ids:
            continue
        if count + len(ids) > token_budget:
            break
        if len(ids) > chunk_limit:
            raise ValueError("each calibration prompt must fit one prefill chunk")
        prompts.append(ids); count += len(ids)
        if count == token_budget:
            break
    if not prompts:
        raise ValueError("no complete prompt fits calibration token budget")
    return prompts


def prompt_ids_sha256(prompts):
    # Same canonical token list serialization as upstream calibration.
    return hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest()


def calibration_process_env(base=None):
    """Use fresh interpreters at both native EngineCore and TP-worker boundaries.

    vLLM's get_mp_context() reads this env directly; set_start_method() alone
    would not override its explicit default fork. Keep this isolated from the
    later service process, as in Ascend's official offline TP example.
    """
    env = dict(os.environ if base is None else base)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["OSCAR_ENABLED"] = "0"
    env.pop("OSCAR_PASSIVE_TIMING_CONTROL", None)
    env.pop("OSCAR_TRACE_DIR", None)
    return env


def collect(config_path, output_dir):
    from tools.model_config import apply_model_options
    from tools.target_cli import target_env
    from tools.prepare_rotations import model_geometry
    from tools.phase import atomic_json
    config = apply_model_options(json.loads(Path(config_path).read_text()), require_model=True)
    # Do this before importing tokenizer/Torch/vLLM, including for direct -m
    # invocation. _capture also sets it before this interpreter is started.
    os.environ.update(calibration_process_env(target_env(config)))
    os.environ.pop("OSCAR_PASSIVE_TIMING_CONTROL", None)
    os.environ.pop("OSCAR_TRACE_DIR", None)
    print(json.dumps({'rotation_capture': 'process_environment',
        'worker_multiproc_method': os.environ['VLLM_WORKER_MULTIPROC_METHOD'],
        'scope': 'isolated_calibration_engine_and_tp_workers'}), flush=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob("rank*.pt")) or (output_dir/'capture-manifest.json').exists():
        raise ValueError("calibration capture requires a fresh directory")
    source = calibration_prompt_source(config, output_dir)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    print(json.dumps({'rotation_capture': 'calibration_data_ready',
        'source': 'explicit' if config.get('rotation_calibration_prompts') else 'bundled_official_gpqa',
        'path': str(source), 'sha256': source_sha256}), flush=True)
    messages = load_messages(source)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    chunk = int(config.get("max_num_batched_tokens", 16384))
    prompts = render_prompt_ids(tokenizer, messages, config.get("rotation_calibration_tokens", 30000), chunk)
    if max(map(len, prompts)) + 1 > min(chunk, config["max_model_len"]):
        raise ValueError("calibration prefill plus one sampled token must fit the explicit native budget")
    layers, dim, fingerprint = model_geometry(Path(config["model"]))
    raw = json.loads((Path(config["model"]) / "config.json").read_text())
    text = raw.get("text_config", raw)
    spec = {"layer_ids": [int(name.split(".")[2]) for name in layers], "head_dim": dim,
        "tp_size": config["tensor_parallel_size"], "global_q_heads": text["num_attention_heads"],
        "global_kv_heads": text["num_key_value_heads"], "token_count": sum(map(len, prompts)),
        "prompt_lengths": list(map(len,prompts)),
        "prompt_sha256": prompt_ids_sha256(prompts), "model_fingerprint": fingerprint,
        "output_dir": str(output_dir)}
    print(json.dumps({'rotation_capture':'input_prepared','prompts':len(prompts),
        'tokens':spec['token_count'],'prompt_sha256':spec['prompt_sha256']}),flush=True)
    # #84/#166: initialize this interpreter's pool even when OMP is pinned.
    # Workers must still spawn: parent initialization alone did not make the
    # two-level fork safe after tokenization/backend threads were started.
    import torch
    threads=int(os.environ.get('OMP_NUM_THREADS','1'))
    if threads<=0:raise ValueError('OMP_NUM_THREADS must be positive')
    torch.set_num_threads(threads)
    from vllm import LLM, SamplingParams
    options = dict(model=config["model"], tensor_parallel_size=config["tensor_parallel_size"],
        dtype="bfloat16", quantization=config.get("quantization"), trust_remote_code=True,
        enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False,
        max_model_len=min(config["max_model_len"], max(map(len, prompts)) + 1), max_num_batched_tokens=chunk,
        max_num_seqs=1, gpu_memory_utilization=config["gpu_memory_utilization"], seed=0,
        hf_overrides=config.get("hf_overrides"), additional_config=config.get("additional_config"),
        worker_extension_cls="oscar_ascend.integration.rotation_capture.RotationCaptureWorker",
        mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="bfloat16")
    # max_tokens=1 samples the prefill logits and schedules no decode forward.
    # The normal service's MTP/APC/graph settings are untouched by this process.
    llm = LLM(**options)
    timeout = config.get("phase_timeout_seconds", 1800)
    try:
        started = llm.collective_rpc("oscar_rotation_capture_start", timeout=timeout, args=(spec,))
        if sorted(row['rank'] for row in started)!=list(range(spec['tp_size'])) or any(
                row.get('layers')!=spec['layer_ids'] or row.get('state')!='armed_after_warmup' for row in started):
            raise RuntimeError("not every native TP worker armed its complete collector")
        completed_tokens=0
        for prompt_index,ids in enumerate(prompts):
            result = llm.generate([{"prompt_token_ids": ids}], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
            if len(result) != 1 or not result[0].finished:
                raise RuntimeError("native calibration prefill did not complete")
            completed_tokens+=len(ids)
            print(json.dumps({'rotation_capture':'native_prefill_complete',
                'prompt':prompt_index+1,'prompts':len(prompts),'tokens':completed_tokens}),flush=True)
        finished = llm.collective_rpc("oscar_rotation_capture_finish", timeout=timeout)
        if sorted(row["rank"] for row in finished) != list(range(spec["tp_size"])) or any(
                row.get('tokens')!=spec['token_count'] or row.get('layers')!=spec['layer_ids'] or
                row.get('status')!='complete' for row in finished):
            raise RuntimeError("native calibration did not return every complete TP rank")
    finally:
        # Abort is idempotent after finish and restores partially armed ranks
        # after RPC/generation failures. Preserve the first exception if cleanup
        # also fails; successful capture still requires successful shutdown.
        primary_error=sys.exc_info()[0] is not None
        cleanup_errors=[]
        try:
            llm.collective_rpc('oscar_rotation_capture_abort',timeout=min(timeout,30))
        except Exception as error:
            cleanup_errors.append('restore: '+str(error))
        try:
            llm.llm_engine.engine_core.shutdown(timeout=30)
        except Exception as error:
            cleanup_errors.append('shutdown: '+str(error))
        if cleanup_errors:
            message='native calibration cleanup failed: '+ '; '.join(cleanup_errors)
            if primary_error:print(message,file=sys.stderr,flush=True)
            else:raise RuntimeError(message)
    report = {**spec, "rank_paths": [row["path"] for row in sorted(finished, key=lambda row: row["rank"])],
        "prompt_source": str(source), "prompt_source_sha256": source_sha256,
        "worker_multiproc_method": os.environ["VLLM_WORKER_MULTIPROC_METHOD"],
        "status": "native_post_rope_capture_complete", "forward_passes_per_prompt": 1,
        "weights_quantization": config.get("quantization"), "performance": "not_measured"}
    for row in finished:
        path=Path(row['path']).resolve()
        if path!=output_dir/f'rank{row["rank"]}.pt' or not path.is_file() or path.stat().st_size==0:
            raise RuntimeError('native worker rank payload path/missing file differs')
    report['rank_sha256']={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in report['rank_paths']}
    atomic_json(output_dir / "capture-manifest.json", report)
    print(json.dumps({"rotation_capture": report["status"], "tokens": spec["token_count"],
        "layers": len(layers), "ranks": spec["tp_size"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.config, args.output_dir)


if __name__ == "__main__":
    main()
