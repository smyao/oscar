"""Archive #86/#94-97/#165/#166: isolated capture lifecycle and native ABI.

CPU tensors below are explicit test doubles. They are never used as native
model evidence, and the real worker rejects a CPU output-projection weight.
"""
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend.integration import optr_capture as capture
from oscar_ascend.optr_policy import optr_policy
from tools import calibrate_optr_traces as cli


def metadata(n, context=0, padding=0, state="ChunkedPrefill"):
    return NS(num_actual_tokens=n, actual_seq_lengths_q=[n, n + padding] if padding else [n],
              seq_lens_list=[context + n, 1] if padding else [context + n], attn_state=NS(name=state))


def frame(n, context=0, padding=0, state="ChunkedPrefill"):
    q = torch.arange((n + padding) * 24).reshape(n + padding, 6, 4).to(torch.bfloat16)
    k, v = q[:, :1].clone(), q[:, :1].clone() / 8
    gate = torch.full_like(q, .5)
    positions = torch.arange(context, context + n + padding)
    return q, k, v, gate, positions, metadata(n, context, padding, state)


def test_complete_chunked_history_sparse_teacher_positions_and_padding():
    trace = capture.LayerTrace(4, 6, 1, 3, 8)
    for n, context, padding, state in ((4, 0, 2, "ChunkedPrefill"), (2, 4, 1, "ChunkedPrefill"),
                                      (1, 6, 3, "DecodeOnly"), (1, 7, 3, "DecodeOnly")):
        args = list(frame(n, context, padding, state))
        for value in args[:4]:
            value[n:] = float("nan")
        staged = trace.prepare(*args)
        output = args[0] / 2
        trace.complete(staged, output.reshape(n + padding, -1))
    payload = trace.payload(list(range(8)))
    assert payload["k"].shape == (8, 1, 4)
    assert payload["q"].shape == (7, 6, 4)
    assert payload["positions"].tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert payload["contexts"].tolist() == [0, 0, 0, 4, 4, 6, 7]
    assert payload["frame_ids"].tolist() == [0, 0, 0, 1, 1, 2, 3]
    assert torch.equal(payload["teacher"], payload["q"] / 2)
    assert torch.isfinite(payload["teacher"]).all()


@pytest.mark.parametrize("case", ["context_gap", "second_request", "position", "mrope", "gate_shape", "gate_nan", "q_nan", "q_layout", "key_nan", "teacher_nan", "teacher_short", "teacher_layout"])
def test_bad_native_frames_fail_closed(case):
    trace = capture.LayerTrace(4, 6, 1, 2, 8)
    args = list(frame(4))
    if case == "context_gap":
        args[-1].seq_lens_list = [5]
    elif case == "second_request":
        args[-1].actual_seq_lengths_q = [2, 4]
    elif case == "position":
        args[-2][2] = 9
    elif case == "mrope":
        args[-2] = args[-2].expand(3, 4).clone(); args[-2][1, 0] = 99
    elif case == "gate_shape":
        args[3] = args[3][:, :1]
    elif case == "gate_nan":
        args[3][1, 0, 0] = float("nan")
    elif case == "q_nan":
        args[0][1, 0, 0] = float("nan")
    elif case == "q_layout":
        args[0] = args[0].transpose(1, 2)
    elif case == "key_nan":
        args[1][1, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        staged = trace.prepare(*args)
        output = args[0].clone()
        if case == "teacher_nan":
            output[1, 0, 0] = float("nan")
        elif case == "teacher_short":
            output = output[:1]
        elif case == "teacher_layout":
            output = output.transpose(1, 2)
        trace.complete(staged, output)
    assert trace.count == 0


def test_gate_hook_uses_per_head_interleaving_not_contiguous_q_gate_halves():
    state, entry = {"request": {}}, {"pending_gate": None, "pending_positions": torch.arange(2)}
    qkv = torch.zeros(2, 56, dtype=torch.bfloat16)
    gate_logits = torch.arange(24).reshape(6, 4).to(torch.bfloat16) / 4 - 3
    qkv[:, :48].reshape(2, 6, 8)[:, :, 4:] = gate_logits
    capture._gate_hook(state, entry, 6, 1, 4)(None, (), (qkv, None))
    assert torch.equal(entry["pending_gate"], torch.sigmoid(gate_logits).expand(2, 6, 4))
    with pytest.raises(RuntimeError, match="order"):
        capture._gate_hook(state, entry, 6, 1, 4)(None, (), (qkv, None))


@pytest.mark.parametrize("positional", [False, True])
def test_native_inplace_output_abi_and_native_return_preserved(positional):
    trace = capture.LayerTrace(4, 6, 1, 3, 4)
    q, k, v, gate, positions, meta = frame(4)
    output = torch.empty_like(q).reshape(4, 24)
    state = {"request": {"request_id": "x"}}
    entry = {"trace": trace, "pending_gate": gate, "pending_positions": positions}
    def native(layer, query, key, value, cache, metadata, *args, **kwargs):
        destination = args[0] if args else kwargs["output"]
        destination.copy_(query.reshape_as(destination) / 2)
        return None
    wrapped = capture._attention_hook(native, state, entry)
    result = wrapped(None, q, k, v, None, meta, output) if positional else wrapped(None, q, k, v, None, meta, output=output)
    assert result is None and torch.equal(trace.teacher[0], trace.q[0] / 2)
    assert entry["pending_gate"] is None and entry["pending_positions"] is None


class NativeImpl:
    __module__ = "vllm_ascend.attention.attention_v1"
    head_size, num_heads, num_kv_heads = 4, 6, 1
    def forward(self, layer, q, k, v, cache, meta, output=None):
        output.copy_(q / 2)
        return output


class Projection(torch.nn.Module):
    def forward(self, hidden):
        return hidden, None


class NativeParent(torch.nn.Module):
    attn_output_gate, q_size, kv_size, hidden_size = True, 24, 4, 5
    def __init__(self):
        super().__init__()
        self.qkv_proj = Projection()
        self.attn = torch.nn.Module()
        self.attn.impl = NativeImpl()
        self.attn.layer_name = "model.layers.3.self_attn.attn"
    def forward(self, positions, qkv, meta):
        qkv, _ = self.qkv_proj(qkv)
        q_gate = qkv[:, :48].reshape(-1, 6, 8)
        q = q_gate[:, :, :4].reshape(-1, 24)
        k, v = qkv[:, 48:52], qkv[:, 52:]
        output = torch.empty_like(q)
        return self.attn.impl.forward(self.attn, q, k, v, None, meta, output=output)


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setenv("OSCAR_ENABLED", "0")
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 0
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    # The production projection reader explicitly requires NPU. Only this host
    # fixture injects a CPU weight, without a bypass flag in the worker API.
    monkeypatch.setattr(capture, "_output_projection", lambda *_: torch.ones(6, 5, 4, dtype=torch.bfloat16))
    monkeypatch.setattr(capture, "_native_core_geometry", lambda *_: {"aicore_count": 20, "vectorcore_count": 40, "selected_cube_cores": 20})
    parent = NativeParent()
    modules = [("model.layers.3.self_attn", parent), ("model.layers.3.self_attn.attn", parent.attn)]
    instance = capture.OptRCaptureWorker()
    instance.get_model = lambda: NS(named_modules=lambda: modules)
    spec = {"layer_ids": [3], "head_dim": 4, "tp_size": 1, "global_q_heads": 6, "global_kv_heads": 1,
            "samples_per_frame": 3, "max_length": 7, "decode_tokens": 2, "request_ids": ["train_4_0"],
            "model_fingerprint": "model", "prompt_sha256": "prompts", "output_dir": str(tmp_path)}
    request = {"request_id": "train_4_0", "split": "train", "length_bucket": 4, "prompt_token_ids": [10, 11, 12, 13]}
    return instance, parent, spec, request


def test_worker_streams_request_to_disk_and_restores_exact_ownership(worker):
    instance, parent, spec, request = worker
    assert instance.oscar_optr_capture_start(spec)["state"] == "armed_after_warmup"
    instance.oscar_optr_capture_begin(request)
    for n, context, state in ((2, 0, "ChunkedPrefill"), (2, 2, "ChunkedPrefill"), (1, 4, "DecodeOnly"), (1, 5, "DecodeOnly")):
        qkv = torch.arange(n * 56).reshape(n, 56).to(torch.bfloat16) / 100
        parent(positions=torch.arange(context, context + n), qkv=qkv, meta=metadata(n, context, state=state))
    instance.oscar_optr_capture_end({"request_id": request["request_id"], "output_token_ids": [14, 15, 16]})
    entry = instance._oscar_optr_capture["layers"][3]
    assert entry["trace"] is None and len(entry["sequences"]) == 1
    payload = torch.load(entry["sequences"][0]["path"], weights_only=True)
    assert payload["token_ids"].tolist() == [10, 11, 12, 13, 14, 15]
    assert payload["generated_output_token_ids"] == [14, 15, 16]
    assert payload["contexts"].tolist() == [0, 0, 2, 2, 4, 5]
    result = instance.oscar_optr_capture_finish()
    assert result["status"] == "complete"
    manifest = json.loads(Path(result["path"]).read_text())
    assert manifest["layers"]["3"]["wo_sha256"] and manifest["layers"]["3"]["sequences"]
    assert "forward" not in parent.attn.impl.__dict__
    assert not parent._forward_pre_hooks and not parent.qkv_proj._forward_hooks
    assert instance.oscar_optr_capture_abort()["status"] == "not_armed"


def test_worker_partial_start_incomplete_finish_and_production_are_closed(worker, monkeypatch):
    instance, parent, spec, request = worker
    monkeypatch.setenv("OSCAR_ENABLED", "1")
    with pytest.raises(RuntimeError, match="isolated native"):
        instance.oscar_optr_capture_start(spec)
    monkeypatch.setenv("OSCAR_ENABLED", "0")
    with pytest.raises(RuntimeError, match="coverage"):
        instance.oscar_optr_capture_start({**spec, "layer_ids": [3, 7]})
    assert "forward" not in parent.attn.impl.__dict__
    assert not parent._forward_pre_hooks and not parent.qkv_proj._forward_hooks
    # A failed start leaves diagnostic files but cannot be reused as capture.
    for file in Path(spec["output_dir"]).glob("*.pt"):
        file.unlink()
    instance.oscar_optr_capture_start(spec)
    instance.oscar_optr_capture_begin(request)
    with pytest.raises(RuntimeError, match="incomplete"):
        instance.oscar_optr_capture_finish()
    assert "forward" not in parent.attn.impl.__dict__ and instance._oscar_optr_capture is None
    assert not (Path(spec["output_dir"]) / "rank0-manifest.json").exists()


def test_projection_reader_rejects_cpu_model_even_if_weights_look_valid():
    parent = NS(o_proj=NS(weight=torch.ones(5, 24, dtype=torch.bfloat16), input_is_parallel=True,
                         tp_rank=0, tp_size=1, bias=None), hidden_size=5)
    with pytest.raises(RuntimeError, match="on NPU"):
        capture._output_projection(parent, 0, 1, 6, 4)


def test_split_plan_has_no_record_leakage_is_reproducible_and_labels_repetition():
    tokenizer = NS(apply_chat_template=lambda message, **_: [ord(message[0]["content"]), 9])
    messages = [[{"role": "user", "content": letter}] for letter in "abcdef"]
    policy = optr_policy({"optr_calibration": {"capture_lengths": {"train": [5], "validation": [5, 9], "test": [5, 9]}}})
    plan = cli.build_request_plan(tokenizer, messages + messages[:1], policy)
    assert plan == cli.build_request_plan(tokenizer, list(reversed(messages)), policy)
    pools = {split: set(next(r for r in plan if r["split"] == split)["source_record_sha256"])
             for split in ("train", "validation", "test")}
    assert not (pools["train"] & pools["validation"] or pools["train"] & pools["test"] or pools["validation"] & pools["test"])
    assert all(len(row["prompt_token_ids"]) == row["length_bucket"] and row["source_repeated"] for row in plan)
    with pytest.raises(ValueError, match="three distinct"):
        cli.build_request_plan(tokenizer, messages[:2], policy)


@pytest.mark.parametrize("failure", [None, "start", "generate", "cleanup"])
def test_native_runner_preserves_service_config_and_shutdown_on_failure(tmp_path, monkeypatch, failure):
    from tools import model_config
    from oscar_ascend import rotations
    monkeypatch.setattr(cli.os, "environ", dict(cli.os.environ))
    monkeypatch.setattr(model_config, "apply_model_options", lambda config, **_: config)
    monkeypatch.setattr(model_config, "resolve_mamba_cache_dtypes", lambda *_a, **_k:
                        {"mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "bfloat16"})
    monkeypatch.setattr(rotations, "calibration_cache_identity", lambda *_: {"test_double": True})
    model = tmp_path / "model"; model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 4, "full_attention_interval": 4,
        "head_dim": 4, "num_attention_heads": 6, "num_key_value_heads": 1}))
    source = tmp_path / "prompts.jsonl"; source.write_text('"a"\n"b"\n"c"\n')
    config = {"model": str(model), "devices": [0, 1, 2, 3], "tensor_parallel_size": 1,
              "max_model_len": 262144, "max_num_batched_tokens": 16384, "gpu_memory_utilization": .7,
              "rotation_calibration_prompts": str(source), "rotation_refinement": "optr",
              "rotation_variant": "perhead_center", "experimental_official_math": True,
              "oscar_dequant_policy": "fp32_metadata", "optr_calibration": {
                  "capture_lengths": {"train": [4], "validation": [4, 8], "test": [4, 8]}, "decode_tokens": 2}}
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config)); before = config_path.read_bytes()
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = NS(from_pretrained=lambda *_a, **_k: NS(apply_chat_template=lambda *_a, **_k: [7, 8]))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    options, calls, run_spec, request = {}, [], {}, {}
    class LLM:
        def __init__(self, **kwargs):
            assert cli.os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
            assert cli.os.environ["OSCAR_ENABLED"] == "0"
            options.update(kwargs); calls.append("warmup")
            self.llm_engine = NS(engine_core=NS(shutdown=lambda **_: calls.append("shutdown")))
        def collective_rpc(self, name, **kwargs):
            calls.append(name)
            if name.endswith("_start"):
                run_spec.update(kwargs["args"][0])
                if failure == "start":
                    raise RuntimeError("native start failed")
                return [{"rank": 0, "layers": [3], "state": "armed_after_warmup"}]
            if name.endswith("_begin"):
                request.clear(); request.update(kwargs["args"][0])
                return [{"rank": 0, "request_id": request["request_id"], "status": "request_armed"}]
            if name.endswith("_end"):
                return [{"rank": 0, "request_id": request["request_id"], "status": "request_saved", "tokens": request["length_bucket"] + 2}]
            if name.endswith("_finish"):
                path = Path(run_spec["output_dir"]) / "rank0-manifest.json"; path.write_text('{"host_double":true}')
                return [{"rank": 0, "layers": [3], "status": "complete", "path": str(path), "sha256": capture.file_sha256(path)}]
            if name.endswith("_abort") and failure == "cleanup":
                raise RuntimeError("native cleanup failed")
            return [{"status": "not_armed"}]
        def generate(self, prompts, params, **kwargs):
            assert params == {"temperature": 0, "max_tokens": 3, "ignore_eos": True}
            if failure == "generate":
                raise RuntimeError("native generate failed")
            return [NS(finished=True, prompt_token_ids=prompts[0]["prompt_token_ids"], outputs=[NS(token_ids=[9, 10, 11])])]
    vllm = types.ModuleType("vllm"); vllm.LLM = LLM; vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    destination = tmp_path / "capture"
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            cli.collect(config_path, destination)
        assert not (destination / "capture-manifest.json").exists()
    else:
        report = cli.collect(config_path, destination)
        assert report["status"] == "native_optr_capture_complete" and len(report["request_plan"]) == 5
        assert report["rank_sha256"] and report["quality"] == "calibration_only_not_benchmark"
    assert calls[0] == "warmup" and calls[-2:] == ["oscar_optr_capture_abort", "shutdown"]
    assert options["enforce_eager"] and options["enable_chunked_prefill"] and not options["enable_prefix_caching"]
    assert options["max_num_seqs"] == 1 and options["max_num_batched_tokens"] == 16384 and options["max_model_len"] == 11
    assert "speculative_config" not in options and options["quantization"] is None
    assert config_path.read_bytes() == before
