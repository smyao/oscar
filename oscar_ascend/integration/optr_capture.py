"""Archive #23-25/#28/#86/#94-97/#165/#166: isolated native OptR traces.

This extension is installed only by the preparation subprocess with OSCAR off.
No hook or CPU copy is installed in serving. Native Qwen3NextAttention and the
Ascend patch define the interleaved per-head Q/gate projection layout; native
AttentionImpl.forward defines the caller-owned output buffer ABI. Target FULL
layers are explicit, and MTP is not sampled (#86). Every failure is visible and
hooks are restored with ownership checks; partial traces are never published.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from oscar_ascend.integration.rotation_capture import full_layer_id, _restore


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(payload, path):
    import torch
    path = Path(path)
    if path.exists():
        raise ValueError(f"OptR trace refuses to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _cpu_bf16(tensor, shape, label):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"OptR {label} must have BF16 shape {tuple(shape)}")
    value = tensor.detach().to(device="cpu").contiguous().clone()
    if not torch.isfinite(value).all():
        raise ValueError(f"OptR {label} contains non-finite values")
    return value


def _token_ids(value, label):
    if not isinstance(value, list) or not value or any(type(v) is not int or v < 0 for v in value):
        raise ValueError(f"OptR {label} requires nonempty nonnegative integer token IDs")
    return list(value)


def _sample_rows(length, maximum):
    """Uniform, deterministic frame sampling, including both frame endpoints."""
    import torch
    if type(length) is not int or length <= 0 or type(maximum) is not int or maximum <= 0:
        raise ValueError("OptR sampling extents must be positive integers")
    count = min(length, maximum)
    if count == 1:
        return torch.tensor([length - 1], dtype=torch.int64)
    return torch.tensor([i * (length - 1) // (count - 1) for i in range(count)], dtype=torch.int64)


def _output_projection(parent, rank, tp_size, hq, dim):
    projection = parent.o_proj
    weight = projection.weight
    if (not getattr(projection, "input_is_parallel", False)
            or getattr(projection, "tp_rank", None) != rank
            or getattr(projection, "tp_size", None) != tp_size
            or getattr(projection, "bias", None) is not None
            or weight.device.type not in {"npu", "privateuseone"}):
        raise RuntimeError("OptR requires the actual native BF16 RowParallel output shard on NPU")
    wo = _cpu_bf16(weight, (parent.hidden_size, hq * dim), "output projection weight")
    return wo.reshape(parent.hidden_size, hq, dim).permute(1, 0, 2).contiguous()


def _native_core_geometry(requested):
    """Same driver property source as RuntimeProvider._device_cube_cores."""
    from vllm_ascend.ops.triton.triton_utils import (
        get_aicore_num, get_vectorcore_num, init_device_properties_triton)
    init_device_properties_triton()
    aic, aiv = int(get_aicore_num()), int(get_vectorcore_num())
    selected = min(aic, 32) if requested is None else requested
    if (aic <= 0 or aiv < 2 * aic or type(selected) is not int or
            not 1 <= selected <= min(aic, 32)):
        raise RuntimeError("OptR native AIC/AIV or configured Cube geometry is invalid")
    return {"aicore_count": aic, "vectorcore_count": aiv, "selected_cube_cores": selected}


class LayerTrace:
    """One request's complete K/V and sparse query/teacher rows, on CPU only."""

    def __init__(self, head_dim, local_q_heads, local_kv_heads, samples_per_frame, max_length):
        if any(type(x) is not int or x <= 0 for x in
               (head_dim, local_q_heads, local_kv_heads, samples_per_frame, max_length)):
            raise ValueError("OptR trace geometry must be positive integers")
        if local_kv_heads != 1 or local_q_heads % local_kv_heads:
            raise ValueError("OptR capture requires disjoint one-KV-head TP shards")
        self.dim, self.hq, self.hk = head_dim, local_q_heads, local_kv_heads
        self.samples, self.max_length = samples_per_frame, max_length
        self.count = 0
        self.frames, self.k, self.v, self.q = [], [], [], []
        self.gate, self.teacher, self.positions, self.contexts, self.frame_ids = [], [], [], [], []

    def prepare(self, query, key, value, gate, positions, metadata):
        import torch
        n = getattr(metadata, "num_actual_tokens", None)
        ends = getattr(metadata, "actual_seq_lengths_q", None)
        lengths = getattr(metadata, "seq_lens_list", None)
        state = getattr(getattr(metadata, "attn_state", None), "name", None)
        if state not in {"PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill", "DecodeOnly"}:
            raise ValueError("OptR capture requires a native prefill or decode frame")
        if (type(n) is not int or n <= 0 or not isinstance(ends, list) or not ends or ends[0] != n
                or not isinstance(lengths, list) or not lengths or type(lengths[0]) is not int
                or lengths[0] != self.count + n):
            raise ValueError("OptR frame must continue exactly one uncached request")
        # Ascend flashcomm may append a dummy padding request. It starts beyond
        # num_actual_tokens, so it must never enter the saved history or teacher.
        if any(type(x) is not int or x <= n for x in ends[1:]):
            raise ValueError("OptR capture saw a second real request")
        if self.count + n > self.max_length or (state == "DecodeOnly" and n != 1):
            raise ValueError("OptR frame exceeds request bounds or native single-token decode")
        tensors = ((query, self.hq, "query"), (key, self.hk, "key"), (value, self.hk, "value"))
        for tensor, heads, label in tensors:
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim not in (2, 3)
                    or tensor.shape[0] < n or tensor.numel() != tensor.shape[0] * heads * self.dim
                    or tensor.dtype != torch.bfloat16
                    or tensor.ndim == 3 and tuple(tensor.shape[1:]) != (heads, self.dim)):
                raise ValueError(f"OptR {label} tensor geometry/dtype differs")
        if (not isinstance(gate, torch.Tensor) or gate.ndim != 3
                or gate.shape[0] < n or tuple(gate.shape[1:]) != (self.hq, self.dim)):
            raise ValueError("OptR gate tensor geometry differs")
        if not torch.isfinite(query[:n]).all() or not torch.isfinite(gate[:n]).all():
            raise ValueError("OptR actual query/gate contains non-finite values")
        if not isinstance(positions, torch.Tensor) or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("OptR native positions must be integer tensors")
        pos = positions.detach().to(device="cpu", dtype=torch.int64)
        if pos.ndim == 2:
            if pos.shape[0] != 3 or pos.shape[1] < n or not torch.equal(pos[:, :n], pos[:1, :n].expand(3, n)):
                raise ValueError("OptR calibration supports text-only equal-plane MRoPE positions")
            pos = pos[0]
        if pos.ndim != 1 or pos.numel() < n or not torch.equal(pos[:n], torch.arange(self.count, self.count + n)):
            raise ValueError("OptR native absolute positions disagree with complete K/V history")
        rows = _sample_rows(n, self.samples)
        device_rows = rows.to(query.device)
        sample_q = query[:n].reshape(n, self.hq, self.dim).index_select(0, device_rows)
        sample_gate = gate[:n].index_select(0, rows.to(gate.device))
        captured = {
            "q": _cpu_bf16(sample_q, (len(rows), self.hq, self.dim), "sampled query"),
            "gate": _cpu_bf16(sample_gate, (len(rows), self.hq, self.dim), "sampled sigmoid gate"),
            "k": _cpu_bf16(key[:n].reshape(n, self.hk, self.dim), (n, self.hk, self.dim), "key"),
            "v": _cpu_bf16(value[:n].reshape(n, self.hk, self.dim), (n, self.hk, self.dim), "value"),
            "positions": pos[:n].index_select(0, rows).clone(),
            "rows": rows, "n": n, "context": self.count, "state": state,
            "input_tokens": query.shape[0],
        }
        if (captured["gate"] < 0).any() or (captured["gate"] > 1).any():
            raise ValueError("OptR gate must already be sigmoid probabilities")
        return captured

    def complete(self, captured, output):
        import torch
        n, rows = captured["n"], captured["rows"]
        if captured["context"] != self.count:
            raise RuntimeError("OptR duplicate/out-of-order native frame")
        if (not isinstance(output, torch.Tensor) or output.ndim not in (2, 3) or output.shape[0] < n
                or output.numel() != output.shape[0] * self.hq * self.dim
                or output.ndim == 3 and tuple(output.shape[1:]) != (self.hq, self.dim)):
            raise ValueError("OptR native pre-gate teacher output ABI differs")
        if not torch.isfinite(output[:n]).all():
            raise ValueError("OptR actual native teacher contains non-finite values")
        teacher = output[:n].reshape(n, self.hq, self.dim).index_select(0, rows.to(output.device))
        teacher = _cpu_bf16(teacher, (len(rows), self.hq, self.dim), "native pre-gate teacher")
        for key in ("q", "k", "v", "gate", "positions"):
            getattr(self, key).append(captured[key])
        self.teacher.append(teacher)
        self.contexts.append(torch.full((len(rows),), self.count, dtype=torch.int64))
        self.frame_ids.append(torch.full((len(rows),), len(self.frames), dtype=torch.int64))
        self.frames.append({"context": self.count, "query_length": n,
                            "sequence_length": self.count + n, "state": captured["state"],
                            "input_tokens": captured["input_tokens"]})
        self.count += n

    def payload(self, token_ids):
        import torch
        token_ids = _token_ids(token_ids, "captured history")
        if self.count != len(token_ids) or not self.frames:
            raise ValueError("OptR complete history and actual token identities differ")
        result = {name: torch.cat(getattr(self, name), dim=0) for name in
                  ("q", "k", "v", "gate", "teacher", "positions", "contexts", "frame_ids")}
        result.update(token_ids=torch.tensor(token_ids, dtype=torch.int64), frames=self.frames)
        return result


def _restore_all(patches, handles):
    errors = []
    for handle in reversed(handles):
        try:
            handle.remove()
        except Exception as error:
            errors.append(str(error))
    try:
        _restore(patches)
    except Exception as error:
        errors.append(str(error))
    if errors:
        raise RuntimeError("OptR hook restoration failed: " + "; ".join(errors))


class OptRCaptureWorker:
    def oscar_optr_capture_start(self, spec):
        if os.environ.get("OSCAR_ENABLED") != "0":
            raise RuntimeError("OptR sampling requires an isolated native worker")
        if getattr(self, "_oscar_optr_capture", None) is not None:
            raise RuntimeError("OptR capture is already armed")
        from vllm.distributed import get_tensor_model_parallel_rank
        import torch
        rank = get_tensor_model_parallel_rank()
        required = ("tp_size", "head_dim", "global_q_heads", "global_kv_heads", "samples_per_frame", "max_length", "decode_tokens")
        if (any(type(spec.get(k)) is not int or spec[k] <= 0 for k in required)
                or type(rank) is not int or not 0 <= rank < spec["tp_size"]
                or spec["global_kv_heads"] != spec["tp_size"]
                or spec["global_q_heads"] % spec["tp_size"]):
            raise ValueError("OptR capture rank/head geometry differs")
        expected = spec.get("layer_ids")
        if (not isinstance(expected, list) or not expected or len(set(expected)) != len(expected)
                or any(type(i) is not int or i < 0 for i in expected)):
            raise ValueError("OptR target FULL layer IDs are invalid")
        directory = Path(spec["output_dir"]).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        state = {"spec": spec, "rank": rank, "layers": {}, "patches": [], "handles": [],
                 "request": None, "completed": set(), "output_dir": directory,
                 "device_geometry": _native_core_geometry(spec.get("workspace_geometry", {}).get("requested_cube_cores"))}
        modules = dict(self.get_model().named_modules())
        seen = set()
        try:
            for name, attention in modules.items():
                layer_id = full_layer_id(getattr(attention, "layer_name", name))
                if layer_id not in expected or not hasattr(attention, "impl"):
                    continue
                impl = attention.impl
                if layer_id in state["layers"] or id(impl) in seen:
                    raise RuntimeError("OptR native FULL layer has multiple owners")
                if not type(impl).__module__.startswith("vllm_ascend.attention."):
                    raise RuntimeError("OptR must observe native Ascend attention")
                parent = modules.get(name.rsplit(".", 1)[0])
                dim, hq, hk = impl.head_size, impl.num_heads, impl.num_kv_heads
                if (parent is None or not getattr(parent, "attn_output_gate", False)
                        or dim != spec["head_dim"] or hq * spec["tp_size"] != spec["global_q_heads"]
                        or hk != 1 or parent.q_size != hq * dim or parent.kv_size != hk * dim):
                    raise RuntimeError("OptR Qwen gate/TP geometry differs")
                wo = _output_projection(parent, rank, spec["tp_size"], hq, dim)
                saved = _save({"schema_version": 1, "wo": wo, "rank": rank, "layer_id": layer_id, "head_dim": dim,
                               "global_q_head_start": rank * hq}, directory / f"rank{rank}-layer{layer_id}-wo.pt")
                entry = {"wo_path": saved["path"], "wo_sha256": saved["sha256"],
                         "sequences": [], "pending_positions": None, "pending_gate": None, "trace": None}
                state["layers"][layer_id] = entry
                state["handles"].append(parent.register_forward_pre_hook(
                    _positions_hook(state, entry), with_kwargs=True))
                state["handles"].append(parent.qkv_proj.register_forward_hook(_gate_hook(state, entry, hq, hk, dim)))
                original, had_own, previous = impl.forward, "forward" in impl.__dict__, impl.__dict__.get("forward")
                wrapped = _attention_hook(original, state, entry)
                impl.forward = wrapped
                state["patches"].append((impl, wrapped, had_own, previous))
                seen.add(id(impl))
            if set(state["layers"]) != set(expected):
                raise RuntimeError("OptR target FULL layer coverage differs; MTP is not part of sampling")
        except BaseException:
            _restore_all(state["patches"], state["handles"])
            raise
        self._oscar_optr_capture = state
        return {"rank": rank, "layers": sorted(expected), "state": "armed_after_warmup"}

    def oscar_optr_capture_begin(self, request):
        state = self._optr_state()
        if state["request"] is not None:
            raise RuntimeError("OptR requests cannot overlap")
        request_id = request.get("request_id")
        if (not isinstance(request_id, str) or not request_id or not all(c.isalnum() or c in "_-" for c in request_id)
                or request_id in state["completed"] or request_id not in state["spec"]["request_ids"]
                or request.get("split") not in {"train", "validation", "test"}
                or type(request.get("length_bucket")) is not int or request["length_bucket"] <= 0):
            raise ValueError("OptR request identity/split is invalid")
        ids = _token_ids(request.get("prompt_token_ids"), "prompt")
        spec = state["spec"]
        if len(ids) != request["length_bucket"] or len(ids) + spec["decode_tokens"] > spec["max_length"]:
            raise ValueError("OptR prompt length differs from the declared calibration bucket")
        for entry in state["layers"].values():
            if entry["pending_gate"] is not None or entry["pending_positions"] is not None:
                raise RuntimeError("OptR previous native frame was not consumed")
            entry["trace"] = LayerTrace(spec["head_dim"], spec["global_q_heads"] // spec["tp_size"], 1,
                                        spec["samples_per_frame"], len(ids) + spec["decode_tokens"])
        state["request"] = {**request, "prompt_token_ids": ids}
        return {"rank": state["rank"], "request_id": request_id, "status": "request_armed"}

    def oscar_optr_capture_end(self, result):
        state = self._optr_state()
        request = state["request"]
        if request is None or result.get("request_id") != request["request_id"]:
            raise ValueError("OptR request completion identity differs")
        output_ids = _token_ids(result.get("output_token_ids"), "native generated output")
        if len(output_ids) != state["spec"]["decode_tokens"] + 1:
            raise ValueError("OptR native generation did not cover the requested decode frames")
        token_ids = request["prompt_token_ids"] + output_ids[:-1]
        frames = None
        try:
            for layer_id, entry in state["layers"].items():
                trace = entry["trace"]
                if entry["pending_positions"] is not None or entry["pending_gate"] is not None:
                    raise RuntimeError("OptR request ended with an incomplete native frame")
                payload = trace.payload(token_ids)
                if frames is not None and frames != payload["frames"]:
                    raise RuntimeError("OptR target FULL layers observed different request schedules")
                frames = payload["frames"]
                if sum(f["query_length"] for f in frames if f["context"] >= len(request["prompt_token_ids"])) != state["spec"]["decode_tokens"]:
                    raise RuntimeError("OptR trace does not contain all native decode frames")
                identity = {"request_id": request["request_id"], "split": request["split"],
                            "length_bucket": request["length_bucket"]}
                payload.update(identity, schema_version=1, rank=state["rank"], layer_id=layer_id,
                               global_kv_head=state["rank"], prompt_length=len(request["prompt_token_ids"]),
                               generated_output_token_ids=output_ids, model_fingerprint=state["spec"]["model_fingerprint"],
                               source_record_sha256=request.get("source_record_sha256", []))
                saved = _save(payload, state["output_dir"] / f"rank{state['rank']}-layer{layer_id}-{request['request_id']}.pt")
                entry["sequences"].append({**identity, **saved})
                entry["trace"] = None
            state["completed"].add(request["request_id"])
        finally:
            for entry in state["layers"].values():
                entry["trace"] = None
                entry["pending_positions"] = entry["pending_gate"] = None
            state["request"] = None
        return {"rank": state["rank"], "request_id": request["request_id"], "status": "request_saved", "tokens": len(token_ids)}

    def oscar_optr_capture_finish(self):
        from tools.phase import atomic_json
        state = self._optr_state()
        try:
            if state["request"] is not None or state["completed"] != set(state["spec"]["request_ids"]):
                raise RuntimeError("OptR capture request coverage is incomplete")
            manifest = {k: state["spec"][k] for k in
                        ("tp_size", "head_dim", "global_q_heads", "global_kv_heads", "model_fingerprint", "prompt_sha256")}
            manifest.update(schema_version=1, rank=state["rank"], status="complete",
                            device_geometry=state["device_geometry"],
                            workspace_geometry=state["spec"].get("workspace_geometry"),
                            layers={str(i): {k: e[k] for k in ("wo_path", "wo_sha256", "sequences")}
                                    for i, e in state["layers"].items()})
            path = state["output_dir"] / f"rank{state['rank']}-manifest.json"
            atomic_json(path, manifest)
            return {"rank": state["rank"], "path": str(path), "sha256": file_sha256(path),
                    "layers": sorted(state["layers"]), "status": "complete"}
        finally:
            try:
                _restore_all(state["patches"], state["handles"])
            finally:
                self._oscar_optr_capture = None

    def oscar_optr_capture_abort(self):
        state = getattr(self, "_oscar_optr_capture", None)
        if state is None:
            return {"status": "not_armed"}
        try:
            _restore_all(state["patches"], state["handles"])
        finally:
            self._oscar_optr_capture = None
        return {"rank": state["rank"], "status": "aborted_hooks_restored"}

    def _optr_state(self):
        state = getattr(self, "_oscar_optr_capture", None)
        if state is None:
            raise RuntimeError("OptR collector was not armed")
        return state


def _positions_hook(state, entry):
    def hook(module, args, kwargs):
        if state["request"] is None or entry["pending_positions"] is not None:
            raise RuntimeError("OptR unexpected native parent forward")
        positions = kwargs.get("positions", args[0] if args else None)
        if positions is None:
            raise ValueError("OptR native parent did not expose absolute positions")
        entry["pending_positions"] = positions.detach().clone()
    return hook


def _gate_hook(state, entry, hq, hk, dim):
    def hook(module, args, output):
        import torch
        if state["request"] is None or entry["pending_gate"] is not None or entry["pending_positions"] is None:
            raise RuntimeError("OptR unexpected QKV projection order")
        qkv = output[0] if isinstance(output, tuple) else output
        if (not isinstance(qkv, torch.Tensor) or qkv.dtype != torch.bfloat16 or qkv.ndim != 2
                or qkv.shape[1] != (2 * hq + 2 * hk) * dim):
            raise ValueError("OptR native Q/gate/K/V projection layout differs")
        # Q and gate are interleaved PER HEAD, not two contiguous Q-size halves.
        gate = qkv[:, :2 * hq * dim].reshape(-1, hq, 2 * dim)[:, :, dim:]
        entry["pending_gate"] = torch.sigmoid(gate)
    return hook


def _attention_hook(original, state, entry):
    def wrapped(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        import torch
        if (state["request"] is None or entry["trace"] is None or entry["pending_gate"] is None
                or entry["pending_positions"] is None or attn_metadata is None):
            raise RuntimeError("OptR native attention frame was not armed completely")
        trace = entry["trace"]
        captured = trace.prepare(query, key, value, entry["pending_gate"], entry["pending_positions"], attn_metadata)
        result = original(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
        output = result if isinstance(result, torch.Tensor) else kwargs.get("output", args[0] if args else None)
        trace.complete(captured, output)
        entry["pending_gate"] = entry["pending_positions"] = None
        return result
    return wrapped
