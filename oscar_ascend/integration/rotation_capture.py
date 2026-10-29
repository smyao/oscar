"""Archive #23-25/#28/#86/#96/#97: one-pass native QKV calibration only.

A public vLLM worker extension installed after model warmup. It never replaces
native attention/GDN, runs in the isolated preparation process, and cannot be
armed in an OSCAR production worker. FP64 CPU statistics are offline work.
"""
from __future__ import annotations

import os
from pathlib import Path
import re


def full_layer_id(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$", name)
    return int(match.group(1)) if match else None


class RotationCaptureWorker:
    def oscar_rotation_capture_start(self, spec: dict):
        if os.environ.get("OSCAR_ENABLED") != "0":
            raise RuntimeError("rotation sampling requires an isolated native worker")
        if getattr(self, "_oscar_rotation_capture", None) is not None:
            raise RuntimeError("rotation collector is already armed")
        from vllm.distributed import get_tensor_model_parallel_rank
        from oscar_ascend.rotation_calibration import OfflineLayerAccumulator
        accumulator = OfflineLayerAccumulator
        if spec.get("rotation_seed") is not None:
            from oscar_ascend.optr_oscar_seed import SEED_RECIPE, BaseSeedAccumulator
            if spec["rotation_seed"] != SEED_RECIPE:
                raise ValueError("unsupported source OSCAR seed capture")
            accumulator = BaseSeedAccumulator
        rank = get_tensor_model_parallel_rank()
        expected = set(spec["layer_ids"])
        lengths = spec.get("prompt_lengths")
        if (not expected or len(expected)!=len(spec['layer_ids']) or
                any(type(i) is not int or i<0 for i in expected) or
                not isinstance(lengths,list) or not lengths or
                any(type(n) is not int or n<=0 for n in lengths) or sum(lengths)!=spec['token_count'] or
                type(rank) is not int or not 0<=rank<spec['tp_size']):
            raise ValueError('calibration rank/layer/prompt geometry is invalid')
        collectors, patches, seen = {}, [], set()
        try:
            for name, layer in self.get_model().named_modules():
                layer_id = full_layer_id(getattr(layer, "layer_name", name))
                if layer_id not in expected or not hasattr(layer, "impl"):
                    continue
                impl = layer.impl
                if layer_id in collectors or id(impl) in seen:
                    raise RuntimeError("native FULL attention collection must have one owner per layer")
                if not type(impl).__module__.startswith("vllm_ascend.attention."):
                    raise RuntimeError("calibration must observe the native Ascend attention implementation")
                dim, hq, hk = impl.head_size, impl.num_heads, impl.num_kv_heads
                if dim != spec["head_dim"] or hq * spec["tp_size"] != spec["global_q_heads"]:
                    raise RuntimeError("calibration Q geometry differs from the model")
                if hk * spec["tp_size"] != spec["global_kv_heads"]:
                    raise RuntimeError("Qwen27B calibration requires disjoint TP KV shards")
                collector = accumulator(head_dim=dim, local_kv_heads=hk,
                    local_q_heads=hq, token_budget=spec["token_count"])
                original = impl.forward
                had_own = "forward" in impl.__dict__
                previous = impl.__dict__.get("forward")
                wrapped = _capture_forward(original, collector, dim, hq, hk, lengths)
                impl.forward = wrapped
                patches.append((impl, wrapped, had_own, previous))
                collectors[layer_id] = collector
                seen.add(id(impl))
            if set(collectors) != expected:
                raise RuntimeError(f"native FULL calibration coverage mismatch: {sorted(collectors)} != {sorted(expected)}")
        except BaseException:
            _restore(patches)
            raise
        self._oscar_rotation_capture = (spec, rank, collectors, patches)
        return {"rank": rank, "layers": sorted(collectors), "state": "armed_after_warmup"}

    def oscar_rotation_capture_finish(self):
        import torch
        state = getattr(self, "_oscar_rotation_capture", None)
        if state is None:
            raise RuntimeError("rotation collector was not armed")
        spec, rank, collectors, patches = state
        try:
            payload = {"rank": rank, "tp_size": spec["tp_size"],
                "global_q_heads": spec["global_q_heads"], "global_kv_heads": spec["global_kv_heads"],
                "prompt_sha256": spec["prompt_sha256"], "model_fingerprint": spec["model_fingerprint"],
                "layers": {layer: collector.export_payload() for layer, collector in collectors.items()}}
            for entry in payload["layers"].values():
                if entry["count"] != spec["token_count"]:
                    raise RuntimeError("incomplete calibration token coverage; do not publish rotations")
            path = Path(spec["output_dir"]) / f"rank{rank}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            return {"rank": rank, "path": str(path), "tokens": spec["token_count"],
                    "layers": sorted(collectors), "status": "complete"}
        finally:
            try:
                _restore(patches)
            finally:
                self._oscar_rotation_capture = None

    def oscar_rotation_capture_abort(self):
        state = getattr(self, "_oscar_rotation_capture", None)
        if state is None:
            return {"status": "not_armed"}
        try:
            _restore(state[3])
        finally:
            self._oscar_rotation_capture = None
        return {"rank": state[1], "status": "aborted_hooks_restored"}


def _capture_forward(original, collector, dim, hq, hk, prompt_lengths):
    next_prompt = 0
    def wrapped(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        nonlocal next_prompt
        if attn_metadata is None or key is None or value is None:
            raise RuntimeError("warmup/decode entered the armed prefill-only calibration")
        n = getattr(attn_metadata, "num_actual_tokens", None)
        if type(n) is not int or n <= 0 or any(n>x.shape[0] for x in (query,key,value)):
            raise RuntimeError("native calibration metadata lacks its actual token extent")
        if (getattr(getattr(attn_metadata,'attn_state',None),'name',None) not in
                {'PrefillNoCache','PrefillCacheHit','ChunkedPrefill'}):
            raise RuntimeError('decode or non-single-prefill frame entered calibration')
        ends=getattr(attn_metadata,'actual_seq_lengths_q',None)
        sequences=getattr(attn_metadata,'seq_lens_list',None)
        if (not isinstance(ends,list) or not ends or ends[0]!=n or
                not isinstance(sequences,list) or not sequences or sequences[0]!=n):
            raise RuntimeError('calibration prompt was chunked or reused cached history')
        # Native flashcomm may append a dummy padding request and include it
        # in aggregate prefill/decode counters. The entire real token extent
        # must belong to the FIRST request; all further boundaries are padding.
        if any(type(x) is not int or x<=n for x in ends[1:]):
            raise RuntimeError('unexpected second real calibration request')
        if next_prompt>=len(prompt_lengths) or n!=prompt_lengths[next_prompt]:
            raise RuntimeError('calibration prompt order/length differs or was observed twice')
        collector.observe(query[:n].reshape(n, hq, dim), key[:n].reshape(n, hk, dim),
                          value[:n].reshape(n, hk, dim))
        result=original(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
        next_prompt+=1
        return result
    return wrapped


def _restore(patches):
    conflicts=[]
    for impl, wrapped, had_own, previous in reversed(patches):
        if impl.__dict__.get("forward") is not wrapped:
            conflicts.append(type(impl).__name__)
            continue
        if had_own:
            impl.forward = previous
        else:
            del impl.forward
    if conflicts:
        raise RuntimeError("native calibration seam changed while armed: "+', '.join(conflicts))
