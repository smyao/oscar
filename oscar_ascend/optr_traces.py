"""Archive #23-25/#86/#97/#148/#170/#173: authenticated, offline-only OptR traces.

Load one target FULL layer at a time, verify whole-request source partitions
and TP alignment, and measure the sum of all head contributions before SSE.
No production runtime imports this module. CPU replay evidence cannot certify
NPU arithmetic, full model quality, or end-to-end throughput (#126/#145/#148).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .optr_artifacts import file_sha256
from .optr_policy import (optr_policy, validate_refinement_config,
                          compatible_capture_profiles)


def _hash(value):
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("OptR content hash must be a lowercase SHA256")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _schema(payload, testing, label):
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise ValueError(f"OptR {label} schema differs")
    if type(payload.get("test_only", False)) is not bool:
        raise ValueError(f"OptR {label} test marker must be boolean")
    if payload.get("test_only", False) and not testing:
        raise ValueError(f"OptR {label} test-only data cannot enter production")


def _member(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("OptR trace path is missing")
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("OptR trace path escapes its capture directory or is missing")
    return path


def _read_json(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError("OptR manifest must be a JSON object")
    return result


def _workspace_geometry(config):
    chunk = config.get("max_num_batched_tokens", 16384)
    captures = config.get("compilation_config", {}).get("cudagraph_capture_sizes", [])
    if type(chunk) is not int or chunk <= 0 or not isinstance(captures, list) or any(type(v) is not int or v <= 0 for v in captures):
        raise ValueError("OptR serving token/capture capacities are invalid")
    capacity = max(chunk, max(captures, default=0))
    history = config.get("history_token_capacity")
    history = capacity if history is None else history
    splits = config.get("attention_splits", 1)
    cores = config.get("cube_cores")
    if (type(history) is not int or not 1 <= history <= capacity or type(splits) is not int or not 1 <= splits <= 32
            or (cores is not None and (type(cores) is not int or not 1 <= cores <= 32))):
        raise ValueError("OptR serving history/split/core geometry is invalid")
    return {"token_capacity": capacity, "history_token_capacity": history,
            "attention_splits": splits, "requested_cube_cores": cores}


def _device_geometry(value, workspace):
    if not isinstance(value, dict):
        raise ValueError("OptR trace lacks measured native AIC/AIV counts")
    aic, aiv, selected = (value.get(k) for k in ("aicore_count", "vectorcore_count", "selected_cube_cores"))
    if (any(type(v) is not int for v in (aic, aiv, selected)) or aic <= 0 or aiv < 2 * aic
            or not 1 <= selected <= min(32, aic)):
        raise ValueError("OptR native AIC/AIV/Cube geometry is invalid")
    expected = min(32, aic) if workspace["requested_cube_cores"] is None else workspace["requested_cube_cores"]
    if selected != expected:
        raise ValueError("OptR selected Cube count differs from serving configuration")


def _plan(top, policy):
    rows = top.get("request_plan")
    expected = [(split, n) for split in ("train", "validation", "test") for n in policy["capture_lengths"][split]]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("OptR request plan does not cover its declared policy")
    sources = {split: set() for split in ("train", "validation", "test")}
    ids = []
    for row, (split, length) in zip(rows, expected):
        if not isinstance(row, dict) or row.get("split") != split or row.get("length_bucket") != length:
            raise ValueError("OptR request plan split/length differs")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in ids:
            raise ValueError("OptR request IDs must be nonempty and unique")
        ids.append(request_id)
        _hash(row.get("prompt_sha256"))
        generated = row.get("generated_output_token_ids")
        if (not isinstance(generated, list) or len(generated) != policy["decode_tokens"] + 1
                or any(type(x) is not int or x < 0 for x in generated)):
            raise ValueError("OptR request plan lacks actual generated token identities")
        records = row.get("source_record_sha256")
        if not isinstance(records, list) or not records or len(set(records)) != len(records):
            raise ValueError("OptR source record partition is missing or duplicated")
        sources[split].update(_hash(value) for value in records)
    if sources["train"] & sources["validation"] or sources["train"] & sources["test"] or sources["validation"] & sources["test"]:
        raise ValueError("OptR train/validation/test overlap complete source records")
    if ids != top.get("request_ids"):
        raise ValueError("OptR request identity list differs from its plan")
    return rows


def load_trace_manifest(path, config, layers, dim, fingerprint, testing=False):
    """Validate manifests now; each large tensor file is authenticated on load."""
    from .integration.rotation_capture import full_layer_id
    from .rotations import calibration_cache_identity
    if type(testing) is not bool:
        raise ValueError("OptR testing flag must be explicit boolean")
    validate_refinement_config(config)
    if config.get("rotation_refinement") != "optr":
        raise ValueError("OptR trace loader requires an OptR configuration")
    path = Path(path).resolve()
    top = _read_json(path)
    _schema(top, testing, "capture")
    root = Path(top.get("output_dir", "")).resolve()
    if (top.get("status") != "native_optr_capture_complete" or not root.is_dir()
            or not root.is_relative_to(path.parent)):
        raise ValueError("OptR capture is incomplete or its directory identity differs")
    policy = optr_policy(config)
    profile = calibration_cache_identity(config, fingerprint)
    current = top.get("optr_policy") == policy and top.get("calibration_profile") == profile
    legacy = any(top.get("calibration_profile") == candidate and
                 top.get("optr_policy") == candidate["refinement_policy"]
                 for candidate in compatible_capture_profiles(profile, config))
    if not (current or legacy):
        raise ValueError("OptR trace policy/calibration profile differs")
    if top.get("prompt_source_sha256") != profile["base_calibration_profile"]["prompt_source"]["sha256"]:
        raise ValueError("OptR trace source hash differs from its authenticated calibration source")
    ids = [full_layer_id(name) for name in layers]
    if any(value is None for value in ids) or len(set(ids)) != len(ids) or not ids or top.get("layer_ids") != ids:
        raise ValueError("OptR target FULL layer identity differs")
    tp = config["tensor_parallel_size"]
    model_raw = _read_json(Path(config["model"]) / "config.json")
    model = model_raw.get("text_config", model_raw)
    global_q, global_kv = model.get("num_attention_heads"), model.get("num_key_value_heads")
    hidden_size = model.get("hidden_size")
    if (type(tp) is not int or tp <= 0 or global_kv != tp or type(global_q) is not int or global_q % tp
            or global_q // tp > 16 or type(dim) is not int or dim < 2
            or type(hidden_size) is not int or hidden_size <= 0
            or (not testing and (tp != 4 or dim != 256))):
        raise ValueError("OptR trace requires the supported disjoint TP KV geometry")
    expected_geometry = {"tp_size": tp, "global_q_heads": global_q, "global_kv_heads": global_kv,
                         "head_dim": dim, "model_fingerprint": fingerprint}
    if any(top.get(key) != value for key, value in expected_geometry.items()):
        raise ValueError("OptR capture model/rank/head geometry differs")
    _hash(top.get("prompt_sha256")); _hash(top.get("prompt_source_sha256"))
    plan = _plan(top, policy)
    if top.get("decode_tokens") != policy["decode_tokens"] or top.get("samples_per_frame") != policy["query_samples_per_frame"]:
        raise ValueError("OptR capture sampling policy differs")
    workspace = _workspace_geometry(config)
    if top.get("workspace_geometry") != workspace:
        raise ValueError("OptR capture serving workspace geometry differs")
    paths, hashes = top.get("rank_paths"), top.get("rank_sha256")
    if not isinstance(paths, list) or len(paths) != tp or not isinstance(hashes, dict) or set(hashes) != set(paths):
        raise ValueError("OptR capture rank paths/hashes are incomplete")
    ranks, owned = [], set()
    for rank, entry in enumerate(paths):
        rank_path = _member(root, entry)
        if rank_path in owned or file_sha256(rank_path) != _hash(hashes[entry]):
            raise ValueError("OptR rank manifest changed or is duplicated")
        owned.add(rank_path)
        payload = _read_json(rank_path)
        _schema(payload, testing, "rank manifest")
        if (payload.get("rank") != rank or payload.get("status") != "complete"
                or any(payload.get(key) != value for key, value in expected_geometry.items())
                or payload.get("prompt_sha256") != top["prompt_sha256"]
                or payload.get("workspace_geometry") != workspace
                or set(payload.get("layers", {})) != {str(i) for i in ids}):
            raise ValueError("OptR rank manifest model/layer/request identity differs")
        _device_geometry(payload.get("device_geometry"), workspace)
        for layer_id in ids:
            layer = payload["layers"][str(layer_id)]
            if not isinstance(layer, dict):
                raise ValueError("OptR layer manifest must be an object")
            _member(root, layer.get("wo_path")); _hash(layer.get("wo_sha256"))
            sequences = layer.get("sequences")
            if not isinstance(sequences, list) or len(sequences) != len(plan):
                raise ValueError("OptR layer request coverage differs")
            for sequence, row in zip(sequences, plan):
                if not isinstance(sequence, dict) or any(sequence.get(k) != row[k] for k in ("request_id", "split", "length_bucket")):
                    raise ValueError("OptR layer request identity/order differs")
                sequence_path = _member(root, sequence.get("path"))
                if sequence_path in owned:
                    raise ValueError("OptR sequence file is shared by multiple layer/rank/request owners")
                owned.add(sequence_path)
                _hash(sequence.get("sha256"))
        ranks.append(payload)
    return {**top, "_rank_manifests": ranks, "_manifest_path": str(path), "_manifest_sha256": file_sha256(path),
            "_testing": testing, "_root": str(root), "_hidden_size": hidden_size}


def _tensor(value, dtype, shape, label):
    import torch
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != dtype
            or tuple(value.shape) != tuple(shape)):
        raise ValueError(f"OptR {label} tensor geometry/dtype differs")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"OptR {label} contains nonfinite values")


def _load_payload(manifest, path, expected_hash):
    import torch
    path = _member(Path(manifest["_root"]), path)
    if file_sha256(path) != _hash(expected_hash):
        raise ValueError("OptR tensor file hash differs")
    result = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(result, dict):
        raise ValueError("OptR tensor payload must be an object")
    if type(result.get("test_only", False)) is not bool or result.get("test_only", False) and not manifest["_testing"]:
        raise ValueError("OptR tensor test-only data cannot enter production")
    return result


def _frame_vectors(payload, policy, workspace, device, group):
    import torch
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("OptR native frames are missing")
    cursor, positions, contexts, frame_ids, current_ends, split_counts = 0, [], [], [], [], []
    query_tile = 128 // group
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("OptR frame must be an object")
        n, inputs, end = frame.get("query_length"), frame.get("input_tokens"), frame.get("sequence_length")
        if (any(type(v) is not int for v in (n, inputs, end)) or n <= 0 or inputs < n
                or inputs > workspace["token_capacity"] or frame.get("context") != cursor or end != cursor + n
                or frame.get("state") not in {"PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill", "DecodeOnly"}):
            raise ValueError("OptR native frame extent/position/schedule differs")
        if (cursor < payload["prompt_length"] and end > payload["prompt_length"]
                or cursor >= payload["prompt_length"] and (n != 1 or frame["state"] != "DecodeOnly")
                or frame["state"] == "DecodeOnly" and n != 1):
            raise ValueError("OptR prefill/decode boundary is invalid")
        count = min(n, policy["query_samples_per_frame"])
        rows = [n - 1] if count == 1 else [j * (n - 1) // (count - 1) for j in range(count)]
        groups = (inputs + query_tile - 1) // query_tile
        cube_parallelism = (device["selected_cube_cores"] + groups - 1) // groups
        arena = workspace["history_token_capacity"] * workspace["attention_splits"] // inputs
        splits = max(1, min(32, cube_parallelism, arena))
        positions.extend(cursor + row for row in rows); contexts.extend([cursor] * count)
        frame_ids.extend([index] * count); current_ends.extend([end] * count); split_counts.extend([splits] * count)
        cursor = end
    if cursor != payload["prompt_length"] + policy["decode_tokens"]:
        raise ValueError("OptR native frames do not cover the complete request")
    return {key: torch.tensor(value, dtype=torch.int64) for key, value in
            (("positions", positions), ("contexts", contexts), ("frame_ids", frame_ids),
             ("current_ends", current_ends), ("history_splits", split_counts))}


def load_layer_samples(manifest, layer_id):
    """Return {split: [head0_samples, head1_samples, ...]} for one FULL layer."""
    import torch
    if layer_id not in manifest.get("layer_ids", []) or "_rank_manifests" not in manifest:
        raise ValueError("OptR layer must belong to a validated trace manifest")
    policy, dim = manifest["optr_policy"], manifest["head_dim"]
    group = manifest["global_q_heads"] // manifest["tp_size"]
    result = {split: [] for split in ("train", "validation", "test")}
    cross_rank, prompt_lists = {}, None
    hidden = None
    for rank, rank_manifest in enumerate(manifest["_rank_manifests"]):
        layer = rank_manifest["layers"][str(layer_id)]
        wo_payload = _load_payload(manifest, layer["wo_path"], layer["wo_sha256"])
        _schema(wo_payload, manifest["_testing"], "output projection")
        wo = wo_payload.get("wo")
        if (not isinstance(wo, torch.Tensor) or wo.ndim != 3 or wo.shape[1] != manifest["_hidden_size"]
                or wo_payload.get("rank") != rank or wo_payload.get("layer_id") != layer_id
                or wo_payload.get("head_dim") != dim or wo_payload.get("global_q_head_start") != rank * group):
            raise ValueError("OptR output projection rank/head/layer identity differs")
        _tensor(wo, torch.bfloat16, (group, wo.shape[1], dim), "output projection")
        if hidden is not None and hidden != wo.shape[1]:
            raise ValueError("OptR output projection hidden dimensions differ across ranks")
        hidden = wo.shape[1]
        wo = wo.float()
        split_samples = {split: [] for split in result}
        prompts = []
        for sequence, row in zip(layer["sequences"], manifest["request_plan"]):
            payload = _load_payload(manifest, sequence["path"], sequence["sha256"])
            _schema(payload, manifest["_testing"], "sequence")
            if (any(payload.get(k) != row[k] for k in ("request_id", "split", "length_bucket"))
                    or payload.get("rank") != rank or payload.get("global_kv_head") != rank
                    or payload.get("layer_id") != layer_id or payload.get("model_fingerprint") != manifest["model_fingerprint"]
                    or payload.get("prompt_length") != row["length_bucket"]
                    or payload.get("source_record_sha256") != row["source_record_sha256"]):
                raise ValueError("OptR sequence model/rank/head/source identity differs")
            length = row["length_bucket"] + policy["decode_tokens"]
            _tensor(payload.get("token_ids"), torch.int64, (length,), "actual token IDs")
            if (payload["token_ids"] < 0).any():
                raise ValueError("OptR token IDs must be nonnegative")
            tokens = payload["token_ids"].tolist()
            generated = payload.get("generated_output_token_ids")
            if (not isinstance(generated, list) or len(generated) != policy["decode_tokens"] + 1
                    or any(type(x) is not int or x < 0 for x in generated)
                    or tokens[row["length_bucket"]:] != generated[:-1]
                    or generated != row["generated_output_token_ids"]):
                raise ValueError("OptR generated-token identities disagree with captured history")
            prompt = tokens[:row["length_bucket"]]
            if _digest([prompt]) != row["prompt_sha256"]:
                raise ValueError("OptR actual prompt token hash differs")
            prompts.append(prompt)
            vectors = _frame_vectors(payload, policy, manifest["workspace_geometry"], rank_manifest["device_geometry"], group)
            m = len(vectors["positions"])
            for key in ("positions", "contexts", "frame_ids"):
                _tensor(payload.get(key), torch.int64, (m,), key)
                if not torch.equal(payload[key], vectors[key]):
                    raise ValueError("OptR sampled query positions/contexts differ from actual native frames")
            for key in ("q", "gate", "teacher"):
                _tensor(payload.get(key), torch.bfloat16, (m, group, dim), key)
            for key in ("k", "v"):
                _tensor(payload.get(key), torch.bfloat16, (length, 1, dim), key)
            if (payload["gate"] < 0).any() or (payload["gate"] > 1).any():
                raise ValueError("OptR gate must contain native sigmoid probabilities")
            request_id = row["request_id"]
            alignment = {"tokens": tokens, "generated": generated, "frames": payload["frames"],
                         "positions": payload["positions"].tolist(), "contexts": payload["contexts"].tolist(),
                         "source_records": payload["source_record_sha256"]}
            if request_id in cross_rank and alignment != cross_rank[request_id]:
                raise ValueError("OptR TP ranks captured different request tokens/positions/frames")
            cross_rank[request_id] = alignment
            sample_hash = _digest({"sequence": sequence["sha256"], "wo": layer["wo_sha256"],
                                   "rank": rank, "layer_id": layer_id})
            sample = {key: payload[key] for key in ("request_id", "q", "gate", "teacher", "token_ids")}
            sample.update(k=payload["k"][:, 0], v=payload["v"][:, 0], wo=wo, **vectors,
                          sample_sha256=sample_hash, split=row["split"], length_bucket=row["length_bucket"],
                          rank=rank, global_kv_head=rank, source_record_sha256=row["source_record_sha256"])
            split_samples[row["split"]].append(sample)
        if _digest(prompts) != manifest["prompt_sha256"] or (prompt_lists is not None and prompts != prompt_lists):
            raise ValueError("OptR complete prompt plan hash or TP token identities differ")
        prompt_lists = prompts
        for split in result:
            result[split].append(split_samples[split])
    return result


def _query_metrics(first, actual, teacher, request_id, identity, *, all_queries):
    """Reduce the same captured rows; preserve individual prefill errors."""
    import torch
    frame_ids = first["frame_ids"]
    _tensor(frame_ids, torch.int64, (actual.shape[0],), "joint frame IDs")
    unique = torch.unique_consecutive(frame_ids)
    if not torch.equal(unique, torch.arange(unique.numel())):
        raise ValueError("OptR joint metric requires ordered contiguous captured frame IDs")
    query_errors = (actual - teacher).square().sum(-1)
    query_energies = teacher.square().sum(-1)
    if not bool(torch.isfinite(query_errors).all() & torch.isfinite(query_energies).all()):
        raise ValueError("OptR joint metric is nonfinite")
    frames = []
    for frame_id in unique.tolist():
        selected = frame_ids == frame_id
        contexts, ends, positions = (first[key][selected] for key in ("contexts", "current_ends", "positions"))
        if (not bool((contexts == contexts[0]).all()) or not bool((ends == ends[0]).all())
                or bool((positions[1:] <= positions[:-1]).any())
                or int(positions[-1]) != int(ends[0]) - 1):
            raise ValueError("OptR joint metric captured frame extents/positions differ")
        errors, energies = query_errors[selected], query_energies[selected]
        error, energy = float(errors.sum()), float(energies.sum())
        terminal_error, terminal_energy = float(errors[-1]), float(energies[-1])
        frame = {"frame_id": frame_id, "context": int(contexts[0]), "current_end": int(ends[0]),
                 "query_positions": positions.tolist(), "squared_error": error, "teacher_energy": energy,
                 "normalized_error": error / max(energy, 1e-20),
                 "terminal_query": {"position": int(positions[-1]), "squared_error": terminal_error,
                     "teacher_energy": terminal_energy,
                     "normalized_error": terminal_error / max(terminal_energy, 1e-20)}}
        if all_queries:
            frame["queries"] = [{"position": position, "squared_error": err, "teacher_energy": eng,
                "normalized_error": err / max(eng, 1e-20)}
                for position, err, eng in zip(positions.tolist(), errors.tolist(), energies.tolist())]
        frames.append(frame)
    error, energy = (sum(frame[key] for frame in frames) for key in ("squared_error", "teacher_energy"))
    return {"request_id": request_id, "sample_sha256": identity, "squared_error": error,
            "teacher_energy": energy, "normalized_error": error / max(energy, 1e-20), "frames": frames}


def _aggregate_requests(rows):
    error, energy = (sum(row[key] for row in rows) for key in ("squared_error", "teacher_energy"))
    return {"squared_error": error, "teacher_energy": energy,
            "normalized_error": error / max(energy, 1e-20), "requests": rows}


def aggregate_layer_error(layer_matrices, head_samples, policy, *,
                          projected_outputs=None, projected_teachers=None):
    """Sum TP/GQA outputs before SSE and preserve each captured forward.

    Archive #170/#173/#176: lower aggregates can conceal worse individual
    queries, and FP32 projection error can conceal BF16 output regressions.
    Current policies retain every captured query and visible output rounding
    view without additional attention replay or W_O products. Each view has
    separate totals; query details do not count an output twice.
    Optional [head][request] FP32 projected tensors let offline head-subset
    selection reuse the same replay. They never enter the serving path.
    """
    import torch
    from .optr_calibration import replay_sample, project_output
    if not isinstance(head_samples, list) or not head_samples or any(not isinstance(head, list) or not head for head in head_samples):
        raise ValueError("OptR joint metric requires every KV head and nonempty request samples")
    heads = len(head_samples)
    mean = layer_matrices.get("k_mean")
    if not isinstance(mean, torch.Tensor) or mean.ndim != 2 or mean.shape[0] != heads:
        raise ValueError("OptR joint metric matrix/head geometry differs")
    dim = mean.shape[1]
    _tensor(mean, torch.float32, (heads, dim), "joint K mean")
    for key in ("Rk", "Rv"):
        _tensor(layer_matrices.get(key), torch.float32, (heads, dim, dim), "joint " + key)
    expected_ids = [sample["request_id"] for sample in head_samples[0]]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("OptR joint metric request IDs are duplicated")
    if any([sample["request_id"] for sample in head] != expected_ids for head in head_samples):
        raise ValueError("OptR joint metric TP request identity/order differs")
    cached = projected_outputs is not None or projected_teachers is not None
    if cached:
        for values in (projected_outputs, projected_teachers):
            if (not isinstance(values, list) or len(values) != heads
                    or any(not isinstance(head, list) or len(head) != len(expected_ids) for head in values)):
                raise ValueError("OptR cached projections require every head/request on both sides")
    projection_views = policy.get("validation_projection_views")
    if projection_views is not None and projection_views != ["fp32", "bf16_output", "bf16_rank_output"]:
        raise ValueError("OptR validation projection views differ")
    rows = {view: [] for view in (projection_views or ["fp32"])}
    with torch.no_grad():
        for request_index, request_id in enumerate(expected_ids):
            actual_sum, teacher_sum, identities = None, None, []
            rank_actual, rank_teacher = None, None
            first = head_samples[0][request_index]
            for head, samples in enumerate(head_samples):
                sample = samples[request_index]
                for key in ("positions", "contexts", "token_ids", "current_ends", "frame_ids"):
                    if key not in sample or key not in first or not torch.equal(sample[key], first[key]):
                        raise ValueError("OptR joint metric TP sample identities/positions differ")
                identities.append(_hash(sample.get("sample_sha256")))
                if cached:
                    actual, teacher = projected_outputs[head][request_index], projected_teachers[head][request_index]
                    shape = (sample["q"].shape[0], sample["wo"].shape[1])
                    _tensor(actual, torch.float32, shape, "cached projected output")
                    _tensor(teacher, torch.float32, shape, "cached projected teacher")
                else:
                    actual = project_output(replay_sample(layer_matrices["Rk"][head], layer_matrices["Rv"][head],
                                                          mean[head], sample, policy), sample)
                    teacher = project_output(sample["teacher"], sample)
                if actual_sum is None:
                    actual_sum, teacher_sum = actual.double(), teacher.double()
                else:
                    if actual.shape != actual_sum.shape or teacher.shape != teacher_sum.shape:
                        raise ValueError("OptR joint metric TP output geometry differs")
                    actual_sum += actual.double(); teacher_sum += teacher.double()
                if projection_views:
                    # Native unquantized RowParallelLinear publishes BF16 GEMM
                    # outputs before TP reduction. Reuse the existing FP32 W_O
                    # products: these casts add no attention replay/matmul.
                    rounded_actual, rounded_teacher = actual.bfloat16().double(), teacher.bfloat16().double()
                    if rank_actual is None:
                        rank_actual, rank_teacher = rounded_actual, rounded_teacher
                    else:
                        rank_actual += rounded_actual; rank_teacher += rounded_teacher
            views = {"fp32": (actual_sum, teacher_sum)}
            if projection_views:
                # Check both final-only and rank+final visible BF16 boundaries.
                # FP64 summation is an explicit CPU reference, NOT a claim to
                # reproduce the closed HCCL tree or fused GEMM accumulation.
                views.update(bf16_output=(actual_sum.bfloat16().double(), teacher_sum.bfloat16().double()),
                             bf16_rank_output=(rank_actual.bfloat16().double(), rank_teacher.bfloat16().double()))
            for view, (actual, teacher) in views.items():
                rows[view].append(_query_metrics(first, actual, teacher, request_id, _digest(identities),
                                                all_queries=projection_views is not None))
    result = _aggregate_requests(rows.pop("fp32"))
    if projection_views:
        result["projection_checks"] = {view: _aggregate_requests(values) for view, values in rows.items()}
    return result
