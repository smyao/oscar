"""Archive #86/#97/#148/#170/#173: trace authenticity and TP/frame output SSE.

Tiny synthetic CPU payloads are explicitly test_only; they do not represent
native NPU model traces. Tests exercise corruption and the cross-head terms
that a sum of per-head proxy losses cannot measure.
"""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend.integration.optr_capture import LayerTrace
from oscar_ascend.optr_artifacts import file_sha256, nonworse
from oscar_ascend.optr_policy import optr_policy
from oscar_ascend import optr_traces as traces
from oscar_ascend.rotations import calibration_cache_identity
from tools.calibrate_model_rotations import prompt_ids_sha256


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def json_save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_bundle(tmp_path, tp=2, dim=4, fingerprint="toy", steps=None):
    model = tmp_path / "model"; model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_attention_heads": tp * 2, "num_key_value_heads": tp,
                                                   "head_dim": dim, "hidden_size": 3,
                                                   "num_hidden_layers": 4, "full_attention_interval": 4}))
    if fingerprint is None:
        from oscar_ascend.rotations import checkpoint_identity
        fingerprint = checkpoint_identity(model)[2]
    config = {"model": str(model), "tensor_parallel_size": tp, "max_num_batched_tokens": 4,
              "rotation_refinement": "optr", "rotation_variant": "perhead_center",
              "experimental_official_math": True, "oscar_dequant_policy": "fp32_metadata",
              "optr_calibration": {"capture_lengths": {"train": [4], "validation": [5], "test": [5]},
                                   "decode_tokens": 2, "query_samples_per_frame": 2}}
    if steps is not None:
        config["optr_calibration"].update(steps=steps, eval_every=1)
    policy = optr_policy(config)
    root = tmp_path / "capture"; root.mkdir()
    plan, prompt_lists = [], []
    for index, split in enumerate(("train", "validation", "test")):
        length = policy["capture_lengths"][split][0]
        prompt = list(range(10 * index, 10 * index + length))
        prompt_lists.append(prompt)
        plan.append({"request_id": f"{split}_{length}_0", "split": split, "length_bucket": length,
                     "prompt_sha256": prompt_ids_sha256([prompt]), "source_record_sha256": [digest(split)],
                     "generated_output_token_ids": [60 + index, 70 + index, 80 + index]})
    workspace = traces._workspace_geometry(config)
    top = {"schema_version": 1, "test_only": True, "status": "native_optr_capture_complete", "output_dir": str(root),
           "tp_size": tp, "head_dim": dim, "global_q_heads": tp * 2, "global_kv_heads": tp,
           "model_fingerprint": fingerprint, "layer_ids": [3], "request_ids": [row["request_id"] for row in plan],
           "request_plan": plan, "prompt_sha256": prompt_ids_sha256(prompt_lists), "prompt_source_sha256": digest("source"),
           "optr_policy": policy, "calibration_profile": calibration_cache_identity(config, fingerprint),
           "decode_tokens": 2, "samples_per_frame": 2, "workspace_geometry": workspace,
           "rank_paths": [], "rank_sha256": {}}
    top["prompt_source_sha256"] = top["calibration_profile"]["base_calibration_profile"]["prompt_source"]["sha256"]
    for rank in range(tp):
        wo_path = root / f"rank{rank}-layer3-wo.pt"
        torch.save({"schema_version": 1, "wo": torch.full((2, 3, dim), .1 * (rank + 1), dtype=torch.bfloat16),
                    "rank": rank, "layer_id": 3, "head_dim": dim, "global_q_head_start": rank * 2, "test_only": True}, wo_path)
        entries = []
        for row, prompt in zip(plan, prompt_lists):
            length = row["length_bucket"]
            trace = LayerTrace(dim, 2, 1, 2, length + 2)
            context = 0
            extents = [4] + ([1] if length == 5 else []) + [1, 1]
            for n in extents:
                from types import SimpleNamespace as NS
                state = "PrefillNoCache" if context == 0 else "DecodeOnly" if n == 1 else "ChunkedPrefill"
                meta = NS(num_actual_tokens=n, actual_seq_lengths_q=[n, 4] if n < 4 else [n],
                          seq_lens_list=[context + n, 1] if n < 4 else [context + n], attn_state=NS(name=state))
                q = torch.full((4, 2, dim), .1 + rank / 10, dtype=torch.bfloat16)
                k, v = q[:, :1].clone(), q[:, :1].clone()
                gate = torch.full_like(q, .5)
                pending = trace.prepare(q, k, v, gate, torch.arange(context, context + 4), meta)
                trace.complete(pending, q)
                context += n
            payload = trace.payload(prompt + row["generated_output_token_ids"][:-1])
            payload.update({**row, "schema_version": 1, "test_only": True, "rank": rank,
                            "global_kv_head": rank, "layer_id": 3, "model_fingerprint": fingerprint, "prompt_length": length})
            seq_path = root / f"rank{rank}-layer3-{row['request_id']}.pt"
            torch.save(payload, seq_path)
            entries.append({**{k: row[k] for k in ("request_id", "split", "length_bucket")},
                            "path": str(seq_path), "sha256": file_sha256(seq_path)})
        rank_data = {k: top[k] for k in ("tp_size", "head_dim", "global_q_heads", "global_kv_heads", "model_fingerprint", "prompt_sha256")}
        rank_data.update(schema_version=1, rank=rank, test_only=True, status="complete", workspace_geometry=workspace,
                         device_geometry={"aicore_count": 20, "vectorcore_count": 40, "selected_cube_cores": 20},
                         layers={"3": {"wo_path": str(wo_path), "wo_sha256": file_sha256(wo_path), "sequences": entries}})
        rank_path = root / f"rank{rank}-manifest.json"
        json_save(rank_path, rank_data)
        top["rank_paths"].append(str(rank_path)); top["rank_sha256"][str(rank_path)] = file_sha256(rank_path)
    path = root / "capture-manifest.json"; json_save(path, top)
    return path, config, top


def load(path, config, testing=True):
    return traces.load_trace_manifest(path, config, ["model.layers.3.self_attn.attn"],
                                      json.loads((Path(config["model"]) / "config.json").read_text())["head_dim"],
                                      json.loads(Path(path).read_text())["model_fingerprint"], testing=testing)


def mutate_payload(path, top, rank, key, mutate):
    rank_path = Path(top["rank_paths"][rank]); rank_data = json.loads(rank_path.read_text())
    layer = rank_data["layers"]["3"]
    if key == "wo":
        file = Path(layer["wo_path"])
        payload = torch.load(file, weights_only=True); mutate(payload); torch.save(payload, file)
        layer["wo_sha256"] = file_sha256(file)
    else:
        row = layer["sequences"][key]
        file = Path(row["path"])
        payload = torch.load(file, weights_only=True); mutate(payload); torch.save(payload, file)
        row["sha256"] = file_sha256(file)
    json_save(rank_path, rank_data)
    top["rank_sha256"][str(rank_path)] = file_sha256(rank_path)
    json_save(path, top)


def test_loader_streams_one_layer_and_reconstructs_actual_frame_partitions(tmp_path):
    path, config, _ = build_bundle(tmp_path)
    manifest = load(path, config)
    samples = traces.load_layer_samples(manifest, 3)
    assert set(samples) == {"train", "validation", "test"}
    assert len(samples["validation"]) == 2 and len(samples["validation"][0]) == 1
    sample = samples["validation"][0][0]
    assert sample["k"].shape == (7, 4) and sample["wo"].shape == (2, 3, 4)
    assert sample["wo"].dtype == torch.float32
    assert sample["positions"].tolist() == [0, 3, 4, 5, 6]
    assert sample["contexts"].tolist() == [0, 0, 4, 5, 6]
    assert sample["current_ends"].tolist() == [4, 4, 5, 6, 7]
    # Real runtime arena for capacity=4, padded inputs=4 limits every S to 1.
    assert sample["history_splits"].tolist() == [1, 1, 1, 1, 1]
    assert len(sample["sample_sha256"]) == 64
    # Canonical manifest may reside above an owned generation directory.
    canonical = path.parent.parent / "capture-manifest.json"; canonical.write_bytes(path.read_bytes())
    assert load(canonical, config)["_root"] == str(path.parent)


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5", "v6"])
def test_legacy_capture_reuse_requires_exact_policy_and_sampling_coverage(tmp_path, version):
    from oscar_ascend.optr_policy import legacy_capture_profile
    path, config, top = build_bundle(tmp_path)
    historical = legacy_capture_profile(top["calibration_profile"], config, version)
    top.update(calibration_profile=historical, optr_policy=historical["refinement_policy"])
    json_save(path, top)
    samples = traces.load_layer_samples(load(path, config), 3)
    assert samples["train"][0][0]["length_bucket"] == 4
    # All historical captures already include each forward's terminal row;
    # the stronger v4 certificate needs no new native sampling policy.
    for split in samples.values():
        for head in split:
            for sample in head:
                for frame_id in torch.unique(sample["frame_ids"]):
                    rows = sample["frame_ids"] == frame_id
                    assert sample["positions"][rows][-1] == sample["current_ends"][rows][0] - 1
    # Adding a training bucket requires another actual native request. Old
    # trace bytes must not be relabelled as the expanded coverage.
    expanded = {**config, "optr_calibration": {**config["optr_calibration"],
                "capture_lengths": {"train": [4, 5], "validation": [5], "test": [5]}}}
    with pytest.raises(ValueError, match="policy/calibration profile"):
        load(path, expanded)
    top["optr_policy"] = optr_policy(config)
    json_save(path, top)
    with pytest.raises(ValueError, match="policy/calibration profile"):
        load(path, config)


def test_old_raw_trace_must_keep_its_real_recipe3_seed_identity(tmp_path):
    from oscar_ascend.optr_policy import legacy_capture_profile
    path, config, top = build_bundle(tmp_path)
    current = top["calibration_profile"]
    historical = legacy_capture_profile(current, config, "v4")
    assert historical["base_calibration_profile"]["recipe_version"] == 3
    historical["base_calibration_profile"] = current["base_calibration_profile"]
    top.update(calibration_profile=historical, optr_policy=historical["refinement_policy"])
    json_save(path, top)
    with pytest.raises(ValueError, match="policy/calibration profile"):
        load(path, config)


@pytest.mark.parametrize("version", ["v5", "v6"])
def test_shared_raw_trace_retains_its_old_seed_identity(tmp_path, version):
    from oscar_ascend.optr_policy import legacy_capture_profile
    from oscar_ascend.optr_oscar_seed import legacy_seed_profile
    path, config, top = build_bundle(tmp_path)
    current = top["calibration_profile"]
    historical = legacy_capture_profile(current, config, version)
    assert historical["base_calibration_profile"] == legacy_seed_profile(current["base_calibration_profile"])
    assert historical["base_calibration_profile"] != current["base_calibration_profile"]
    assert historical["base_calibration_profile"]["recipe_version"] == 6
    top.update(calibration_profile=historical, optr_policy=historical["refinement_policy"])
    json_save(path, top)
    assert traces.load_layer_samples(load(path, config), 3)["train"][0]
    historical["base_calibration_profile"] = legacy_capture_profile(current, config, "v4")["base_calibration_profile"]
    json_save(path, top)
    with pytest.raises(ValueError, match="policy/calibration profile"):
        load(path, config)
    historical["base_calibration_profile"] = current["base_calibration_profile"]
    json_save(path, top)
    with pytest.raises(ValueError, match="policy/calibration profile"):
        load(path, config)


@pytest.mark.parametrize("corruption", ["source_overlap", "policy", "profile", "rank_hash", "rank_order", "escape", "test_only"])
def test_manifest_corruption_and_test_only_fail_closed(tmp_path, corruption):
    path, config, top = build_bundle(tmp_path)
    if corruption == "source_overlap":
        top["request_plan"][1]["source_record_sha256"] = top["request_plan"][0]["source_record_sha256"]
    elif corruption == "policy":
        top["optr_policy"]["decode_tokens"] += 1
    elif corruption == "profile":
        top["calibration_profile"]["model_fingerprint"] = "wrong"
    elif corruption == "rank_hash":
        top["rank_sha256"][top["rank_paths"][0]] = "0" * 64
    elif corruption == "rank_order":
        top["rank_paths"].reverse()
    elif corruption == "escape":
        top["output_dir"] = str(tmp_path.parent)
    json_save(path, top)
    with pytest.raises(ValueError):
        load(path, config, testing=corruption != "test_only")


@pytest.mark.parametrize("corruption", ["k_nan", "q_dtype", "gate_range", "position", "context", "tokens", "generated", "source", "rank", "wo_head", "wo_nan", "wo_schema", "file_hash"])
def test_tensor_identity_corruption_rejected_even_after_manifest_is_rehashed(tmp_path, corruption):
    path, config, top = build_bundle(tmp_path)
    def alter(payload):
        if corruption == "k_nan": payload["k"][0, 0, 0] = float("nan")
        elif corruption == "q_dtype": payload["q"] = payload["q"].float()
        elif corruption == "gate_range": payload["gate"][0, 0, 0] = 2
        elif corruption == "position": payload["positions"][1] -= 1
        elif corruption == "context": payload["contexts"][-1] -= 1
        elif corruption == "tokens": payload["token_ids"][0] += 1
        elif corruption == "generated": payload["generated_output_token_ids"][-1] += 1
        elif corruption == "source": payload["source_record_sha256"] = [digest("wrong")]
        elif corruption == "rank": payload["rank"] = 99
        elif corruption == "wo_head": payload["global_q_head_start"] = 99
        elif corruption == "wo_nan": payload["wo"][0, 0, 0] = float("nan")
        elif corruption == "wo_schema": payload["schema_version"] = 99
    if corruption == "file_hash":
        rank = json.loads(Path(top["rank_paths"][0]).read_text())
        file = Path(rank["layers"]["3"]["sequences"][0]["path"])
        with file.open("ab") as stream: stream.write(b"modified")
    else:
        mutate_payload(path, top, 1, "wo" if corruption.startswith("wo_") else 0, alter)
    manifest = load(path, config)
    with pytest.raises(ValueError):
        traces.load_layer_samples(manifest, 3)


def test_cross_rank_frame_alignment_and_measured_core_validation(tmp_path):
    path, config, top = build_bundle(tmp_path)
    # Change padding shape without changing tokens. Individually valid frames
    # may still not be combined as one shared TP model request.
    mutate_payload(path, top, 1, 1, lambda payload: payload["frames"][-1].update(input_tokens=3))
    with pytest.raises(ValueError, match="TP ranks"):
        traces.load_layer_samples(load(path, config), 3)
    rank_path = Path(top["rank_paths"][0]); rank = json.loads(rank_path.read_text())
    rank["device_geometry"]["vectorcore_count"] = 20
    json_save(rank_path, rank); top["rank_sha256"][str(rank_path)] = file_sha256(rank_path); json_save(path, top)
    with pytest.raises(ValueError, match="AIC/AIV"):
        load(path, config)


@pytest.mark.parametrize("marker", ["rank", "wo", "sequence"])
def test_nested_test_only_payload_cannot_be_relabelled_by_unmarked_top_manifest(tmp_path, marker):
    path, config, top = build_bundle(tmp_path, tp=4, dim=256)
    top["test_only"] = False
    for rank in range(4):
        rank_path = Path(top["rank_paths"][rank]); rank_data = json.loads(rank_path.read_text())
        rank_data["test_only"] = marker == "rank" and rank == 0
        layer = rank_data["layers"]["3"]
        wo_path = Path(layer["wo_path"]); wo = torch.load(wo_path, weights_only=True)
        wo["test_only"] = marker == "wo" and rank == 0
        torch.save(wo, wo_path); layer["wo_sha256"] = file_sha256(wo_path)
        for i, entry in enumerate(layer["sequences"]):
            seq_path = Path(entry["path"]); seq = torch.load(seq_path, weights_only=True)
            seq["test_only"] = marker == "sequence" and rank == 0 and i == 0
            torch.save(seq, seq_path); entry["sha256"] = file_sha256(seq_path)
        json_save(rank_path, rank_data); top["rank_sha256"][str(rank_path)] = file_sha256(rank_path)
    json_save(path, top)
    with pytest.raises(ValueError, match="test-only"):
        traces.load_layer_samples(load(path, config, testing=False), 3)


def test_frame_split_formula_matches_runtime_geometry(tmp_path):
    path, config, top = build_bundle(tmp_path)
    manifest = load(path, config)
    samples = traces.load_layer_samples(manifest, 3)
    sample = samples["validation"][0][0]
    rank = manifest["_rank_manifests"][0]
    payload = torch.load(rank["layers"]["3"]["sequences"][1]["path"], weights_only=True)
    workspace = {"token_capacity": 16384, "history_token_capacity": 16384, "attention_splits": 1, "requested_cube_cores": None}
    from oscar_ascend.runtime import WorkspaceGeometry
    runtime = WorkspaceGeometry(16384, 2, 1, 64, 1, 20)
    vectors = traces._frame_vectors(payload, manifest["optr_policy"], workspace, rank["device_geometry"], 2)
    expected = [runtime.splits_for_tokens(payload["frames"][int(frame)]["input_tokens"]) for frame in sample["frame_ids"]]
    assert vectors["history_splits"].tolist() == expected == [20] * 5


def metric_samples():
    heads = []
    for head in range(2):
        heads.append([{"request_id": "r", "q": torch.zeros(1, 1, 2, dtype=torch.bfloat16),
                       "k": torch.zeros(1, 2, dtype=torch.bfloat16), "v": torch.zeros(1, 2, dtype=torch.bfloat16),
                       "gate": torch.ones(1, 1, 2, dtype=torch.bfloat16),
                       "teacher": torch.tensor([[[1., 0.]]], dtype=torch.bfloat16),
                       "wo": torch.tensor([[[1., 0.]]]), "positions": torch.tensor([0]),
                       "contexts": torch.tensor([0]), "current_ends": torch.tensor([1]),
                       "frame_ids": torch.tensor([0]), "token_ids": torch.tensor([5]),
                       "history_splits": torch.tensor([1]), "sample_sha256": digest(str(head)), "rank": head}])
    constants = {"Rk": torch.eye(2).expand(2, 2, 2).clone(), "Rv": torch.eye(2).expand(2, 2, 2).clone(),
                 "k_mean": torch.zeros(2, 2)}
    return constants, heads


def test_joint_output_metric_keeps_cross_rank_cancellation_and_cross_terms(monkeypatch):
    from oscar_ascend import optr_calibration
    constants, heads = metric_samples()
    def cancel(_rk, _rv, _mean, sample, _policy):
        return torch.tensor([[[2., 0.]]]) if sample["rank"] == 0 else torch.tensor([[[0., 0.]]])
    monkeypatch.setattr(optr_calibration, "replay_sample", cancel)
    metric = traces.aggregate_layer_error(constants, heads, {})
    assert metric["squared_error"] == 0 and metric["teacher_energy"] == 4
    assert metric["requests"][0]["frames"] == [{"frame_id": 0, "context": 0,
        "current_end": 1, "query_positions": [0], "squared_error": 0.,
        "teacher_energy": 4., "normalized_error": 0., "terminal_query": {
            "position": 0, "squared_error": 0., "teacher_energy": 4., "normalized_error": 0.}}]
    assert nonworse(metric, metric)
    monkeypatch.setattr(optr_calibration, "replay_sample", lambda *_: torch.tensor([[[2., 0.]]]))
    aligned = traces.aggregate_layer_error(constants, heads, {})
    assert aligned["squared_error"] == 4  # Sum of per-rank SSE would be only 2.
    assert not nonworse(metric, aligned)
    assert aligned["requests"][0]["sample_sha256"] == metric["requests"][0]["sample_sha256"]
    heads[1][0]["token_ids"][0] = 6
    with pytest.raises(ValueError, match="identities"):
        traces.aggregate_layer_error(constants, heads, {})


def test_joint_metric_exposes_decode_regression_hidden_by_lower_request_error():
    """#170/#173: real CPU replay, not a mocked metric or a model-quality claim.

    An orthogonal proposal can reduce one request's aggregate while making
    its final generated query worse. Every decode is its own captured frame,
    so preserve its error for the deployment selector instead of averaging it
    away with the other generated queries.
    """
    rng = torch.Generator().manual_seed(75)
    dim, group, length = 8, 2, 40
    positions = torch.tensor([0, 4, 8, 13, 17, 22, 26, 31, *range(32, 40)])
    contexts = torch.tensor([0] * 8 + list(range(32, 40)))
    ends = torch.tensor([32] * 8 + list(range(33, 41)))
    k = torch.randn(length, dim, generator=rng).bfloat16()
    v = torch.randn(length, dim, generator=rng).bfloat16()
    q = torch.randn(16, group, dim, generator=rng).bfloat16()
    scores = torch.einsum("mgd,ld->mgl", q.float(), k.float()) / dim ** .5
    scores.masked_fill_(torch.arange(length)[None, None] > positions[:, None, None], -torch.inf)
    teacher = torch.einsum("mgl,ld->mgd", scores.softmax(-1), v.float()).bfloat16()
    sample = {"request_id": "frame-regression", "q": q, "k": k, "v": v,
              "positions": positions, "contexts": contexts, "current_ends": ends,
              "gate": torch.sigmoid(torch.randn(16, group, dim, generator=rng)).bfloat16(),
              "wo": torch.randn(group, 12, dim, generator=rng), "teacher": teacher,
              "token_ids": torch.arange(length), "frame_ids": torch.tensor([0] * 8 + list(range(1, 9))),
              "sample_sha256": digest("synthetic-frame-counterexample")}
    baseline = {"Rk": torch.eye(dim)[None], "Rv": torch.eye(dim)[None], "k_mean": torch.zeros(1, dim)}
    for _ in range(18):
        correction = torch.randn(dim, dim, generator=rng) * .03
        rk = torch.matrix_exp(correction - correction.T)
        correction = torch.randn(dim, dim, generator=rng) * .03
        rv = torch.matrix_exp(correction - correction.T)
    candidate = {**baseline, "Rk": rk[None], "Rv": rv[None]}
    policy = {"sink_tokens": 2, "recent_tokens": 4}
    before = traces.aggregate_layer_error(baseline, [[sample]], policy)
    after = traces.aggregate_layer_error(candidate, [[sample]], policy)
    assert after["normalized_error"] < before["normalized_error"]
    previous = before["requests"][0]["frames"][-1]
    current = after["requests"][0]["frames"][-1]
    assert current["normalized_error"] > previous["normalized_error"] * 1.1
    assert current["query_positions"] == [39] and current["context"] == 39 and current["current_end"] == 40
    for metric in (before, after):
        request = metric["requests"][0]
        assert len(request["frames"]) == 9
        assert request["frames"][0]["query_positions"] == positions[:8].tolist()
        for key in ("squared_error", "teacher_energy"):
            assert request[key] == sum(frame[key] for frame in request["frames"]) == metric[key]


def test_joint_metric_rejects_unordered_frame_ids():
    constants, heads = metric_samples()
    for head in heads:
        head[0]["frame_ids"] = torch.tensor([1])
    with pytest.raises(ValueError, match="ordered contiguous"):
        traces.aggregate_layer_error(constants, heads, {})


def test_cached_projections_preserve_joint_frame_metrics_without_another_replay(monkeypatch):
    from oscar_ascend import optr_calibration
    constants, heads = metric_samples()
    expected = traces.aggregate_layer_error(constants, heads, {})
    actual, teacher = [], []
    for head, samples in enumerate(heads):
        actual.append([optr_calibration.project_output(optr_calibration.replay_sample(
            constants["Rk"][head], constants["Rv"][head], constants["k_mean"][head], sample, {}), sample)
            for sample in samples])
        teacher.append([optr_calibration.project_output(sample["teacher"], sample) for sample in samples])
    def forbidden(*_args, **_kwargs):
        pytest.fail("cached subset aggregation must not replay or project again")
    monkeypatch.setattr(optr_calibration, "replay_sample", forbidden)
    monkeypatch.setattr(optr_calibration, "project_output", forbidden)
    assert traces.aggregate_layer_error(constants, heads, {}, projected_outputs=actual,
                                         projected_teachers=teacher) == expected


def test_cached_terminal_prefill_metric_blocks_regression_hidden_in_frame_total(monkeypatch):
    """#170/#173: final prefill output selects the next token, not its frame mean."""
    from oscar_ascend import optr_calibration
    constants, heads = metric_samples()
    for head in heads:
        sample = head[0]
        for key in ("q", "k", "v", "gate", "teacher"):
            sample[key] = sample[key].repeat(2, *([1] * (sample[key].ndim - 1)))
        sample.update(positions=torch.tensor([0, 1]), contexts=torch.tensor([0, 0]),
                      current_ends=torch.tensor([2, 2]), frame_ids=torch.tensor([0, 0]),
                      token_ids=torch.tensor([5, 6]), history_splits=torch.tensor([1, 1]))
    def forbidden(*_args, **_kwargs):
        pytest.fail("terminal query evidence must reuse cached projections")
    monkeypatch.setattr(optr_calibration, "replay_sample", forbidden)
    monkeypatch.setattr(optr_calibration, "project_output", forbidden)
    teacher = [[torch.ones(2, 1)] for _ in heads]
    baseline = [[torch.tensor([[4.], [1.]])], [torch.ones(2, 1)]]
    selected = [[torch.ones(2, 1)], [torch.tensor([[1.], [2.]])]]
    before = traces.aggregate_layer_error(constants, heads, {}, projected_outputs=baseline,
                                         projected_teachers=teacher)
    after = traces.aggregate_layer_error(constants, heads, {}, projected_outputs=selected,
                                        projected_teachers=teacher)
    assert before["squared_error"] == 9. and after["squared_error"] == 1.
    assert before["requests"][0]["frames"][0]["terminal_query"]["squared_error"] == 0.
    terminal = after["requests"][0]["frames"][0]["terminal_query"]
    assert terminal == {"position": 1, "squared_error": 1., "teacher_energy": 4., "normalized_error": .25}
    assert not nonworse(before, after, require_terminal=True)
    # Historical frame-only acceptance would allow exactly these outputs.
    for value in (before, after):
        value["requests"][0]["frames"][0].pop("terminal_query")
    assert nonworse(before, after, require_frames=True)


def test_joint_metric_requires_actual_terminal_sample_for_every_frame():
    constants, heads = metric_samples()
    for head in heads:
        head[0]["current_ends"] = torch.tensor([2])
    cached = [[torch.ones(1, 1)] for _ in heads]
    with pytest.raises(ValueError, match="frame extents/positions"):
        traces.aggregate_layer_error(constants, heads, {}, projected_outputs=cached,
                                     projected_teachers=cached)


@pytest.mark.parametrize("fault", ["missing_teacher", "heads", "requests", "shape", "dtype", "nonfinite"])
def test_cached_projection_geometry_and_finiteness_are_checked(fault):
    constants, heads = metric_samples()
    actual, teacher = [[torch.ones(1, 1)] for _ in heads], [[torch.ones(1, 1)] for _ in heads]
    if fault == "missing_teacher": teacher = None
    elif fault == "heads": actual.pop()
    elif fault == "requests": actual[0].append(torch.ones(1, 1))
    elif fault == "shape": actual[0][0] = torch.ones(1, 2)
    elif fault == "dtype": teacher[0][0] = teacher[0][0].double()
    elif fault == "nonfinite": actual[0][0].fill_(float("nan"))
    with pytest.raises(ValueError, match="cached"):
        traces.aggregate_layer_error(constants, heads, {}, projected_outputs=actual,
                                     projected_teachers=teacher)


@pytest.mark.parametrize("real_fitter", [False, True])
@pytest.mark.parametrize("interrupt_publication", [False, True])
def test_full_prepare_load_publish_and_cache_reuse_with_authenticated_traces(tmp_path, monkeypatch, real_fitter, interrupt_publication):
    from tools import prepare_optr_rotations as prepare
    from oscar_ascend.optr_oscar_seed import SEED_RECIPE, SOURCE_COMMIT, OBJECTIVE as SEED_OBJECTIVE
    from oscar_ascend.rotations import checkpoint_identity, load_active_artifact, prepare_device_rotations
    path, config, top = build_bundle(tmp_path, dim=64, fingerprint=None, steps=1)
    names, dim, fingerprint = checkpoint_identity(config["model"])
    seed_path = tmp_path / "seed.pt"
    base_config = {k: v for k, v in config.items() if k not in ("rotation_refinement", "optr_calibration")}
    base_config["rotation_artifact"] = str(seed_path)
    base_config["rotation_seed"] = SEED_RECIPE
    baseline = {"format_version": 2, "source_grouping": "layer_kv_head", "rotation_variant": "perhead_center",
                "global_kv_heads": 2, "head_dim": dim, "model_fingerprint": fingerprint,
                "pr_fingerprint": SOURCE_COMMIT, "objective": SEED_OBJECTIVE, "calibrated": True,
                "accumulation_dtype": "float32", "test_only": True,
                "calibration_profile": calibration_cache_identity(base_config, fingerprint),
                "layers": {names[0]: {"layer_id": 3, "Rk": -torch.eye(dim).expand(2, dim, dim).clone(),
                                       "Rv": -torch.eye(dim).expand(2, dim, dim).clone(), "k_mean": torch.zeros(2, dim)}}}
    seed_calls, captures, fits = [], [], []
    monkeypatch.setattr(prepare, "_base_config", lambda *_: base_config)
    def seed(*_a, **_kw):
        seed_calls.append(True)
        if not seed_path.exists():
            torch.save(baseline, seed_path)
        return baseline
    monkeypatch.setattr(prepare, "_prepare_base", seed)
    def native_capture(config_path, destination):
        captures.append(True)
        assert json.loads(config_path.read_text()) == config
        # Relocate the synthetic owned capture generation; tensor bytes and
        # token/teacher identity remain unchanged, only manifest paths move.
        old = path.parent
        destination.rmdir()
        old.rename(destination)
        rank_paths, hashes = [], {}
        for rank in range(2):
            rank_path = destination / f"rank{rank}-manifest.json"
            rank_data = json.loads(rank_path.read_text())
            layer = rank_data["layers"]["3"]
            layer["wo_path"] = str(destination / Path(layer["wo_path"]).name)
            for row in layer["sequences"]:
                row["path"] = str(destination / Path(row["path"]).name)
            json_save(rank_path, rank_data)
            rank_paths.append(str(rank_path)); hashes[str(rank_path)] = file_sha256(rank_path)
        top.update(output_dir=str(destination), rank_paths=rank_paths, rank_sha256=hashes)
        json_save(destination / "capture-manifest.json", top)
        return top
    def retain(rk, rv, mean, train, validation, *, policy):
        fits.append(True)
        assert len(train) == len(validation) == rk.shape[0]
        assert train[0][0]["wo"].dtype == torch.float32
        assert torch.equal(rk[0], rk[1]) and torch.equal(rv[0], rv[1])
        assert policy["fitted_grouping"] == "layer"
        return rk.clone(), rv.clone(), {"status": "explicit_test_baseline"}
    def forbidden(*_a, **_kw):
        pytest.fail("authenticated completed work must not resample or refit")
    output = tmp_path / "refined" / "calibrated.pt"
    if interrupt_publication:
        def interrupted(*_a, **_kw):
            raise RuntimeError("interrupted before final publication")
        with monkeypatch.context() as patch:
            patch.setattr(prepare, "publish_artifact", interrupted)
            with pytest.raises(RuntimeError, match="before final publication"):
                prepare.prepare(config, output=output, capture_runner=native_capture,
                                testing=True, fitter=None if real_fitter else retain)
        assert not output.exists()
        # Completed layers include validated joint/test evidence. A restarted
        # process validates that checkpoint and publishes without replaying it.
        with monkeypatch.context() as patch:
            patch.setattr(traces, "load_layer_samples", forbidden)
            result = prepare.prepare(config, output=output, capture_runner=forbidden,
                                     testing=True, fitter=forbidden)
    else:
        result = prepare.prepare(config, output=output, capture_runner=native_capture,
                                 testing=True, fitter=None if real_fitter else retain)
    assert not result["reused"] and result["recipe_version"] == 5
    assert len(seed_calls) == 1 + interrupt_publication and len(captures) == 1
    if not real_fitter:
        assert len(fits) == 1 and not result["accepted_layers"]
    artifact = load_active_artifact(config, path=output, device="cpu", testing=True)
    assert artifact["test_only"] and artifact["objective"] == "optr_post_gate_output_v1"
    assert artifact["refinement_certificate"]["performance"] == "not_measured"
    rotations = prepare_device_rotations(artifact, device="cpu", testing=True, tp_rank=1, tp_size=2)
    assert rotations[names[0]].key_transposed.shape == (dim, dim)
    with pytest.raises(ValueError, match="test"):
        load_active_artifact(config, path=output, device="cpu", testing=False)
    monkeypatch.setattr(prepare, "_prepare_base", forbidden)
    second = prepare.prepare(config, output=output, capture_runner=forbidden, testing=True, fitter=forbidden)
    assert second["reused"] and second["source"] == "optr_authenticated_cache"
