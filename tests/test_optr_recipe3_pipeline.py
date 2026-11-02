"""Archive #86/#97/#165/#166/#170/#174: authenticated recipe3 OptR pipeline.

Real CPU recipe3 pairs, exclusion plans, hashed tensors and joint replay.
Only the optimizer proposal and capture are synthetic; no NPU/model claim.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend.optr_artifacts import file_sha256
from oscar_ascend.optr_calibration import replay_sample
from oscar_ascend.optr_policy import optr_policy
from oscar_ascend.optr_recipe3_fit import CANDIDATE_STRATEGY, checkpoint_steps
from oscar_ascend.optr_traces import _frame_vectors, _workspace_geometry
from oscar_ascend.rotation_calibration import checkpoint_pair_artifact, fit_checkpoint_pair, publish_checkpoint_pair
from oscar_ascend.rotations import calibration_cache_identity, checkpoint_identity, load_active_artifact
from tools import prepare_optr_recipe3_base as base
from tools import prepare_optr_rotations as prepare
from tools.calibrate_model_rotations import prompt_ids_sha256, render_prompt_ids
from tools.optr_recipe3_sampling import build_recipe3_request_plan
from test_official_rotation_calibration import NAME, rank_states
from test_optr_recipe3_sampling import Tokenizer


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.delenv("OSCAR_ROTATION_ARTIFACT", raising=False)
    monkeypatch.setattr(base, "ROOT", tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    json_write(model / "config.json", {"num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "head_dim": 8, "hidden_size": 3, "num_attention_heads": 4, "num_key_value_heads": 2})
    (model / "model.safetensors").write_bytes(b"explicit test weight identity")
    messages = [[{"role": "user", "content": f"source-record-{i}"}] for i in range(36)]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps({"messages": row}) for row in messages) + "\n")
    config = {"model": str(model), "devices": [0, 1, 2, 3], "tensor_parallel_size": 2,
              "dtype": "bfloat16", "mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "bfloat16",
              "rotation_method": "qqt_sst", "rotation_variant": "perhead_center",
              "rotation_refinement": "optr", "optr_baseline": "recipe3",
              "experimental_official_math": True, "oscar_dequant_policy": "fp32_metadata",
              "rotation_calibration_tokens": 13, "rotation_calibration_prompts": str(source),
              "max_num_batched_tokens": 8, "max_model_len": 64, "cube_cores": 1,
              "sink_tokens": 1, "recent_tokens": 2,
              "optr_calibration": {"capture_lengths": {split: [12] for split in ("train", "validation", "test")},
                  "decode_tokens": 2, "query_samples_per_frame": 2, "steps": 1, "eval_every": 1}}
    names, dim, fingerprint = checkpoint_identity(model)
    assert names == [NAME] and dim == 8
    base_config = base.base_config(config, fingerprint)
    profile = calibration_cache_identity(base_config, fingerprint)
    tokenizer = Tokenizer()
    prompts = render_prompt_ids(tokenizer, messages, 13, 8)
    prompt_sha = prompt_ids_sha256(prompts)
    ranks = rank_states(fingerprint, prompt=prompt_sha, n=12)
    key, value = fit_checkpoint_pair(ranks, layer_names=names, head_dim=dim,
        model_path=model, model_fingerprint=fingerprint, global_q_heads=4, global_kv_heads=2,
        tp_size=2, prompt_sha256=prompt_sha, max_tokens=13, calibration_profile=profile, testing=True)
    artifact = checkpoint_pair_artifact(key, value, layer_names=names, head_dim=dim,
        model_fingerprint=fingerprint, calibration_profile=profile, testing=True)
    # Imported legacy pairs may lack raw capture manifests. Original token
    # hash/count still authenticate the source reconstruction.
    destination = Path(base_config["rotation_artifact"])
    publish_checkpoint_pair(key, value, destination.parent, artifact=artifact,
                            artifact_name=destination.name)
    rng = torch.Generator().manual_seed(918)
    candidate = {**artifact["layers"][NAME],
                 "Rk": torch.linalg.qr(torch.randn(2, dim, dim, generator=rng)).Q.contiguous(),
                 "Rv": torch.linalg.qr(torch.randn(2, dim, dim, generator=rng)).Q.contiguous()}
    return dict(config=config, base_config=base_config, baseline=artifact, candidate=candidate,
                messages=messages, tokenizer=tokenizer, source=source, fingerprint=fingerprint,
                output=tmp_path / "refined" / "refined.pt", calls=[])


def proposal(case, *, unchanged=False):
    def fit(rk, rv, mean, train, validation, *, policy):
        case["calls"].append("fit")
        assert torch.equal(mean, case["baseline"]["layers"][NAME]["k_mean"])
        assert torch.count_nonzero(mean) and not torch.equal(rk[0], rk[1])
        assert all(sample["split"] == "train" for head in train for sample in head)
        assert all(sample["split"] == "validation" for head in validation for sample in head)
        constants = case["baseline"]["layers"][NAME] if unchanged else case["candidate"]
        return {"Rk": constants["Rk"], "Rv": constants["Rv"],
                "decision": {"accepted": not unchanged, "policy": policy, "fitted_grouping": "layer_kv_head",
                    "deployment_candidate_selection": CANDIDATE_STRATEGY,
                    "candidate_steps": [0, *checkpoint_steps(policy)], "candidate_count": 1},
                "candidates": [{"id": "primary", "phase": "primary", "step": None,
                                "Rk": constants["Rk"], "Rv": constants["Rv"], "proxy_losses": None}]}
    return fit


def capture(case, *, fail_test=False, corrupt=None):
    def collect(config_path, directory):
        case["calls"].append("capture")
        config = json.loads(Path(config_path).read_text())
        metadata = config["optr_seed_capture"]
        assert metadata["baseline_artifact_sha256"] == file_sha256(case["base_config"]["rotation_artifact"])
        assert metadata["calibration_profile"]["recipe_version"] == 3
        assert metadata["capture_manifest"] is None
        policy = optr_policy(config)
        source_hash = file_sha256(case["source"])
        plan, provenance = build_recipe3_request_plan(case["tokenizer"], case["messages"], policy,
                                                       metadata, source_sha256=source_hash)
        for row in plan:
            row["generated_output_token_ids"] = [1000, 1001, 1002]
        workspace = _workspace_geometry(config)
        device = {"aicore_count": 20, "vectorcore_count": 40, "selected_cube_cores": 1}
        top = {"schema_version": 1, "test_only": True, "status": "native_optr_capture_complete",
               "output_dir": str(directory), "tp_size": 2, "head_dim": 8, "global_q_heads": 4,
               "global_kv_heads": 2, "model_fingerprint": case["fingerprint"], "layer_ids": [3],
               "request_ids": [row["request_id"] for row in plan],
               "prompt_sha256": prompt_ids_sha256([row["prompt_token_ids"] for row in plan]),
               "request_plan": [{key: value for key, value in row.items() if key != "prompt_token_ids"} for row in plan],
               "prompt_source": str(case["source"]), "prompt_source_sha256": source_hash,
               "recipe3_sampling": provenance, "calibration_profile": calibration_cache_identity(config, case["fingerprint"]),
               "optr_policy": policy, "decode_tokens": 2, "samples_per_frame": 2,
               "workspace_geometry": workspace, "rank_paths": [], "rank_sha256": {}}
        for rank in range(2):
            rng = torch.Generator().manual_seed(rank + 42)
            wo = torch.randn(2, 3, 8, generator=rng).bfloat16()
            wo_path = directory / f"rank{rank}-layer3-wo.pt"
            torch.save({"schema_version": 1, "test_only": True, "wo": wo, "rank": rank,
                        "layer_id": 3, "head_dim": 8, "global_q_head_start": 2 * rank}, wo_path)
            sequences = []
            for row in plan:
                length = row["length_bucket"]
                cursor, frames = 0, []
                for n in (8, 4, 1, 1):
                    frames.append({"context": cursor, "query_length": n, "sequence_length": cursor + n,
                        "input_tokens": n, "state": "PrefillNoCache" if cursor == 0 else
                            "ChunkedPrefill" if cursor < length else "DecodeOnly"})
                    cursor += n
                vectors = _frame_vectors({"frames": frames, "prompt_length": length}, policy, workspace, device, 2)
                m = len(vectors["positions"])
                sample = {"request_id": row["request_id"],
                    "q": torch.randn(m, 2, 8, generator=rng).bfloat16(),
                    "k": torch.randn(length + 2, 8, generator=rng).bfloat16(),
                    "v": torch.randn(length + 2, 8, generator=rng).bfloat16(),
                    "gate": torch.sigmoid(torch.randn(m, 2, 8, generator=rng)).bfloat16(),
                    "teacher": torch.zeros(m, 2, 8, dtype=torch.bfloat16), "wo": wo.float(), **vectors}
                constants = case["baseline"]["layers"][NAME] if fail_test and row["split"] == "test" else case["candidate"]
                sample["teacher"] = replay_sample(constants["Rk"][rank], constants["Rv"][rank],
                    constants["k_mean"][rank], sample, {**policy, "replay_partition": "captured_runtime_frames"}).bfloat16()
                payload = {**{key: sample[key] for key in ("q", "gate", "teacher", "positions", "contexts", "frame_ids")},
                    "k": sample["k"][:, None], "v": sample["v"][:, None], "frames": frames,
                    "token_ids": torch.tensor(row["prompt_token_ids"] + row["generated_output_token_ids"][:-1]),
                    "schema_version": 1, "test_only": True, "rank": rank, "global_kv_head": rank,
                    "layer_id": 3, "model_fingerprint": case["fingerprint"], "prompt_length": length,
                    **{key: row[key] for key in ("request_id", "split", "length_bucket", "source_record_sha256", "generated_output_token_ids")}}
                seq_path = directory / f"rank{rank}-{row['request_id']}.pt"
                torch.save(payload, seq_path)
                sequences.append({**{key: row[key] for key in ("request_id", "split", "length_bucket")},
                                  "path": str(seq_path), "sha256": file_sha256(seq_path)})
            rank_manifest = {key: top[key] for key in ("tp_size", "head_dim", "global_q_heads", "global_kv_heads", "model_fingerprint", "prompt_sha256")}
            rank_manifest.update(schema_version=1, rank=rank, test_only=True, status="complete", workspace_geometry=workspace,
                device_geometry=device, layers={"3": {"wo_path": str(wo_path), "wo_sha256": file_sha256(wo_path), "sequences": sequences}})
            rank_path = directory / f"rank{rank}-manifest.json"
            json_write(rank_path, rank_manifest)
            top["rank_paths"].append(str(rank_path))
            top["rank_sha256"][str(rank_path)] = file_sha256(rank_path)
        if corrupt == "exclusion":
            top["recipe3_sampling"]["excluded_record_sha256"].pop()
        elif corrupt == "seed":
            top["recipe3_sampling"]["seed_capture"]["baseline_artifact_sha256"] = "c" * 64
        return top
    return collect


def forbidden(*_args, **_kwargs):
    pytest.fail("authenticated cached work must not be recomputed")


def test_complete_pipeline_preserves_baseline_publishes_strict_fit_and_reuses_it(case):
    before = Path(case["base_config"]["rotation_artifact"]).read_bytes()
    result = prepare.prepare(case["config"], output=case["output"], testing=True,
                             capture_runner=capture(case), fitter=proposal(case), base_runner=forbidden)
    assert not result["reused"] and result["accepted_layers"] == [NAME]
    assert case["calls"] == ["capture", "fit"]
    assert Path(case["base_config"]["rotation_artifact"]).read_bytes() == before
    artifact = load_active_artifact(case["config"], path=case["output"], device="cpu", testing=True)
    assert artifact["refinement_certificate"]["schema_version"] == 3
    assert artifact["refinement_certificate"]["independent_test"]["status"] == "passed"
    assert torch.equal(artifact["layers"][NAME]["k_mean"], case["baseline"]["layers"][NAME]["k_mean"])
    assert torch.equal(artifact["layers"][NAME]["Rk"], case["candidate"]["Rk"])
    assert (case["output"].parent / ".optr.complete.json").is_file()
    again = prepare.prepare(case["config"], output=case["output"], testing=True,
        capture_runner=forbidden, fitter=forbidden, base_runner=forbidden)
    assert again["reused"]
    Path(case["base_config"]["rotation_artifact"]).write_bytes(before + b"changed")
    with pytest.raises(ValueError, match="baseline file changed"):
        prepare.prepare(case["config"], output=case["output"], testing=True)


def test_independent_test_failure_does_not_publish_or_reselect_on_resume(case):
    with pytest.raises(prepare.IndependentTestRegression, match="not published"):
        prepare.prepare(case["config"], output=case["output"], testing=True,
            capture_runner=capture(case, fail_test=True), fitter=proposal(case), base_runner=forbidden)
    assert not case["output"].exists()
    assert not (case["output"].parent / ".optr.complete.json").exists()
    failures = list(case["output"].parent.rglob("*independent-test-failed.json"))
    assert failures and case["calls"] == ["capture", "fit"]
    with pytest.raises(prepare.IndependentTestRegression, match="not published"):
        prepare.prepare(case["config"], output=case["output"], testing=True,
            capture_runner=forbidden, fitter=forbidden, base_runner=forbidden)
    assert not case["output"].exists()


def test_changed_fitter_refits_from_authenticated_capture_without_recapture(case, monkeypatch):
    from oscar_ascend import optr_checkpoint
    from tools import prepare_optr_recipe3
    prepare.prepare(case["config"], output=case["output"], testing=True,
                    capture_runner=capture(case), fitter=proposal(case), base_runner=forbidden)
    manifest = case["output"].parent / "native-traces" / "capture-manifest.json"
    trace_bytes = manifest.read_bytes()
    changed_digest = "c" * 64
    monkeypatch.setattr(optr_checkpoint, "fitting_implementation_digest", lambda: changed_digest)
    monkeypatch.setattr(prepare_optr_recipe3, "fitting_implementation_digest", lambda: changed_digest)
    result = prepare.prepare(case["config"], output=case["output"], testing=True,
        capture_runner=forbidden, fitter=proposal(case), base_runner=forbidden)
    assert not result["reused"] and result["accepted_layers"] == [NAME]
    assert case["calls"] == ["capture", "fit", "fit"]
    assert manifest.read_bytes() == trace_bytes
    artifact = load_active_artifact(case["config"], path=case["output"], device="cpu", testing=True)
    assert artifact["refinement_certificate"]["fitting_implementation_sha256"] == changed_digest


def test_no_strict_improvement_is_not_published_as_optr(case):
    with pytest.raises(ValueError, match="no strict validation improvement"):
        prepare.prepare(case["config"], output=case["output"], testing=True,
            capture_runner=capture(case), fitter=proposal(case, unchanged=True), base_runner=forbidden)
    assert not case["output"].exists()
    assert not (case["output"].parent / ".optr.complete.json").exists()


def test_interrupted_capture_uses_fresh_generation_and_preserves_baseline(case):
    before = Path(case["base_config"]["rotation_artifact"]).read_bytes()
    partial = []
    def interrupted(_config, directory):
        partial.append(directory)
        (directory / "partial-only.pt").write_bytes(b"incomplete CPU fixture")
        raise RuntimeError("injected capture interruption")
    with pytest.raises(RuntimeError, match="capture interruption"):
        prepare.prepare(case["config"], output=case["output"], testing=True,
                        capture_runner=interrupted, fitter=forbidden, base_runner=forbidden)
    assert not case["output"].exists()
    assert not (case["output"].parent / "native-traces" / "capture-manifest.json").exists()
    result = prepare.prepare(case["config"], output=case["output"], testing=True,
                             capture_runner=capture(case), fitter=proposal(case), base_runner=forbidden)
    assert result["accepted_layers"] == [NAME]
    assert (partial[0] / "partial-only.pt").read_bytes() == b"incomplete CPU fixture"
    manifest = json.loads((case["output"].parent / "native-traces" / "capture-manifest.json").read_text())
    assert Path(manifest["output_dir"]) != partial[0]
    assert Path(case["base_config"]["rotation_artifact"]).read_bytes() == before


@pytest.mark.parametrize("corrupt", ["exclusion", "seed"])
def test_prepare_rejects_trace_provenance_before_fitting(case, corrupt):
    with pytest.raises(ValueError, match="recipe3 trace"):
        prepare.prepare(case["config"], output=case["output"], testing=True,
            capture_runner=capture(case, corrupt=corrupt), fitter=forbidden, base_runner=forbidden)
    assert not case["output"].exists() and case["calls"] == ["capture"]

