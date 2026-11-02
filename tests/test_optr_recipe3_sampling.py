"""Archive #86/#97/#165/#170/#174: seed exclusion and independent bucket inputs.

Synthetic tokenizers prove source/identity semantics only. They are not NPU
captures or evidence that local layer improvement raises model accuracy.
"""
import copy
import hashlib
import json

import pytest

from tools.calibrate_model_rotations import load_messages, prompt_ids_sha256, render_prompt_ids
from tools.calibration_data import calibration_prompt_source
from tools.optr_recipe3_sampling import (SOURCE_PLAN, _digest, build_recipe3_request_plan,
                                        validate_recipe3_sampling, validate_recipe3_trace_manifest)


class Tokenizer:
    def __init__(self, count=4):
        self.count = count

    def apply_chat_template(self, message, **_):
        first = int(_digest(message)[:8], 16)
        return [first + i for i in range(self.count)]


def fixture(tmp_path, *, messages=None, budget=13, tokenizer=None, lengths=None):
    messages = messages or [[{"role": "user", "content": f"record-{i}"}] for i in range(36)]
    tokenizer = tokenizer or Tokenizer()
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps({"messages": row}) for row in messages) + "\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    prompts = render_prompt_ids(tokenizer, messages, budget, 16384)
    profile = {"recipe_version": 3, "rotation_variant": "perhead_center", "model_fingerprint": "model",
               "prompt_source": {"kind": "explicit", "sha256": source_sha}, "max_tokens": budget}
    seed = {"schema_version": 1, "model_fingerprint": "model", "calibration_profile": profile,
            "prompt_source": profile["prompt_source"], "prompt_sha256": prompt_ids_sha256(prompts),
            "token_count": sum(map(len, prompts)), "capture_manifest_available": False,
            "capture_manifest": None, "baseline_artifact_sha256": "a" * 64}
    policy = {"source_plan": SOURCE_PLAN, "sampling_chunk_tokens": 16384,
              "capture_lengths": lengths or {"train": [8, 13, 21], "validation": [7, 11, 17], "test": [7, 11, 17]}}
    return tokenizer, messages, policy, seed, source_sha, source


def build(args):
    tokenizer, messages, policy, seed, source_sha, _ = args
    return build_recipe3_request_plan(tokenizer, messages, policy, seed, source_sha256=source_sha)


def check(args, plan, provenance):
    _, messages, policy, seed, source_sha, _ = args
    validate_recipe3_sampling(provenance, plan, policy, seed, source_sha, messages=messages)


def test_reconstructs_original_order_before_excluding_and_partitioning(tmp_path):
    args = fixture(tmp_path)
    plan, provenance = build(args)
    assert provenance["excluded_record_sha256"] == sorted(_digest(message) for message in args[1][:3])
    assert provenance["seed_prompt_token_ids"] == render_prompt_ids(args[0], args[1], 13, 16384)
    excluded = set(provenance["excluded_record_sha256"])
    partitions = provenance["partition_record_sha256"]
    assert all(not excluded.intersection(pool) for pool in partitions.values())
    assert sum(map(len, partitions.values())) == 33
    assert not (set(partitions["train"]) & set(partitions["validation"]) or
                set(partitions["validation"]) & set(partitions["test"]) or
                set(partitions["train"]) & set(partitions["test"]))
    for row in plan:
        assert not excluded.intersection(row["source_record_sha256"])
        assert len(row["prompt_token_ids"]) == row["length_bucket"]
        assert row["prompt_sha256"] == prompt_ids_sha256([row["prompt_token_ids"]])
        assert sum(span["end"] - span["start"] for span in row["source_record_spans"]) == row["length_bucket"]
    check(args, plan, provenance)


def test_all_occurrences_of_seed_record_are_excluded(tmp_path):
    messages = [[{"role": "user", "content": f"record-{i}"}] for i in range(36)]
    messages.extend(messages[:3])
    args = fixture(tmp_path, messages=messages)
    plan, provenance = build(args)
    assert sum(map(len, provenance["partition_record_sha256"].values())) == 33
    assert all(not set(provenance["excluded_record_sha256"]).intersection(row["source_record_sha256"]) for row in plan)


def test_each_length_has_distinct_deterministic_complete_record_start(tmp_path):
    args = fixture(tmp_path)
    plan, provenance = build(args)
    assert (plan, provenance) == build(args)
    for split in ("train", "validation", "test"):
        rows = [row for row in plan if row["split"] == split]
        assert len(set(row["source_record_offset"] for row in rows)) == len(rows)
        assert len(set(row["prompt_token_ids"][0] for row in rows)) == len(rows)
        assert all(row["source_record_spans"][0]["start"] == 0 for row in rows)
        assert all(second["prompt_token_ids"][:len(first["prompt_token_ids"])] != first["prompt_token_ids"]
                   for first, second in zip(rows, rows[1:]))


def test_actual_record_spans_and_repetition_are_explicit(tmp_path):
    args = fixture(tmp_path, lengths={split: [4, 200] for split in ("train", "validation", "test")})
    plan, provenance = build(args)
    for row in plan:
        if row["length_bucket"] == 4:
            assert len(row["source_record_sha256"]) == 1
            assert not row["source_repeated"]
        else:
            assert row["source_repeated"]
            assert row["source_record_spans"][-1]["cycle"] > 0
    check(args, plan, provenance)


@pytest.mark.parametrize("field,value", [("prompt_sha256", "b" * 64), ("token_count", 11),
                                          ("prompt_lengths", [3, 3, 6]), ("baseline_artifact_sha256", "unknown")])
def test_unknown_or_mismatched_seed_reconstruction_is_rejected(tmp_path, field, value):
    args = list(fixture(tmp_path))
    args[3][field] = value
    with pytest.raises(ValueError, match="recipe3|SHA256"):
        build(args)


def test_authenticated_pair_without_raw_manifest_is_reconstructible(tmp_path):
    args = fixture(tmp_path)
    assert args[3]["capture_manifest"] is None
    _, provenance = build(args)
    assert provenance["seed_capture"]["capture_manifest"] is None
    assert prompt_ids_sha256(provenance["seed_prompt_token_ids"]) == args[3]["prompt_sha256"]


def test_available_prompt_lengths_must_match_exact_prompt_boundaries(tmp_path):
    args = list(fixture(tmp_path))
    args[3]["prompt_lengths"] = [4, 4, 4]
    build(args)
    args[3]["prompt_lengths"] = [6, 6]
    with pytest.raises(ValueError, match="reconstruction differs"):
        build(args)


def test_changed_tokenizer_or_source_order_cannot_relabel_old_seed(tmp_path):
    args = list(fixture(tmp_path))
    args[0] = Tokenizer(5)
    with pytest.raises(ValueError, match="reconstruction differs"):
        build(args)
    args = list(fixture(tmp_path))
    args[1] = list(reversed(args[1]))
    with pytest.raises(ValueError, match="reconstruction differs"):
        build(args)


def test_fails_if_original_baseline_consumed_all_available_records(tmp_path):
    args = fixture(tmp_path, budget=200)
    with pytest.raises(ValueError, match="fewer than three"):
        build(args)


def test_fails_instead_of_reusing_same_start_when_split_has_too_few_records(tmp_path):
    args = fixture(tmp_path, messages=[[{"role": "user", "content": str(i)}] for i in range(9)])
    with pytest.raises(ValueError, match="distinct source start"):
        build(args)


@pytest.mark.parametrize("mutation", ["excluded", "partitions", "seed_tokens", "source_identity", "source_lengths",
                                      "offset", "spans", "actual_records", "seed_file", "missing_provenance"])
def test_retained_capture_rejects_tampered_provenance(tmp_path, mutation):
    args = fixture(tmp_path)
    plan, provenance = build(args)
    if mutation == "excluded":
        provenance["excluded_record_sha256"].pop()
    elif mutation == "partitions":
        provenance["partition_record_sha256"]["validation"][0] = provenance["partition_record_sha256"]["train"][0]
    elif mutation == "seed_tokens":
        provenance["seed_prompt_token_ids"][0][0] += 1
    elif mutation == "source_identity":
        provenance["source_records"][0]["sha256"] = "c" * 64
    elif mutation == "source_lengths":
        provenance["source_records"][0]["token_count"] += 1
    elif mutation == "offset":
        plan[0]["source_record_offset"] += 1
    elif mutation == "spans":
        plan[0]["source_record_spans"][0]["end"] -= 1
    elif mutation == "actual_records":
        plan[0]["source_record_sha256"] = provenance["excluded_record_sha256"][:1]
    elif mutation == "seed_file":
        provenance["seed_capture"]["baseline_artifact_sha256"] = "c" * 64
    else:
        provenance = None
    with pytest.raises(ValueError, match="recipe3"):
        check(args, plan, provenance)


def test_trace_loader_checks_source_bytes_and_current_baseline_binding(tmp_path):
    args = fixture(tmp_path)
    plan, provenance = build(args)
    _, _, policy, seed, source_sha, source = args
    top = {"prompt_source": str(source), "prompt_source_sha256": source_sha,
           "recipe3_sampling": provenance, "request_plan": plan}
    config = {"optr_baseline": "recipe3", "optr_seed_capture": seed}
    validate_recipe3_trace_manifest(top, config, policy)
    wrong = copy.deepcopy(config)
    wrong["optr_seed_capture"]["baseline_artifact_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="provenance differs"):
        validate_recipe3_trace_manifest(top, wrong, policy)
    with pytest.raises(ValueError, match="baseline capture metadata"):
        validate_recipe3_trace_manifest(top, {"optr_baseline": "recipe3"}, policy)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="source file hash differs"):
        validate_recipe3_trace_manifest(top, config, policy)


def test_real_bundled_records_reconstruct_and_exclude_before_partition(tmp_path):
    source = calibration_prompt_source({}, tmp_path / "bundled")
    messages = load_messages(source)
    args = fixture(tmp_path, messages=messages, budget=30000, tokenizer=Tokenizer(300))
    plan, provenance = build(args)
    assert len(messages) == 198
    assert len(provenance["excluded_record_sha256"]) == 100
    assert sorted(map(len, provenance["partition_record_sha256"].values())) == [32, 33, 33]
    assert all(not set(provenance["excluded_record_sha256"]).intersection(row["source_record_sha256"]) for row in plan)
