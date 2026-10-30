"""Archive G14/G15/#97: prepare real rotations without an online data dependency.

The official GPQA bytes and their SHA bind calibration inputs. Invalid inputs
must fail before publishing a CSV; these are host tests, not NPU acceptance.
"""
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import urllib.request

import pytest

from tools import calibration_data
from oscar_ascend.rotations import calibration_cache_identity, rotation_artifact_path


EXPECTED_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
GPQA_FIELDS = {"Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"}


def test_bundled_official_gpqa_has_pinned_bytes_and_all_198_questions():
    assert calibration_data.GPQA_SHA256 == EXPECTED_SHA256
    assert calibration_data.GPQA_URL == "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    assert calibration_data.BUNDLED_GPQA == Path(calibration_data.__file__).with_name("data") / "gpqa_diamond.csv.gz"
    data = gzip.decompress(calibration_data.BUNDLED_GPQA.read_bytes())
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    assert GPQA_FIELDS <= set(reader.fieldnames)
    rows = list(reader)
    assert len(rows) == 198
    assert all(all(row[field] for field in GPQA_FIELDS) for row in rows)


def test_default_source_needs_no_network_and_ignores_old_hf_cache(tmp_path, monkeypatch):
    def no_network(*_args, **_kwargs):
        pytest.fail("default calibration data attempted a network or subprocess call")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(subprocess, "run", no_network)
    old_cache = tmp_path / "hf-cache" / "oscar-calibration" / "gpqa_diamond.csv"
    old_cache.parent.mkdir(parents=True)
    old_cache.write_bytes(b"stale or corrupted downloaded data")
    monkeypatch.setenv("HF_HOME", str(old_cache.parent.parent))
    destination = tmp_path / "capture"
    source = calibration_data.calibration_prompt_source({}, destination)
    assert source == (destination / "gpqa_diamond.csv").resolve()
    assert source.read_bytes() == gzip.decompress(calibration_data.BUNDLED_GPQA.read_bytes())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_SHA256
    assert old_cache.read_bytes() == b"stale or corrupted downloaded data"


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_explicit_source_expands_user_path_without_copying(tmp_path, suffix):
    source = tmp_path / ("custom" + suffix)
    source.write_text("explicit calibration input\n")
    # Exercise real expanduser semantics without changing the user's HOME.
    user_path = "~/" + os.path.relpath(source, Path.home())
    destination = tmp_path / "unused-output"
    actual = calibration_data.calibration_prompt_source(
        {"rotation_calibration_prompts": user_path}, destination)
    assert actual == source.resolve()
    assert actual.read_text() == "explicit calibration input\n"
    assert not (destination / "gpqa_diamond.csv").exists()


def test_missing_explicit_source_is_not_replaced_with_default(tmp_path):
    missing = tmp_path / "missing.jsonl"
    destination = tmp_path / "output"
    with pytest.raises((FileNotFoundError, ValueError)):
        calibration_data.calibration_prompt_source(
            {"rotation_calibration_prompts": str(missing)}, destination)
    assert not (destination / "gpqa_diamond.csv").exists()


@pytest.mark.parametrize("contents", [b"not a gzip archive", gzip.compress(b"incorrect GPQA bytes")])
def test_corrupt_or_unmatched_bundled_data_is_rejected_before_csv_write(tmp_path, monkeypatch, contents):
    bundled = tmp_path / "invalid.csv.gz"
    bundled.write_bytes(contents)
    monkeypatch.setattr(calibration_data, "BUNDLED_GPQA", bundled)
    destination = tmp_path / "output"
    with pytest.raises((ValueError, OSError, EOFError)):
        calibration_data.calibration_prompt_source({}, destination)
    assert not (destination / "gpqa_diamond.csv").exists()


def test_pinned_digest_is_checked_before_csv_write(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_data, "GPQA_SHA256", "0" * 64)
    destination = tmp_path / "output"
    with pytest.raises(ValueError):
        calibration_data.calibration_prompt_source({}, destination)
    assert not (destination / "gpqa_diamond.csv").exists()


def test_default_profile_binds_gpqa_digest_and_changes_rotation_directory(tmp_path, monkeypatch):
    config = {"rotation_method": "qqt_sst", "rotation_calibration_tokens": 30000}
    profile = calibration_cache_identity(config, "model-identity")
    assert profile["prompt_source"]["sha256"] == EXPECTED_SHA256
    before = rotation_artifact_path(config, "model-identity", root=tmp_path)
    monkeypatch.setattr(calibration_data, "GPQA_SHA256", "0" * 64)
    changed = calibration_cache_identity(config, "model-identity")
    assert changed["prompt_source"]["sha256"] == "0" * 64
    after = rotation_artifact_path(config, "model-identity", root=tmp_path)
    assert before != after
    assert before.name == after.name == "calibrated.pt"


def test_source_seed_never_renders_unused_overlength_prompt(tmp_path, monkeypatch):
    """Reach the model boundary after the first complete 512+-row prompt.

    An unused trailing record must not block seed preparation. This exercises
    the real collect dispatch up to geometry lookup; no model is constructed.
    """
    import json
    from tools import calibrate_model_rotations as capture, model_config, prepare_rotations, target_cli
    from oscar_ascend.optr_oscar_seed import SEED_RECIPE

    messages = list(range(6))
    train = calibration_data.partition_calibration_messages(messages)["train"]
    lengths = {message: (600 if index == 0 else 20000)
               for index, (_, message) in enumerate(train)}
    rendered = []
    class Tokenizer:
        def apply_chat_template(self, message, **kwargs):
            length = lengths[message]
            rendered.append(length)
            return list(range(length))

    class ModelBoundaryReached(Exception):
        pass

    def model_boundary(_path):
        raise ModelBoundaryReached

    source = tmp_path / "source.jsonl"
    source.write_text('"source records are supplied by the test tokenizer"\n')
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "/unused/model", "rotation_seed": SEED_RECIPE,
        "rotation_calibration_tokens": 30000, "max_num_batched_tokens": 16384, "max_model_len": 32768}))
    monkeypatch.setattr(model_config, "apply_model_options", lambda config, **kwargs: config)
    monkeypatch.setattr(target_cli, "target_env", lambda _: {})
    monkeypatch.setattr(capture, "calibration_process_env", lambda _: {})
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.delenv("OSCAR_PASSIVE_TIMING_CONTROL", raising=False)
    monkeypatch.delenv("OSCAR_TRACE_DIR", raising=False)
    monkeypatch.setattr(capture, "calibration_prompt_source", lambda *_: source)
    monkeypatch.setattr(capture, "load_messages", lambda _: messages)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())))
    monkeypatch.setattr(prepare_rotations, "model_geometry", model_boundary)
    with pytest.raises(ModelBoundaryReached):
        capture.collect(config, tmp_path / "capture")
    assert rendered == [600]


@pytest.mark.parametrize("sizes,expected", [
    ([700, 200], [700]), ([100, 400, 100, 100], [100, 400, 100]),
    ([256, 256, 20], [256, 256]), ([100, 200], [100, 200]),
])
def test_seed_early_stop_preserves_previous_complete_prompt_selection(sizes, expected):
    from tools.calibrate_model_rotations import render_prompt_ids, source_seed_prompt_prefix
    calls = []
    class Tokenizer:
        def apply_chat_template(self, length, **kwargs):
            calls.append(length)
            return list(range(length))
    before = source_seed_prompt_prefix(render_prompt_ids(Tokenizer(), sizes, 30000, 16384))
    calls.clear()
    after = render_prompt_ids(Tokenizer(), sizes, 30000, 16384, stop_after_tokens=512)
    assert after == before and list(map(len, after)) == expected
    assert calls == expected


def test_seed_early_stop_keeps_needed_prompt_chunk_and_whole_prompt_budget_checks():
    from tools.calibrate_model_rotations import render_prompt_ids
    class Tokenizer:
        def apply_chat_template(self, length, **kwargs):
            return list(range(length))
    with pytest.raises(ValueError, match="one prefill chunk"):
        render_prompt_ids(Tokenizer(), [100, 20000], 30000, 16384, stop_after_tokens=512)
    with pytest.raises(ValueError, match="no complete prompt"):
        render_prompt_ids(Tokenizer(), [600], 500, 16384, stop_after_tokens=512)
    actual = render_prompt_ids(Tokenizer(), [400, 200], 500, 16384, stop_after_tokens=512)
    assert list(map(len, actual)) == [400]  # Do not truncate the next prompt.
    # Recipe2/3 omit the stopping extent and still process the original budget.
    with pytest.raises(ValueError, match="one prefill chunk"):
        render_prompt_ids(Tokenizer(), [600, 20000], 30000, 16384)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_seed_prompt_stopping_extent_must_be_a_positive_integer(invalid):
    from tools.calibrate_model_rotations import render_prompt_ids
    with pytest.raises(ValueError, match="stopping extent"):
        render_prompt_ids(None, [], 30000, 16384, stop_after_tokens=invalid)


def _record_identity(message):
    return hashlib.sha256(json.dumps(message, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class _TracePlanTokenizer:
    """Deterministic host stub; no claim about the model's real tokenization."""
    def apply_chat_template(self, message, **kwargs):
        return list(bytes.fromhex(_record_identity(message)))


_TRACE_LENGTHS = {"train": [4096, 32768, 65536],
                  "validation": [4096, 32768, 65536], "test": [32768, 65536]}


def test_bundled_gpqa_trace_plan_is_unchanged_by_shared_partition_extraction(tmp_path):
    from tools.calibrate_model_rotations import load_messages
    from tools.calibrate_optr_traces import build_request_plan
    messages = load_messages(calibration_data.calibration_prompt_source({}, tmp_path))
    plan = build_request_plan(_TracePlanTokenizer(), messages, {"capture_lengths": _TRACE_LENGTHS})
    # Frozen before extraction, using the complete bundled 198-record source,
    # every request field and every packed/repeated token in the old algorithm.
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == "dcc66db1cd157db2f82c35f9c59c1cdf746d5e361ec48d3adfe0f40d926b6ee9"


def test_record_partition_is_disjoint_deduplicated_and_independent_of_source_order(tmp_path):
    from tools.calibrate_model_rotations import load_messages
    messages = load_messages(calibration_data.calibration_prompt_source({}, tmp_path))
    partitions = calibration_data.partition_calibration_messages(messages)
    assert partitions == calibration_data.partition_calibration_messages(list(reversed(messages)) + messages[:4])
    identities = {split: {key for key, _ in rows} for split, rows in partitions.items()}
    assert [len(identities[split]) for split in ("train", "validation", "test")] == [66, 66, 66]
    assert len(set.union(*identities.values())) == 198
    assert all([key for key, _ in rows] == sorted(key for key, _ in rows) for rows in partitions.values())
    for count in (0, 1, 2):
        with pytest.raises(ValueError, match="at least three distinct"):
            calibration_data.partition_calibration_messages(messages[:count] * 3)


@pytest.mark.parametrize("seed_enabled", [True, False])
def test_real_gpqa_seed_uses_only_training_records_before_model_load(tmp_path, monkeypatch, seed_enabled):
    """Archive #173/#175/#176: heldout records must not train initialization."""
    from tools import calibrate_model_rotations as capture, model_config, prepare_rotations, target_cli
    from tools.calibrate_optr_traces import build_request_plan
    from oscar_ascend.optr_oscar_seed import SEED_RECIPE
    source = calibration_data.calibration_prompt_source({}, tmp_path / "source")
    messages = capture.load_messages(source)
    plan = build_request_plan(_TracePlanTokenizer(), messages, {"capture_lengths": _TRACE_LENGTHS})
    identities = {row["split"]: set(row["source_record_sha256"]) for row in plan}
    # This proves why reading the source in its original order leaks heldout
    # records even though the later OptR trace partitions are disjoint.
    assert [next(split for split, ids in identities.items() if _record_identity(m) in ids)
            for m in messages[:6]] == ["test", "test", "validation", "train", "train", "train"]
    rendered = []
    class Tokenizer:
        def apply_chat_template(self, message, **kwargs):
            rendered.append(_record_identity(message))
            return list(range(600))
    class ModelBoundaryReached(Exception):
        pass
    def model_boundary(_path):
        raise ModelBoundaryReached
    config = {"model": "/unused/model", "rotation_calibration_tokens": 30000,
              "max_num_batched_tokens": 16384, "max_model_len": 32768}
    if seed_enabled:
        config["rotation_seed"] = SEED_RECIPE
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(model_config, "apply_model_options", lambda config, **kwargs: config)
    monkeypatch.setattr(target_cli, "target_env", lambda _: {})
    monkeypatch.setattr(capture, "calibration_process_env", lambda _: {})
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.delenv("OSCAR_PASSIVE_TIMING_CONTROL", raising=False)
    monkeypatch.delenv("OSCAR_TRACE_DIR", raising=False)
    monkeypatch.setattr(capture, "calibration_prompt_source", lambda *_: source)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())))
    monkeypatch.setattr(prepare_rotations, "model_geometry", model_boundary)
    with pytest.raises(ModelBoundaryReached):
        capture.collect(path, tmp_path / "capture")
    if seed_enabled:
        assert rendered == [min(identities["train"])]
        assert set(rendered) <= identities["train"]
        assert set(rendered).isdisjoint(identities["validation"] | identities["test"])
    else:
        # Recipe2/3 preserve their original order and full whole-prompt budget.
        assert rendered == [_record_identity(m) for m in messages[:50]]
