"""Archive G14/G15/#97: prepare real rotations without an online data dependency.

The official GPQA bytes and their SHA bind calibration inputs. Invalid inputs
must fail before publishing a CSV; these are host tests, not NPU acceptance.
"""
import csv
import gzip
import hashlib
import io
import os
from pathlib import Path
import socket
import subprocess
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
