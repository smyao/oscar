"""Archive #97/#173/#175: source seed authentication and candidate handoff."""
import copy
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend.optr_artifacts import file_sha256
from oscar_ascend.optr_oscar_seed import BaseSeedAccumulator, SEED_RECIPE, legacy_seed_profile
from oscar_ascend.rotations import calibration_cache_identity, checkpoint_identity
from tools import prepare_optr_seed as seed
from tools.prepare_optr_rotations import _base_config, _prepare_base
from tools.calibrate_model_rotations import source_seed_prompt_prefix
from test_optr_integration import configuration


def case(tmp_path):
    config = configuration(tmp_path)
    config['served_model_name'] = 'host-test-model'
    model_path = Path(config['model']) / 'config.json'
    model = json.loads(model_path.read_text())
    model['architectures'] = ['Qwen3_5ForConditionalGeneration']
    model_path.write_text(json.dumps(model))
    names, dim, fingerprint = checkpoint_identity(config["model"])
    base = _base_config(config, fingerprint)
    # Test artifacts stay in the test directory rather than the project cache.
    base["rotation_artifact"] = str(tmp_path / "source-seed" / "seed.pt")
    profile = calibration_cache_identity(base, fingerprint)
    assert profile == calibration_cache_identity(config, fingerprint)["base_calibration_profile"]
    config_path = tmp_path / "base-config.json"
    config_path.write_text(json.dumps(base))

    def capture(path, directory):
        assert json.loads(Path(path).read_text()) == base
        ranks = []
        for rank in range(2):
            rng = torch.Generator().manual_seed(rank + 37)
            collector = BaseSeedAccumulator(head_dim=dim, local_q_heads=2,
                local_kv_heads=1, token_budget=40, testing=True)
            collector.observe(torch.randn(40, 2, dim, generator=rng).bfloat16(),
                torch.randn(40, 1, dim, generator=rng).bfloat16(),
                torch.randn(40, 1, dim, generator=rng).bfloat16())
            payload = {"rank": rank, "tp_size": 2, "global_q_heads": 4,
                       "global_kv_heads": 2, "prompt_sha256": "e" * 64,
                       "model_fingerprint": fingerprint, "layers": {3: collector.export_payload()}}
            rank_path = directory / f"rank{rank}.pt"
            torch.save(payload, rank_path)
            ranks.append(str(rank_path))
        return {"status": "native_post_rope_capture_complete", "calibration_profile": profile,
                "model_fingerprint": fingerprint, "layer_ids": [3], "token_count": 40,
                "prompt_sha256": "e" * 64, "rank_paths": ranks,
                "rank_sha256": {p: file_sha256(p) for p in ranks}}
    return base, config_path, capture


def test_seed_prepare_publish_load_and_candidate_handoff_without_recipe3_pair(tmp_path):
    base, config_path, capture = case(tmp_path)
    result = seed.prepare(base, config_path=config_path, testing=True, capture_runner=capture)
    assert result["reused"] is False
    loaded = seed.load_seed(base, testing=True)
    assert loaded["source_provenance"]["recipe"] == SEED_RECIPE
    assert not list(Path(base["rotation_artifact"]).parent.glob("*rotation*.pt"))
    def forbidden(*a, **kw):
        raise AssertionError("matching source seed must not recapture")
    assert seed.prepare(base, config_path=config_path, testing=True, capture_runner=forbidden)["reused"]
    handed = _prepare_base(base, tmp_path, testing=True, import_pair=None,
                           base_runner=lambda *_: None)
    assert torch.equal(handed["layers"][next(iter(handed["layers"]))]["Rk"],
                       loaded["layers"][next(iter(loaded["layers"]))]["Rk"])
    with pytest.raises(ValueError, match="test-only"):
        seed.load_seed(base)


@pytest.mark.parametrize("corruption", ["hash", "profile", "pending"])
def test_seed_rejects_corrupt_or_incomplete_publication(tmp_path, corruption):
    base, config_path, capture = case(tmp_path)
    seed.prepare(base, config_path=config_path, testing=True, capture_runner=capture)
    destination = Path(base["rotation_artifact"])
    if corruption == "hash":
        destination.write_bytes(destination.read_bytes() + b"changed")
    elif corruption == "profile":
        manifest_path = destination.parent / ".seed.complete.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["profile"]["source_provenance"]["sample_cap"] = 1
        manifest_path.write_text(json.dumps(manifest))
    else:
        (destination.parent / ".seed.pending.json").write_text("{}")
    with pytest.raises(ValueError, match="seed manifest|seed publication"):
        seed.load_seed(base, testing=True)


def test_seed_loader_rejects_authenticated_old_full_source_artifact(tmp_path):
    base, config_path, capture = case(tmp_path)
    seed.prepare(base, config_path=config_path, testing=True, capture_runner=capture)
    destination = Path(base["rotation_artifact"])
    artifact = torch.load(destination, map_location="cpu", weights_only=False)
    old_profile = legacy_seed_profile(artifact["calibration_profile"])
    artifact["calibration_profile"] = old_profile
    artifact["source_provenance"] = old_profile["source_provenance"]
    torch.save(artifact, destination)
    manifest_path = destination.parent / ".seed.complete.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(profile=old_profile, sha256=file_sha256(destination))
    manifest_path.write_text(json.dumps(manifest))
    # Even a matching file hash cannot make a historical full-source seed a
    # current train-only seed. Raw trace compatibility is a separate path.
    with pytest.raises(ValueError, match="seed manifest/hash/profile"):
        seed.load_seed(base, testing=True)


def test_seed_corrupt_capture_cannot_publish_and_interruption_recaptures(tmp_path):
    base, config_path, capture = case(tmp_path)
    def corrupt(*args):
        report = capture(*args)
        report["rank_sha256"][report["rank_paths"][0]] = "0" * 64
        return report
    with pytest.raises(ValueError, match="rank payload hashes"):
        seed.prepare(base, config_path=config_path, testing=True, capture_runner=corrupt)
    destination = Path(base["rotation_artifact"])
    assert not destination.exists()
    destination.write_text("interrupted generation")
    (destination.parent / ".seed.pending.json").write_text("{}")
    assert not seed.prepare(base, config_path=config_path, testing=True, capture_runner=capture)["reused"]
    assert seed.load_seed(base, testing=True)["calibrated"]


def test_seed_prompt_prefix_keeps_whole_prompts_and_avoids_unused_forwards():
    for sizes, expected in (([700, 200], [700]), ([100, 400, 100, 100], [100, 400, 100]),
                            ([256, 256, 20], [256, 256]), ([100, 200], [100, 200])):
        prompts = [list(range(n)) for n in sizes]
        before = copy.deepcopy(prompts)
        assert list(map(len, source_seed_prompt_prefix(prompts))) == expected
        assert prompts == before


def test_production_base_child_uses_source_entry_config_and_inherited_live_output(tmp_path, monkeypatch):
    import os
    import sys
    from tools import prepare_optr_rotations
    from tools.model_config import apply_model_options
    base, _, _ = case(tmp_path)
    monkeypatch.setenv('OSCAR_ROTATION_ARTIFACT', '/tmp/old-final.pt')
    before = dict(os.environ)
    called = []
    def run(command, **kwargs):
        called.append(command)
        assert command == [sys.executable, '-m', 'tools.prepare_optr_seed', '--config',
                           str(tmp_path / 'base-config.json')]
        child = json.loads(Path(command[-1]).read_text())
        assert child == base and child['rotation_seed'] == SEED_RECIPE
        resolved = apply_model_options(child, require_model=True)
        assert resolved['rotation_seed'] == SEED_RECIPE
        assert resolved['rotation_artifact'] == base['rotation_artifact']
        assert kwargs['check'] is True and kwargs['timeout'] > 0
        assert 'OSCAR_ROTATION_ARTIFACT' not in kwargs['env']
        assert not kwargs.get('capture_output')
        assert kwargs.get('stdout') is None and kwargs.get('stderr') is None
    def load(config, *, testing):
        assert config == base and testing is False
        return {'test_only': False, 'source': 'host-double-not-npu-evidence'}
    monkeypatch.setattr(prepare_optr_rotations.subprocess, 'run', run)
    monkeypatch.setattr(seed, 'load_seed', load)
    result = _prepare_base(base, tmp_path, testing=False, import_pair=None, base_runner=None)
    assert result['test_only'] is False and len(called) == 1 and dict(os.environ) == before
