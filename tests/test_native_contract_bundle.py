"""Archive #74-78/#94/#125: fresh clones must retain real CPU source contracts."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools.native_contract_sources import NativeSources, NativeSourceError
from tools import local_simulation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def distributed_sources(tmp_path):
    # Deliberately no native .git directories, installed vLLM or source trees.
    for relative in ("tests/native_contract_manifest.json", "references/native_cpu_contracts.zip"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return NativeSources(tmp_path)


def test_fresh_clone_uses_identical_complete_native_files(distributed_sources):
    source = distributed_sources
    evidence = source.verify()
    assert all(row["mode"] == "bundled_snapshot" and not row["git_checkout_verified"] for row in evidence)
    assert sum(row["files_sha256_verified"] for row in evidence) == len(source.files)
    for test in (ROOT / "tests").glob("test_*.py"):
        for node in ast.walk(ast.parse(test.read_text())):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.startswith(("references/vllm/", "references/vllm-ascend/"))
                    and node.value.endswith(".py")):
                assert node.value in source.files, f"unshipped test dependency: {test.name}: {node.value}"
    for reference in source.manifest["references"]:
        assert "Apache License" in source.read_text(reference["path"] + "/LICENSE")


def test_bundle_corruption_is_an_error_not_a_skip(distributed_sources):
    path = distributed_sources.root / distributed_sources.manifest["source_bundle"]["path"]
    path.write_bytes(path.read_bytes() + b"damaged")
    with pytest.raises(NativeSourceError, match="bundle hash mismatch"):
        distributed_sources.verify()


def test_individual_native_hash_is_enforced_even_after_container_resigning(distributed_sources):
    source = distributed_sources
    path = source.root / source.manifest["source_bundle"]["path"]
    with zipfile.ZipFile(path) as original:
        files = {name: original.read(name) for name in original.namelist()}
    member = next(name for name in files if name.endswith("scheduler.py"))
    files[member] += b"\n# unexpected change\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    source.manifest["source_bundle"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(NativeSourceError, match="native source hash mismatch"):
        source.read_bytes(member)


def test_present_incomplete_checkout_cannot_be_hidden_by_bundle(distributed_sources):
    source = distributed_sources
    reference = source.manifest["references"][0]
    (source.root / reference["path"]).mkdir()
    member = reference["path"] + "/" + next(iter(reference["files"]))
    with pytest.raises(NativeSourceError, match="incomplete reference checkout"):
        source.read_bytes(member)


def test_missing_bundle_stops_before_launching_tests(distributed_sources, tmp_path, monkeypatch, capsys):
    source = distributed_sources
    (source.root / source.manifest["source_bundle"]["path"]).unlink()
    monkeypatch.setattr(local_simulation, "NativeSources", lambda _root: source)
    def forbidden_phase(*args, **kwargs):
        pytest.fail("missing input must stop before test subprocess")
    monkeypatch.setattr(local_simulation, "run_phase", forbidden_phase)
    output = tmp_path / "run"
    assert local_simulation.run(output, 3) == 2
    report = json.loads((output / "report.json").read_text())
    assert report["failure_stage"] == "native_source_inputs" and report["phases"] == []
    assert "missing native CPU source bundle" in capsys.readouterr().out
