"""Archive #176: automatic failure evidence cannot restart or bypass fitting."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from tools import prepare_rotations, prepare_optr_rotations
from oscar_ascend.optr_checkpoint import fitting_implementation_digest
from test_optr_integration import configuration

ROOT = Path(__file__).resolve().parents[1]
BASE = "8c414aa"
MESSAGE = "recipe3 OptR found no strict validation improvement; no refined PT is published"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = {**configuration(tmp_path), "optr_baseline": "recipe3"}
    fake = ModuleType("tools.optr_rejection_report")
    calls = []
    def report(config, **kwargs):
        calls.append((config, kwargs))
        return {"status": "no_strict_improvement", "fit_reexecuted": False}
    fake.report_no_improvement = report
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    return config, fake, calls


def test_automatic_summary_preserves_original_failure_and_all_forwarded_arguments(setup, monkeypatch, capsys):
    config, _, calls = setup
    failure = ValueError(MESSAGE)
    forwarded = []
    def fail(config, **kwargs):
        forwarded.append((config, kwargs))
        raise failure
    monkeypatch.setattr(prepare_optr_rotations, "prepare", fail)
    options = dict(config_path="target.json", output="refined.pt", import_pair=("k", "v"),
                   capture_runner=object(), testing=True)
    with pytest.raises(ValueError) as exc:
        prepare_rotations.prepare(config, **options)
    assert exc.value is failure
    assert forwarded == [(config, options)]
    assert calls == [(config, {"output": "refined.pt", "testing": True})]
    line = capsys.readouterr().out.strip()
    from tools.phase import _compact_should_mirror
    assert _compact_should_mirror(line, {})
    summary = json.loads(line.split("OPTR_PROGRESS ", 1)[1])
    assert summary["phase"] == "no_improvement_report"
    assert summary["fit_reexecuted"] is False


@pytest.mark.parametrize("reason", ["independent test regressed", "corrupt artifact", "incomplete capture"])
def test_unrelated_errors_do_not_trigger_or_get_hidden_by_report(setup, monkeypatch, reason):
    config, _, calls = setup
    failure = ValueError(reason)
    def fail(*args, **kwargs): raise failure
    monkeypatch.setattr(prepare_optr_rotations, "prepare", fail)
    with pytest.raises(ValueError) as exc:
        prepare_rotations.prepare(config, testing=True)
    assert exc.value is failure and not calls


def test_success_and_legacy_policy_remain_unchanged(setup, monkeypatch, capsys):
    config, _, calls = setup
    result = {"source": "optr_authenticated_cache", "reused": True}
    monkeypatch.setattr(prepare_optr_rotations, "prepare", lambda *_args, **_kw: result)
    assert prepare_rotations.prepare(config, testing=True) is result
    assert not calls and not capsys.readouterr().out
    config.pop("optr_baseline")
    def fail(*args, **kwargs): raise ValueError(MESSAGE)
    monkeypatch.setattr(prepare_optr_rotations, "prepare", fail)
    with pytest.raises(ValueError, match="no strict"):
        prepare_rotations.prepare(config, testing=True)
    assert not calls


def test_corrupt_diagnostic_evidence_keeps_original_nonzero_failure(setup, monkeypatch, capsys):
    config, fake, _ = setup
    failure = ValueError(MESSAGE)
    def fail(*args, **kwargs): raise failure
    def bad_report(*args, **kwargs): raise ValueError("checkpoint hash differs")
    monkeypatch.setattr(prepare_optr_rotations, "prepare", fail)
    fake.report_no_improvement = bad_report
    with pytest.raises(ValueError) as exc:
        prepare_rotations.prepare(config, testing=True)
    assert exc.value is failure
    assert "report_failed" in capsys.readouterr().out


def test_fitting_implementation_bytes_and_digest_keep_completed_device_work_reusable():
    source = subprocess.check_output(["git", "show", f"{BASE}:oscar_ascend/optr_checkpoint.py"], cwd=ROOT)
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fitting_implementation_digest")
    names = next(ast.literal_eval(n.value) for n in function.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "names" for t in n.targets))
    digest = hashlib.sha256()
    for name in names:
        original = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=ROOT)
        assert (ROOT / name).read_bytes() == original, name
        digest.update(name.encode() + b"\0" + original)
    assert fitting_implementation_digest() == digest.hexdigest()


def test_preparer_source_outside_the_existing_optr_dispatch_is_unchanged():
    before = ast.parse(subprocess.check_output(["git", "show", f"{BASE}:tools/prepare_rotations.py"], cwd=ROOT))
    after = ast.parse((ROOT / "tools/prepare_rotations.py").read_text())
    def remove_dispatch(tree):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
        expected = ast.parse('config.get("rotation_refinement") == "optr"', mode="eval").body
        function.body = [n for n in function.body if not (
            isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(expected))]
    remove_dispatch(before)
    remove_dispatch(after)
    assert ast.dump(before) == ast.dump(after)
