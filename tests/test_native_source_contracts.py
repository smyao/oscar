"""Archive #74-78/#127/#128/#156: reference provenance and scheduler drift.

Local development tests only; never a serve/readiness preflight. Updating the
reference checkout requires reviewing each changed native seam, not silently
regenerating these fingerprints. Reference trees are read-only.
"""
import ast
import hashlib
import json
from pathlib import Path
from tools.native_contract_sources import native_sources

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/native_contract_manifest.json"


def semantic_node(value):
    if value is Ellipsis:
        return {"constant_type": "Ellipsis"}
    if isinstance(value, ast.AST):
        # Empty type_params was added in Python 3.12. Ignore only that empty
        # field; no operators, constants or control flow are discarded.
        return [type(value).__name__, {key: semantic_node(child) for key, child in ast.iter_fields(value)
                if not (key == "type_params" and child == [])}]
    if isinstance(value, list):
        return [semantic_node(child) for child in value]
    return value


def method_fingerprint(source, qualified):
    tree = ast.parse(source)
    for name in qualified.split("."):
        tree = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and node.name == name)
    payload = json.dumps(semantic_node(tree), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_reference_evidence_is_bound_to_actual_checkouts_and_recorded_seams():
    manifest = json.loads(MANIFEST.read_text())
    sources = native_sources()
    sources.verify()
    for reference in manifest["references"]:
        for relative, expected in reference["files"].items():
            source = sources.read_bytes(f'{reference["path"]}/{relative}')
            assert hashlib.sha256(source).hexdigest() == expected["sha256"], relative
            for method, fingerprint in expected["methods"].items():
                assert method_fingerprint(source.decode(), method) == fingerprint, \
                    f"{relative}:{method}: native control flow changed; reassess prefix policy"


def test_semantic_fingerprint_detects_scheduler_formula_and_eagle_changes():
    source = native_sources().read_text("references/vllm/vllm/v1/core/sched/scheduler.py")
    method = "Scheduler._mamba_block_aligned_split"
    digest = method_fingerprint(source, method)
    for old, new in (("if self.use_eagle:", "if not self.use_eagle:"),
                     ("num_new_tokens // block_size * block_size", "num_new_tokens"),
                     ("request.num_tokens % block_size", "request.num_prompt_tokens % block_size")):
        assert old in source
        assert method_fingerprint(source.replace(old, new), method) != digest
    assert method_fingerprint(source + "\n# metadata-only comment\n", method) == digest
