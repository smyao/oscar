"""#145/#157/#164/#172: exact derivatives and real build-cache admission.

These are source, dispatch and host build-orchestration tests. The compiler
fixture emits fake artifacts; none of these assertions certify CANN or NPU.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re

import pytest

from oscar_ascend.ops.cv_dispatch import OFFICIAL_CV_MAP, select_cv_op
from tools import build_ops, generate_active_kernels as generated, install_serve
from tools import probe_aligned_attention, probe_official_attention, probe_prefix_cache
from tools.environment import file_fingerprint
from tools.serving_variants import variant_config
from test_build_reuse import compiler  # noqa: F401: use the real checked builder fixture


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (*generated.TOOL_PATHS, generated.MANIFEST,
          *(entry[key] for entry in generated.KERNELS for key in ("mother", "active")))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stage(root):
    for name in INPUTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    return root


@pytest.mark.parametrize("entry", generated.KERNELS)
def test_complete_mother_and_derivative_are_exact_reversible_sources(entry):
    mother = (ROOT / entry["mother"]).read_bytes()
    active = (ROOT / entry["active"]).read_bytes()
    module = importlib.import_module(entry["transform"])
    assert sha(mother) == entry["mother_sha256"]
    assert active != mother
    assert module.transform(mother.decode()).encode() == active
    assert module.restore(active.decode()).encode() == mother
    # All public kernel declarations and host launch definitions are retained
    # verbatim; only the independently audited internal methods differ.
    public_start = mother.index(b'extern "C" __global__ __aicore__ void ')
    assert active[active.index(b'extern "C" __global__ __aicore__ void '):] == mother[public_start:]


def test_checked_manifest_binds_all_generation_tools_and_all_sources():
    generated.check_generated_sources()
    assert generated.main(["--check"]) == 0
    manifest = json.loads((ROOT / generated.MANIFEST).read_bytes())
    assert manifest["generation_tools_sha256"] == {
        name: sha((ROOT / name).read_bytes()) for name in generated.TOOL_PATHS}
    for name, expected in generated.outputs().items():
        assert (ROOT / name).read_bytes() == expected
    fingerprints = file_fingerprint(ROOT / "csrc")
    for name in (generated.MANIFEST, *(entry[key] for entry in generated.KERNELS
                                     for key in ("mother", "active"))):
        assert fingerprints[name.removeprefix("csrc/")] == sha((ROOT / name).read_bytes())


@pytest.mark.parametrize("name", INPUTS)
def test_any_source_or_tool_drift_fails_closed_without_rewriting(tmp_path, name):
    root = stage(tmp_path)
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n// unaudited source drift\n")
    before = {item: (root / item).read_bytes() for item in INPUTS}
    with pytest.raises(ValueError, match="changed|drift"):
        generated.check_generated_sources(root)
    assert {item: (root / item).read_bytes() for item in INPUTS} == before


@pytest.mark.parametrize("damage", ["missing", "symlink"])
def test_missing_or_symlinked_derivative_is_rejected(tmp_path, damage):
    root = stage(tmp_path)
    path = root / generated.KERNELS[0]["active"]
    path.unlink()
    if damage == "symlink":
        path.symlink_to(ROOT / generated.KERNELS[0]["active"])
    with pytest.raises(ValueError, match="missing or symlinked"):
        generated.check_generated_sources(root)


def test_non_reversible_transform_is_rejected_before_any_output_write(tmp_path, monkeypatch):
    root = stage(tmp_path)
    module = importlib.import_module(generated.KERNELS[0]["transform"])
    monkeypatch.setattr(module, "restore", lambda _: "not the full protected mother")
    before = {name: (root / name).read_bytes() for name in INPUTS}
    with pytest.raises(ValueError, match="complete mother"):
        generated.check_generated_sources(root)
    assert before == {name: (root / name).read_bytes() for name in INPUTS}


def compiled_sources():
    cmake = (ROOT / "csrc/CMakeLists.txt").read_text()
    body = re.search(r"ascendc_library\(oscar_ascend_kernels SHARED\s+(.*?)\)",
                     cmake, re.DOTALL).group(1)
    return re.findall(r"kernels/[\w]+\.cpp", body)


def test_production_cmake_compiles_only_one_definition_of_each_unchanged_abi():
    sources = compiled_sources()
    assert len(sources) == len(set(sources))
    texts = {name: (ROOT / "csrc" / name).read_text() for name in sources}
    for entry in generated.KERNELS:
        active = entry["active"].removeprefix("csrc/")
        mother = entry["mother"].removeprefix("csrc/")
        assert active in sources and mother not in sources
        symbols = re.findall(r'extern "C" __global__ __aicore__ void (\w+)\(', texts[active])
        assert symbols
        for symbol in symbols:
            definition = 'extern "C" __global__ __aicore__ void ' + symbol + '('
            assert sum(text.count(definition) for text in texts.values()) == 1


@pytest.mark.parametrize("name", INPUTS)
def test_generation_gate_runs_before_any_completed_build_reuse(compiler, monkeypatch, name):
    compiler.build()
    compiler.commands.clear()
    state = compiler.build_dir / "oscar_build_signature.json"
    prior_state = state.read_bytes()
    path = compiler.root / name
    path.write_bytes(path.read_bytes() + b"\n// unexpected change\n")
    damaged = path.read_bytes()
    monkeypatch.setattr(build_ops, "reusable_build", lambda *_:
                        pytest.fail("must validate derivatives before examining a cached build"))
    with pytest.raises(ValueError, match="changed|drift"):
        compiler.build()
    assert compiler.commands == []
    assert path.read_bytes() == damaged and state.read_bytes() == prior_state


def test_previous_mother_only_artifact_cannot_reuse_new_active_signature(compiler):
    built = compiler.build()
    state = compiler.build_dir / "oscar_build_signature.json"
    previous = json.loads(state.read_text())
    source = previous["configuration"]["source"]
    for name in (generated.MANIFEST, *(entry["active"] for entry in generated.KERNELS)):
        del source[name.removeprefix("csrc/")]
    previous["signature"] = sha(json.dumps(previous["configuration"], sort_keys=True).encode())
    assert previous["signature"] != built["signature"]
    state.write_text(json.dumps(previous))
    stale = compiler.build_dir / "mother-only.o"
    stale.write_bytes(b"old source generation")
    compiler.commands.clear()
    rebuilt = compiler.build()
    assert not rebuilt["reused"] and rebuilt["signature"] == built["signature"]
    assert [name for name, _ in compiler.commands] == ["configure-1", "compile-1"]
    assert not stale.exists()


def test_changed_generator_identity_is_bound_into_the_build_signature(compiler):
    built = compiler.build()
    tool = compiler.root / generated.TOOL_PATHS[0]
    tool.write_bytes(tool.read_bytes() + b"\n# audited generator revision\n")
    with pytest.raises(ValueError, match="manifest"):
        compiler.build()
    # Explicit generation is a separate developer action, never the build.
    for name, data in generated.outputs(compiler.root).items():
        (compiler.root / name).write_bytes(data)
    compiler.commands.clear()
    rebuilt = compiler.build()
    assert rebuilt["signature"] != built["signature"] and not rebuilt["reused"]
    assert [name for name, _ in compiler.commands] == ["configure-1", "compile-1"]


@pytest.mark.parametrize("probe", [probe_official_attention, probe_aligned_attention, probe_prefix_cache])
def test_probe_evidence_binds_complete_mothers_active_sources_and_transforms(probe):
    bindings = probe.source_binding()
    for name in INPUTS:
        assert bindings[name] == sha((ROOT / name).read_bytes())


@pytest.mark.parametrize("later_draft", [False, True])
def test_real_candidate_entrypoint_builds_and_routes_the_same_abi_to_active_source(capsys, later_draft):
    assert install_serve.main(["--variant", "candidate", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    command = dict(plan["stages"])["build-ops"]
    assert command[1:3] == ["-m", "tools.build_ops"]
    config = variant_config(json.loads((ROOT / "configs/target.json").read_text()), "candidate")
    names = ("fast_unpack", "mixed_cv", "striped_cache", "decode_bundle",
             "c4_member_pipeline", "mdl_decode", "c16_member_pipeline", "hybrid_l1_decode",
             "c16_l1_reuse", "bf16_attention")
    selected = select_cv_op(16, 6, 1, 128, 1 if later_draft else 4,
                            q1_draft=later_draft,
                            **{name: config.get("experimental_" + name, False) for name in names})
    assert config["experimental_official_math"] is True
    operator = OFFICIAL_CV_MAP[selected]
    assert operator == probe_official_attention.OPERATORS["q1" if later_draft else "q4"]
    launch = operator.removesuffix("_out") + "_launch"
    active = (ROOT / generated.KERNELS[0]["active"]).read_text()
    assert launch in active and launch in (ROOT / "csrc/official_attention_bindings.cpp").read_text()
    assert generated.KERNELS[0]["active"].removeprefix("csrc/") in compiled_sources()
    assert probe_official_attention.WRITER == "store_aligned_int2_out"
    assert generated.KERNELS[1]["active"].removeprefix("csrc/") in compiled_sources()
