# 档案 G7/G8/#79–#85/#98–#100：完整同配置产物才可复用；漂移或损坏必须清建。
"""Exercise build-cache decisions with fake compiler output, never an NPU."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from oscar_ascend.ops.contracts import ABI_VERSION, SOURCE_CAPABILITIES
from oscar_ascend.ops.loader import validate_build_artifacts
from tools import build_ops


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    source = root / "csrc/bindings.cpp"
    source.parent.mkdir()
    source.write_text("original source\n")
    cann = root / "cann"
    (cann / "include").mkdir(parents=True)
    (cann / "tools/tikcpp/ascendc_kernel_cmake").mkdir(parents=True)
    (cann / "version.info").write_text("CANN test version\n")
    npu = root / "torch_npu"
    header = npu / "include/torch_npu/csrc/core/npu/NPUStream.h"
    header.parent.mkdir(parents=True)
    header.write_text("test header\n")
    monkeypatch.setattr(build_ops, "ROOT", root)
    monkeypatch.setenv("ASCEND_CANN_PACKAGE_PATH", str(cann))
    monkeypatch.setattr(build_ops.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(npu / "__init__.py")))
    versions = {"torch": "2.12.0", "torch_npu": "2.12.0", "pybind11": "3.1.0"}
    monkeypatch.setattr(build_ops.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(build_ops.shutil, "which", lambda _: "/test/bin/cmake")
    commands = []
    build_dir = root / "build/ascendc"

    def run_phase(name, command, **kwargs):
        commands.append((name, command))
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        manifest_path = build_dir / "build_manifest.json"
        if name.startswith("configure-"):
            signature = next(x.split("=", 1)[1] for x in command if x.startswith("-DOSCAR_BUILD_SIGNATURE="))
            manifest_path.write_text(json.dumps({
                "abi_version": ABI_VERSION, "execution_interface": "direct_launch",
                "soc": kwargs["env"]["SOC_VERSION"],
                "extension": str(build_dir / "_oscar_ascend_ops.so"),
                "kernel_library": str(build_dir / f"liboscar_ascend_kernels_{signature[:16]}.so"),
                "source_capabilities": sorted(SOURCE_CAPABILITIES),
            }))
        else:
            manifest = json.loads(manifest_path.read_text())
            for kind in ("extension", "kernel_library"):
                Path(manifest[kind]).write_bytes(f"fake compiled {kind}".encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build_ops, "run_phase", run_phase)
    return SimpleNamespace(root=root, source=source, cann=cann, versions=versions,
                           commands=commands, build_dir=build_dir,
                           build=lambda: build_ops.build(root / "logs", 30, "ascend910b4"))


def test_same_completed_build_skips_cmake_and_preserves_evidence(compiler, monkeypatch):
    built = compiler.build()
    assert not built["reused"]
    assert [x[0] for x in compiler.commands] == ["configure-1", "compile-1"]
    state_path = compiler.build_dir / "oscar_build_signature.json"
    prior_state = state_path.read_bytes()
    compiler.commands.clear()
    # A completed cache does not need to invoke or locate CMake again.
    monkeypatch.setattr(build_ops.shutil, "which", lambda _: None)
    reused = compiler.build()
    assert compiler.commands == []
    assert reused["reused"] and reused["signature"] == built["signature"]
    assert reused["device_completion"] == "not_run"
    assert state_path.read_bytes() == prior_state
    assert json.loads((compiler.root / "reports/build.json").read_text()) == reused
    validate_build_artifacts(compiler.build_dir / "build_manifest.json")


@pytest.mark.parametrize("change", ["source", "cann-version", "package-version", "compiler-flags"])
def test_changed_build_configuration_cleans_before_rebuild(compiler, monkeypatch, change):
    built = compiler.build()
    stale_object = compiler.build_dir / "old-partial-device-object.o"
    stale_object.write_bytes(b"must not be passed to the linker")
    if change == "source":
        compiler.source.write_text("modified source\n")
    elif change == "cann-version":
        (compiler.cann / "version.info").write_text("new CANN test version\n")
    elif change == "package-version":
        compiler.versions["torch_npu"] = "2.12.1"
    else:
        monkeypatch.setenv("CXXFLAGS", "-DOSCAR_TEST_CHANGED=1")
    compiler.commands.clear()
    rebuilt = compiler.build()
    assert not rebuilt["reused"] and rebuilt["signature"] != built["signature"]
    assert [x[0] for x in compiler.commands] == ["configure-1", "compile-1"]
    assert not stale_object.exists()
    validate_build_artifacts(compiler.build_dir / "build_manifest.json")


@pytest.mark.parametrize("damage", ["extension", "kernel_library", "missing-kernel", "extra-kernel",
                                   "invalid-state", "missing-manifest", "invalid-manifest", "missing-capability"])
def test_incomplete_or_corrupted_cache_is_cleanly_rebuilt(compiler, damage):
    built = compiler.build()
    stale_object = compiler.build_dir / "old-partial-device-object.o"
    stale_object.write_bytes(b"must not survive a cache failure")
    manifest_path = compiler.build_dir / "build_manifest.json"
    if damage in ("extension", "kernel_library"):
        artifact = Path(built[damage])
        old_stat = artifact.stat()
        artifact.write_bytes(b"corrupted with unchanged mtime")
        os.utime(artifact, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    elif damage == "missing-kernel":
        Path(built["kernel_library"]).unlink()
    elif damage == "extra-kernel":
        (compiler.build_dir / "liboscar_ascend_kernels_stale.so").write_bytes(b"older output")
    elif damage == "invalid-state":
        (compiler.build_dir / "oscar_build_signature.json").write_text("{truncated")
    elif damage == "missing-manifest":
        manifest_path.unlink()
    elif damage == "invalid-manifest":
        manifest_path.write_text("{truncated")
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["source_capabilities"].remove("attention_cv_out")
        manifest_path.write_text(json.dumps(manifest))
    compiler.commands.clear()
    rebuilt = compiler.build()
    assert not rebuilt["reused"] and rebuilt["signature"] == built["signature"]
    assert [x[0] for x in compiler.commands] == ["configure-1", "compile-1"]
    assert not stale_object.exists()
    assert rebuilt["device_completion"] == "not_run"
    validate_build_artifacts(manifest_path)
