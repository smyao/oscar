# Archive G7/G8/#79-85/#98-109/#164: isolated signed build/cache host contracts.
# Fake compiler outputs establish orchestration only, not CANN/NPU completion.
import json
from pathlib import Path

import pytest

from tools import build_c16_timer as builder
from tools.phase import PhaseResult


def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    monkeypatch.setattr(builder, "ROOT", root)
    sources = [builder.SOURCE_RELATIVE / name for name in ("CMakeLists.txt", "generate.py", "timer.h")]
    sources += [Path("csrc/kernels/attention_cv_aligned_cluster16.cpp"), Path("csrc/kernels/oscar_common.h"),
                Path("csrc/aligned_attention_bindings.cpp"), Path("oscar_ascend/c16_timer_schema.py"),
                builder.CLOCK_FILE, Path("tools/build_c16_timer.py")]
    for source in sources:
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("host source fixture " + str(source))
    production = root / "build/ascendc/production.so"
    production.parent.mkdir(parents=True)
    production.write_bytes(b"production library must stay byte identical")
    return root


def fake_compiler(root, monkeypatch, *, fail_phase=None, missing_artifact=False, mutate_source=False):
    state = {"phases": [], "signature": None, "cmake_dir": None}
    toolchain = {"soc": "ascend910b4", "cann": "/fixture/cann/9.1",
        "cann_versions": {"version": "hash"}, "package_paths": {"torch": "/fixture/torch", "torch_npu": "/fixture/torch_npu"},
        "package_versions": {"torch": "2.7", "torch_npu": "2.7", "pybind11": "3"},
        "python": "/fixture/python", "python_version": "3.12", "cmake": "/fixture/cmake"}
    monkeypatch.setattr(builder, "discover_toolchain", lambda soc, env: (toolchain, {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}))
    def phase(name, command, **kwargs):
        state["phases"].append(name)
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        log = kwargs["log_dir"] / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("fake compiler receipt\n")
        if name == fail_phase:
            return PhaseResult(name, command, 17, 0, False, False, str(log), True)
        if name == "c16-generate":
            directory = Path(command[command.index("--output-dir") + 1])
            assert not directory.is_relative_to(root / "csrc")
            directory.mkdir(parents=True)
            generated = {}
            for key, filename in (("kernel", "c16_timer_kernel.cpp"), ("bindings", "bindings.cpp"), ("header", "launch.h")):
                path = directory / filename
                path.write_text("generated " + key)
                generated[key] = str(path)
            (directory / "generation.json").write_text(json.dumps(generated))
        elif name == "c16-configure":
            state["cmake_dir"] = Path(command[command.index("-B") + 1])
            state["signature"] = next(value.split("=", 1)[1] for value in command if value.startswith("-DOSCAR_BUILD_SIGNATURE="))
            assert "-DCMAKE_BUILD_TYPE=Release" in command
            assert any(value.startswith("-DOSCAR_TIMER_KERNEL_SOURCE=") for value in command)
            state["cmake_dir"].mkdir(parents=True)
        elif name == "c16-compile":
            base = state["cmake_dir"]
            signature = state["signature"][:16]
            library = base / f"liboscar_c16_timer_{signature}.so"
            kernel = base / "lib" / f"liboscar_c16_timer_kernels_{signature}.so"
            kernel.parent.mkdir()
            if not missing_artifact:
                library.write_bytes(b"compiled binding fixture")
            kernel.write_bytes(b"compiled kernel fixture")
            (base / "build_manifest.json").write_text(json.dumps({"library": str(library), "kernel_library": str(kernel)}))
            if mutate_source:
                (root / "csrc/kernels/oscar_common.h").write_text("changed during build")
        return PhaseResult(name, command, 0, 0, False, False, str(log), True)
    monkeypatch.setattr(builder, "run_phase", phase)
    return state, toolchain


def test_plan_is_read_only_without_toolchain_import_or_generation(tmp_path, monkeypatch, capsys):
    root = checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(builder, "discover_toolchain", lambda *a: pytest.fail("plan inspected target toolchain"))
    monkeypatch.setattr(builder, "run_phase", lambda *a, **k: pytest.fail("plan ran subprocess"))
    logs, output, base = root / "logs", root / "receipt.json", root / "build/c16_timer"
    assert builder.main(["--plan", "--output", str(output), "--log-dir", str(logs), "--output-dir", str(base)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["device_access"] is False and str(builder.CLOCK_FILE) in plan["source_sha256"]
    assert not output.exists() and not logs.exists() and not base.exists()


def test_required_inputs_are_deliverable_without_ignored_reference_checkout():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    tracked = set(subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True,
                                capture_output=True).stdout.decode().split("\0"))
    new_delivery = {
        "tools/build_c16_timer.py", "oscar_ascend/c16_timer_schema.py",
        "csrc/experiments/c16_timer_20261019/CMakeLists.txt",
        "csrc/experiments/c16_timer_20261019/generate.py",
        "csrc/experiments/c16_timer_20261019/timer.h",
        "csrc/experiments/c16_timer_20261019/clock_contract.json"}
    required = {str(path) for path in builder.required_sources()}
    assert required <= tracked | new_delivery
    assert all((root / path).is_file() for path in required)
    assert not any(path.startswith("references/") for path in required)
    contract = json.loads((root / builder.CLOCK_FILE).read_text())
    assert contract["clock_hz"] == builder.CLOCK_HZ == contract["ticks_per_microsecond"] * 1_000_000
    assert contract["soc"] == "ascend910b4"
    assert len(contract["source_file_sha256"]) == 64
    assert [part["line_start"] for part in contract["source_excerpts"]] == [25, 354]


def test_isolated_build_receipt_and_idempotent_signed_reuse(tmp_path, monkeypatch):
    root = checkout(tmp_path, monkeypatch)
    state, _ = fake_compiler(root, monkeypatch)
    sources = builder.file_fingerprint(root / "csrc")
    original = (root / "build/ascendc/production.so").read_bytes()
    options = dict(output=root / "receipt.json", log_dir=root / "logs", output_dir=root / "build/c16_timer")
    first = builder.build(**options)
    assert first["status"] == "built" and first["reused"] is False
    assert first["library_sha256"] == builder._digest(first["library"])
    assert first["kernel_library_sha256"] == builder._digest(first["kernel_library"])
    assert first["phase_schema_id"] == builder.SCHEMA_ID
    assert str(builder.CLOCK_FILE) in first["source_sha256"]
    assert builder.file_fingerprint(root / "csrc") == sources
    assert (root / "build/ascendc/production.so").read_bytes() == original
    second = builder.build(**options)
    assert second["status"] == "built" and second["reused"] is True
    assert second["library"] == first["library"]
    assert state["phases"] == ["c16-generate", "c16-configure", "c16-compile"]


@pytest.mark.parametrize("change", ["library", "kernel_library", "clock", "python_path", "npu_path", "cann_version", "generated"])
def test_changed_artifact_source_or_toolchain_forces_new_separate_attempt(tmp_path, monkeypatch, change):
    root = checkout(tmp_path, monkeypatch)
    state, toolchain = fake_compiler(root, monkeypatch)
    options = dict(output=root / "receipt.json", log_dir=root / "logs", output_dir=root / "build/c16_timer")
    first = builder.build(**options)
    if change in {"library", "kernel_library"}:
        Path(first[change]).write_bytes(b"damaged artifact")
    elif change == "clock":
        (root / builder.CLOCK_FILE).write_text("changed timebase source")
    elif change == "generated":
        Path(next(iter(first["generated_sha256"]))).write_text("changed generated source")
    elif change == "npu_path":
        toolchain["package_paths"]["torch_npu"] = "/different/torch_npu"
    elif change == "cann_version":
        toolchain["cann_versions"]["version"] = "changed CANN"
    else:
        toolchain["python"] = "/different/python"
    second = builder.build(**options)
    assert second["reused"] is False and second["library"] != first["library"]
    assert len(state["phases"]) == 6
    assert Path(first["kernel_library"]).exists()  # old attempt is preserved, never removed blindly


@pytest.mark.parametrize("subdir", ["csrc/generated", "references/modified", "build/ascendc"])
def test_output_cannot_change_production_source_reference_or_build(tmp_path, monkeypatch, subdir):
    root = checkout(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="must not touch"):
        builder.build(output=root / "receipt.json", log_dir=root / "logs", output_dir=root / subdir)


def test_failed_compile_returns_original_code_and_no_success_receipt(tmp_path, monkeypatch):
    root = checkout(tmp_path, monkeypatch)
    fake_compiler(root, monkeypatch, fail_phase="c16-compile")
    output = root / "receipt.json"
    assert builder.main(["--output", str(output), "--log-dir", str(root / "logs"),
                         "--output-dir", str(root / "build/c16_timer")]) == 17
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["returncode"] == 17
    assert not list((root / "build/c16_timer").rglob("completed.json"))


@pytest.mark.parametrize("mode", ["missing_artifact", "mutate_source"])
def test_compiler_success_is_not_sufficient_for_valid_artifact(tmp_path, monkeypatch, mode):
    root = checkout(tmp_path, monkeypatch)
    fake_compiler(root, monkeypatch, **{mode: True})
    with pytest.raises(builder.C16BuildError):
        builder.build(output=root / "receipt.json", log_dir=root / "logs", output_dir=root / "build/c16_timer")
    assert not (root / "receipt.json").exists()
    assert not list((root / "build/c16_timer").rglob("completed.json"))
