# Archive G18/#85/#98: compiled files are not load evidence; preload the exact
# validated dependency before importing its extension, and reject stale files.
"""Exercise native-loader ordering and failure boundaries without an NPU."""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import uuid
import weakref

import pytest

from oscar_ascend.ops import loader, meta
from oscar_ascend.ops.contracts import ABI_VERSION, SOURCE_CAPABILITIES
from tools.environment import file_fingerprint


def _write_build(root, source, *, suffix="0123456789abcdef"):
    root.mkdir(parents=True, exist_ok=True)
    extension = root / "_oscar_ascend_ops.so"
    kernel = root / f"liboscar_ascend_kernels_{suffix}.so"
    extension.write_bytes(b"fixture extension " + suffix.encode())
    kernel.write_bytes(b"fixture kernel " + suffix.encode())
    configuration = {"source": file_fingerprint(source), "soc": "ascend910b4"}
    signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    state = {"build": "passed", "configuration": configuration, "signature": signature,
             "extension": str(extension), "kernel_library": str(kernel),
             "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (extension, kernel)}}
    manifest = {"abi_version": ABI_VERSION, "execution_interface": "direct_launch",
                "source_capabilities": sorted(SOURCE_CAPABILITIES),
                **{key: value for key, value in state.items() if key != "configuration"}}
    (root / "oscar_build_signature.json").write_text(json.dumps(state))
    path = root / "build_manifest.json"
    path.write_text(json.dumps(manifest))
    return SimpleNamespace(manifest=path, extension=extension, kernel=kernel)


@pytest.fixture
def native_loader(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    source = root / "csrc"
    source.mkdir()
    (source / "kernel.cpp").write_text("source used to compile fixture\n")
    built = _write_build(root / "build/ascendc", source)
    monkeypatch.setattr(loader, "__file__", str(root / "oscar_ascend/ops/loader.py"))
    monkeypatch.setattr(loader, "_loaded", None)
    monkeypatch.setattr(loader, "_loaded_path", None)
    monkeypatch.setattr(loader, "_kernel_library", None, raising=False)
    monkeypatch.setitem(sys.modules, "_oscar_ascend_ops", ModuleType("preexisting_test_sentinel"))
    events = []
    handle_refs = []
    module = ModuleType("_oscar_ascend_ops")
    module.abi_version = lambda: ABI_VERSION
    module.capabilities = lambda: sorted(SOURCE_CAPABILITIES)

    class KernelHandle:
        pass

    def preload(path, *, mode):
        events.append(("kernel", path, mode))
        handle = KernelHandle()
        handle_refs.append(weakref.ref(handle))
        return handle

    real_import = loader.importlib.import_module

    def import_module(name, package=None):
        if name == "torch_npu":
            events.append(("torch_npu",))
            return ModuleType("torch_npu")
        return real_import(name, package)

    def spec_from_file_location(name, path):
        events.append(("extension_spec", str(path)))
        assert name == "_oscar_ascend_ops"
        return SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: events.append(("extension_exec",))))

    def module_from_spec(spec):
        # ExtensionFileLoader's create_module performs dlopen here, before
        # exec_module; loading only before exec_module would still be too late.
        events.append(("extension_create",))
        return module

    import ctypes
    monkeypatch.setattr(ctypes, "CDLL", preload)
    monkeypatch.setattr(loader.importlib, "import_module", import_module)
    monkeypatch.setattr(loader.importlib.util, "spec_from_file_location", spec_from_file_location)
    monkeypatch.setattr(loader.importlib.util, "module_from_spec", module_from_spec)
    monkeypatch.setattr(meta, "register_meta", lambda: events.append(("meta",)))
    return SimpleNamespace(**vars(built), root=root, source=source, events=events,
                           handle_refs=handle_refs, module=module, ctypes=ctypes)


def test_preloads_validated_absolute_kernel_before_extension_dlopen(native_loader, monkeypatch):
    fixture = native_loader
    unrelated = fixture.root / "old-project"
    unrelated.mkdir()
    (unrelated / fixture.kernel.name).write_bytes(b"unrelated stale dependency")
    monkeypatch.chdir(unrelated)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(unrelated))
    assert loader.load_extension(fixture.manifest) is fixture.module
    assert [event[0] for event in fixture.events] == [
        "torch_npu", "kernel", "extension_spec", "extension_create", "extension_exec", "meta"]
    assert fixture.events[1] == ("kernel", str(fixture.kernel), os.RTLD_NOW | os.RTLD_LOCAL)
    assert fixture.events[2] == ("extension_spec", str(fixture.extension))
    assert os.environ["LD_LIBRARY_PATH"] == str(unrelated)


def test_preloaded_handle_lives_with_cached_extension(native_loader):
    fixture = native_loader
    assert loader.load_extension(fixture.manifest) is fixture.module
    gc.collect()
    assert len(fixture.handle_refs) == 1
    assert fixture.handle_refs[0]() is not None
    assert loader._kernel_library is fixture.handle_refs[0]()
    initial_events = list(fixture.events)
    assert loader.load_extension(fixture.manifest) is fixture.module
    assert fixture.events == initial_events
    assert loader._kernel_library is fixture.handle_refs[0]()


def test_preload_failure_exposes_exact_dependency_and_stops_extension_import(native_loader, monkeypatch):
    fixture = native_loader
    failure = OSError("fixture dependent driver symbol missing")

    def broken(path, *, mode):
        fixture.events.append(("kernel", path, mode))
        raise failure

    monkeypatch.setattr(fixture.ctypes, "CDLL", broken)
    with pytest.raises(loader.OperatorUnavailable) as caught:
        loader.load_extension(fixture.manifest)
    assert caught.value.__cause__ is failure
    assert str(fixture.kernel) in str(caught.value)
    assert "driver symbol missing" in str(caught.value)
    assert [event[0] for event in fixture.events] == ["torch_npu", "kernel"]
    assert loader._loaded is None and loader._loaded_path is None


def test_extension_import_failure_does_not_publish_loaded_success(native_loader, monkeypatch):
    fixture = native_loader
    failure = ImportError("fixture extension symbol missing")

    def broken(spec):
        fixture.events.append(("extension_create",))
        raise failure

    monkeypatch.setattr(loader.importlib.util, "module_from_spec", broken)
    with pytest.raises(ImportError) as caught:
        loader.load_extension(fixture.manifest)
    assert caught.value is failure
    assert [event[0] for event in fixture.events] == ["torch_npu", "kernel", "extension_spec", "extension_create"]
    assert loader._loaded is None and loader._loaded_path is None
    assert sys.modules["_oscar_ascend_ops"].__name__ == "preexisting_test_sentinel"


@pytest.mark.parametrize("artifact", ["extension", "kernel"])
def test_hash_mismatch_fails_before_backend_import_or_preload(native_loader, artifact):
    fixture = native_loader
    path = getattr(fixture, artifact)
    before = path.stat()
    path.write_bytes(b"tampered artifact")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(loader.OperatorUnavailable, match="SHA256 mismatch"):
        loader.load_extension(fixture.manifest)
    assert fixture.events == []


def test_missing_kernel_fails_before_backend_import_or_preload(native_loader):
    fixture = native_loader
    fixture.kernel.unlink()
    with pytest.raises(loader.OperatorUnavailable, match="artifact missing"):
        loader.load_extension(fixture.manifest)
    assert fixture.events == []


def test_changed_source_fails_before_backend_import_or_preload(native_loader):
    fixture = native_loader
    (fixture.source / "kernel.cpp").write_text("uncompiled modified source\n")
    with pytest.raises(loader.OperatorUnavailable, match="source changed"):
        loader.load_extension(fixture.manifest)
    assert fixture.events == []


def test_second_build_in_same_process_cannot_replace_loaded_device_code(native_loader):
    fixture = native_loader
    loader.load_extension(fixture.manifest)
    second = _write_build(fixture.root / "second-build", fixture.source, suffix="fedcba9876543210")
    initial_events = list(fixture.events)
    with pytest.raises(loader.OperatorUnavailable, match="restart"):
        loader.load_extension(second.manifest)
    assert fixture.events == initial_events
    assert loader._loaded is fixture.module


@pytest.mark.skipif(sys.platform != "linux", reason="ELF DT_NEEDED/SONAME regression requires Linux")
def test_real_elf_dependency_without_rpath_loads_after_exact_kernel_preload(tmp_path, monkeypatch):
    # This tests the OS dynamic linker, without Torch, CANN or Python headers.
    # Only extension registration is replaced; its module-creation seam really
    # dlopens the dependent ELF, which must resolve the preloaded SONAME.
    import ctypes
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        pytest.skip("real ELF regression requires a C compiler")
    root = tmp_path.resolve()
    source = root / "csrc"
    source.mkdir()
    kernel_source = source / "kernel.c"
    extension_source = source / "extension.c"
    kernel_source.write_text("int oscar_link_probe(void) { return 73; }\n")
    extension_source.write_text("extern int oscar_link_probe(void);\nint oscar_link_result(void) { return oscar_link_probe(); }\n")
    built = _write_build(root / "build/ascendc", source, suffix=uuid.uuid4().hex[:16])
    subprocess.run([compiler, "-shared", "-fPIC", str(kernel_source),
                    "-Wl,-soname," + built.kernel.name, "-o", str(built.kernel)], check=True, timeout=30)
    subprocess.run([compiler, "-shared", "-fPIC", str(extension_source), "-L" + str(built.kernel.parent),
                    "-Wl,--no-as-needed", "-l:" + built.kernel.name, "-o", str(built.extension)], check=True, timeout=30)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (built.kernel, built.extension)}
    for path in (built.manifest, built.manifest.parent / "oscar_build_signature.json"):
        evidence = json.loads(path.read_text())
        evidence["sha256"] = hashes
        path.write_text(json.dumps(evidence))

    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    unprepared = subprocess.run([sys.executable, "-c",
        "import ctypes,os,sys; ctypes.CDLL(sys.argv[1],mode=os.RTLD_NOW|os.RTLD_LOCAL)",
        str(built.extension)], env=environment, cwd=root, capture_output=True, text=True, timeout=15)
    assert unprepared.returncode != 0
    assert built.kernel.name in unprepared.stderr and "cannot open shared object file" in unprepared.stderr

    monkeypatch.setattr(loader, "__file__", str(root / "oscar_ascend/ops/loader.py"))
    monkeypatch.setattr(loader, "_loaded", None)
    monkeypatch.setattr(loader, "_loaded_path", None)
    monkeypatch.setattr(loader, "_kernel_library", None, raising=False)
    monkeypatch.setitem(sys.modules, "_oscar_ascend_ops", ModuleType("elf_test_sentinel"))
    real_import = loader.importlib.import_module
    monkeypatch.setattr(loader.importlib, "import_module",
        lambda name, package=None: ModuleType(name) if name == "torch_npu" else real_import(name, package))
    monkeypatch.setattr(loader.importlib.util, "spec_from_file_location",
        lambda name, path: SimpleNamespace(origin=str(path), loader=SimpleNamespace(exec_module=lambda module: None)))

    def create_module(spec):
        dependency = ctypes.CDLL(spec.origin, mode=os.RTLD_NOW | os.RTLD_LOCAL)
        assert dependency.oscar_link_result() == 73
        return SimpleNamespace(abi_version=lambda: ABI_VERSION,
                               capabilities=lambda: sorted(SOURCE_CAPABILITIES), dependency=dependency)

    monkeypatch.setattr(loader.importlib.util, "module_from_spec", create_module)
    monkeypatch.setattr(meta, "register_meta", lambda: None)
    loaded = loader.load_extension(built.manifest)
    assert loaded.dependency.oscar_link_result() == 73
    assert loader._kernel_library._name == str(built.kernel)
