# 档案 #27/#28/#69/#77/#110/#140；H01/H02/H05/H09：扫描可执行语法，
# 只允许统一 helper 内已验证为 CPU 的原生整数镜像转列表，不放行 NPU 回读。
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_production_cannot_import_test_oracle():
    for path in (ROOT / "oscar_ascend").rglob("*.py"):
        if path.name == "reference.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").endswith("reference"), path
            if isinstance(node, ast.Import):
                assert all(not item.name.endswith("reference") for item in node.names), path


def test_no_silent_exception_swallowing():
    for folder in ("oscar_ascend", "tools"):
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    assert not all(isinstance(statement, ast.Pass) for statement in node.body), path


def test_online_hooks_do_not_copy_device_data_to_host():
    for path in (ROOT / "oscar_ascend/integration").rglob("*.py"):
        tree = ast.parse(path.read_text())
        cpu_mirror = None
        if path.name == "cpu_metadata.py":
            cpu_mirror = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                               and node.name == "cpu_int_mirror_values"), None)
            validator = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                              and node.name == "validate_cpu_int_mirror"), None)
            assert cpu_mirror is not None
            assert validator is not None
            # The sole conversion must consume the checked return value;
            # adding helpers or allowing another file is not a whitelist.
            checked = cpu_mirror.body[1]
            assert isinstance(checked, ast.Assign) and len(checked.targets) == 1
            assert isinstance(checked.targets[0], ast.Name) and checked.targets[0].id == "starts"
            assert isinstance(checked.value, ast.Call) and isinstance(checked.value.func, ast.Name)
            assert checked.value.func.id == "validate_cpu_int_mirror"
            assert isinstance(checked.value.args[0], ast.Name) and checked.value.args[0].id == "starts"
            guard = validator.body[1]
            assert isinstance(guard, ast.If) and isinstance(guard.test, ast.BoolOp)
            assert isinstance(guard.test.op, ast.Or)
            assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Raise)
            # Non-CPU tensors must be rejected, not copied or inspected.
            assert any(isinstance(node, ast.Compare)
                       and isinstance(node.left, ast.Attribute) and node.left.attr == "type"
                       and isinstance(node.left.value, ast.Attribute)
                       and node.left.value.attr == "device"
                       and isinstance(node.left.value.value, ast.Name)
                       and node.left.value.value.id == "starts"
                       and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq)
                       and any(isinstance(value, ast.Constant) and value.value == "cpu"
                               for value in node.comparators)
                       for node in guard.test.values)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "tolist":
                    assert cpu_mirror is not None and node in ast.walk(cpu_mirror), path
                    value = node.func.value
                    assert isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)
                    assert value.value.id == "starts", path
                else:
                    assert node.func.attr not in {"cpu", "numpy", "item"}, path
                if node.func.attr == "to":
                    values = [a.value for a in node.args if isinstance(a, ast.Constant)]
                    assert "cpu" not in values, path


def test_shell_scripts_do_not_copy_or_modify_native_sources():
    for path in (ROOT / "scripts").glob("*.sh"):
        source = path.read_text()
        assert "set -euo pipefail" in source
        assert not re.search(r"(?:^|[;\s])cp(?:\s|$)", source), path
        assert not any("<<" in line and "|" in line for line in source.splitlines()), path


def test_source_headers_trace_archive():
    paths = []
    for folder in ("oscar_ascend", "tools", "scripts", "tests", "benchmarks", "csrc"):
        paths += [p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".cpp", ".h", ".cmake", ".sh"}]
    paths += [ROOT / "csrc/CMakeLists.txt"]
    for path in paths:
        header = "\n".join(path.read_text().splitlines()[:18])
        assert any(word in header for word in ("档案", "Archive", "archive")), f"missing historical fault trace: {path}"
