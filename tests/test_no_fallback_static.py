# 档案 #27/#28/#69/#77/#110/#140；H01/H02/H05/H09：扫描可执行语法，
# 只允许已验证为 CPU 的原生 query_start_loc_cpu 镜像转列表，不放行 NPU 回读。
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
        if path.name == "current_attention.py":
            cpu_mirror = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                               and node.name == "current_cumulative_lengths"), None)
            assert cpu_mirror is not None
            # Verify the explicit `starts.device.type != "cpu"` rejection
            # remains in the same helper as the one allowed CPU conversion.
            assert any(isinstance(node, ast.Compare)
                       and isinstance(node.left, ast.Attribute) and node.left.attr == "type"
                       and isinstance(node.left.value, ast.Attribute)
                       and node.left.value.attr == "device"
                       and isinstance(node.left.value.value, ast.Name)
                       and node.left.value.value.id == "starts"
                       and any(isinstance(value, ast.Constant) and value.value == "cpu"
                               for value in node.comparators)
                       for node in ast.walk(cpu_mirror))
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
