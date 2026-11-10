"""Archive #27/#29/#75/#94/#125: preserve the requested SDK launch contract.

Host shell tests replace only the four SDK source paths with temporary stubs.
No installed SDK, MetaService, model, package installer or NPU is invoked.
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SDK_PATHS = (
    "/usr/local/Ascend/ascend-toolkit/set_env.sh",
    "/usr/local/Ascend/nnal/atb/set_env.sh",
    "/usr/local/memfabric_hybrid/set_env.sh",
    "/usr/local/memcache_hybrid/set_env.sh",
)
SDK_NAMES = ("cann", "atb", "memfabric", "memcache")
EXPECTED_ENV = {
    "PYTHONHASHSEED": "0",
    "MMC_LOCAL_SERVICE_BACKEND_ID": "1",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "MMC_LOCAL_CONFIG_PATH": "/usr/local/memcache_hybrid/latest/config/mmc-local.conf",
    "MMC_META_CONFIG_PATH": "/usr/local/memcache_hybrid/latest/config/mmc-meta.conf",
    "OSCAR_OFFLOAD_SDK_SOURCE": "installed",
}


@pytest.fixture
def launch_tree(tmp_path):
    """Use the real scripts in an isolated tree with observable SDK/install stubs."""
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    helper_text = (ROOT / "scripts/env_node93_memcache.sh").read_text()
    sdk_files = []
    for name, source_path in zip(SDK_NAMES, SDK_PATHS):
        stub = tmp_path / f"{name}-set_env.sh"
        assert source_path in helper_text, f"missing user-selected SDK source: {source_path}"
        helper_text = helper_text.replace(source_path, str(stub))
        sdk_files.append(stub)
    helper = scripts / "env_node93_memcache.sh"
    helper.write_text(helper_text)
    wrapper = scripts / "start_node93_oscar.sh"
    wrapper.write_text((ROOT / "scripts/start_node93_oscar.sh").read_text())
    capture = tmp_path / "capture.py"
    capture.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "keys = json.loads(os.environ['TEST_ENV_KEYS'])\n"
        "values = {key: value for key, value in os.environ.items() "
        "if key in keys or key.startswith(('MMC_', 'TEST_SDK_'))}\n"
        "Path(os.environ['TEST_LAUNCH_RESULT']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'env': values, 'pid': os.getpid(), 'cwd': os.getcwd()}))\n"
        "raise SystemExit(int(os.environ.get('TEST_INSTALL_RETURN_CODE', '0')))\n"
    )
    (scripts / "install_serve.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(capture))} \"$@\"\n"
    )
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith(("MMC_", "TEST_SDK_")):
            env.pop(name)
    env.update({name: "inherited-wrong-value" for name in EXPECTED_ENV})
    trace = tmp_path / "sdk-order.tsv"
    output = tmp_path / "launch.json"
    env.update(TEST_SDK_TRACE=str(trace), TEST_LAUNCH_RESULT=str(output),
               TEST_ENV_KEYS=json.dumps(list(EXPECTED_ENV)))
    env.pop("OSCAR_TEST_SDK_OPTIONAL", None)
    # Environment loading has no host/configuration or MetaService phase. Any
    # such command is a test failure, never a real operation on the test host.
    guards = tmp_path / "guard-bin"
    guards.mkdir()
    for command in ("python", "python3", "mkdir", "rm", "mount", "sysctl", "touch", "tee", "sed"):
        guard = guards / command
        guard.write_text("#!/usr/bin/env bash\n"
                         f"printf '%s\\n' 'unexpected launch action: {command}' >&2\nexit 97\n")
        guard.chmod(0o755)
    env["PATH"] = str(guards) + os.pathsep + env.get("PATH", os.defpath)
    (project / "existing-config.conf").write_text("preserve the existing service configuration\n")
    for name, stub in zip(SDK_NAMES, sdk_files):
        stub.write_text(
            # Real set_env scripts may read unset optional variables under a strict caller.
            ': "${OSCAR_TEST_SDK_OPTIONAL}"\n'
            f"printf '%s\\t%s\\t%s\\t%s\\t%s\\n' {name} \"$PYTHONHASHSEED\" "
            '"$MMC_LOCAL_SERVICE_BACKEND_ID" "$ASCEND_RT_VISIBLE_DEVICES" '
            '"$CUDA_VISIBLE_DEVICES" >> "$TEST_SDK_TRACE"\n'
            f"export TEST_SDK_{name.upper()}=loaded\n"
            "return 0\n"
        )
    return SimpleNamespace(project=project, helper=helper, wrapper=wrapper,
                           capture=capture, sdks=sdk_files, env=env,
                           trace=trace, output=output)


def run_wrapper(tree, *args):
    process = subprocess.Popen(["bash", str(tree.wrapper), *args], cwd=tree.project.parent,
                               env=tree.env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(timeout=10)
    return SimpleNamespace(returncode=process.returncode, pid=process.pid,
                           stdout=stdout, stderr=stderr)


def sdk_calls(tree):
    return [] if not tree.trace.exists() else [line.split("\t") for line in tree.trace.read_text().splitlines()]


def assert_selected_environment(result):
    assert {key: result["env"][key] for key in EXPECTED_ENV} == EXPECTED_ENV
    assert "MMC_DIR_STORE_CONFIG_PATH" not in result["env"]
    for name in SDK_NAMES:
        assert result["env"][f"TEST_SDK_{name.upper()}"] == "loaded"


def test_helper_sources_sdk_order_in_current_shell_and_restores_strict_options(launch_tree):
    tree = launch_tree
    original_files = {path: path.read_bytes() for path in tree.project.rglob("*") if path.is_file()}
    result = subprocess.run(
        ["bash", "-c", 'set -euo pipefail; source "$1"; '
         '[[ "$-" == *e* && "$-" == *u* && -o pipefail ]] || exit 91; '
         'exec "$2" "$3"', "test", str(tree.helper), sys.executable, str(tree.capture)],
        env=tree.env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert sdk_calls(tree) == [[name, "0", "1", "0,1,2,3", "0,1,2,3"] for name in SDK_NAMES]
    assert_selected_environment(json.loads(tree.output.read_text()))
    assert {path: path.read_bytes() for path in tree.project.rglob("*") if path.is_file()} == original_files


def test_helper_reasserts_user_values_after_sdk_defaults(launch_tree):
    tree = launch_tree
    last_sdk = tree.sdks[-1]
    last_sdk.write_text(last_sdk.read_text().replace("return 0\n", "") +
                        "\n".join(f"export {key}=sdk-default" for key in EXPECTED_ENV) +
                        "\nset +eu\nset +o pipefail\nreturn 0\n")
    result = run_wrapper(tree)
    assert result.returncode == 0, result.stderr
    assert_selected_environment(json.loads(tree.output.read_text()))


@pytest.mark.parametrize("arguments", [[], ["--plan"], ["--rear-cards"], ["--no-enable-prefix-caching"],
    ["--config", "my configuration.json", "--log-dir", "logs with spaces", "--literal=$(untouched)", ""]])
def test_wrapper_execs_only_candidate_offload_and_preserves_user_arguments(launch_tree, arguments):
    tree = launch_tree
    result = run_wrapper(tree, *arguments)
    assert result.returncode == 0, result.stderr
    captured = json.loads(tree.output.read_text())
    assert captured["argv"] == ["--variant", "candidate", "--kv-offload", "--enable-prefix-caching", *arguments]
    assert captured["pid"] == result.pid  # Wrapper exec replaces its process; no detached service.
    assert captured["cwd"] == str(tree.project)
    assert_selected_environment(captured)
    assert [call[0] for call in sdk_calls(tree)] == list(SDK_NAMES)


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_wrapper_help_does_not_load_sdk_or_launch_install(launch_tree, flag):
    tree = launch_tree
    for sdk in tree.sdks:
        sdk.unlink()
    result = run_wrapper(tree, flag)
    assert result.returncode == 0, result.stderr
    assert "--plan" in result.stdout and "SDK" in result.stdout
    assert not tree.trace.exists() and not tree.output.exists()


@pytest.mark.parametrize("failed_index", range(4))
def test_sdk_failure_preserves_status_and_visible_error_without_starting(launch_tree, failed_index):
    tree = launch_tree
    status = 31 + failed_index
    sdk = tree.sdks[failed_index]
    sdk.write_text(sdk.read_text().replace("return 0\n",
        f"printf '%s\\n' 'SDK failure marker {failed_index}' >&2\nreturn {status}\n"))
    result = run_wrapper(tree)
    assert result.returncode == status
    assert f"SDK failure marker {failed_index}" in result.stderr
    assert [call[0] for call in sdk_calls(tree)] == list(SDK_NAMES[:failed_index + 1])
    assert not tree.output.exists()


def test_install_failure_exit_code_remains_visible_to_caller(launch_tree):
    launch_tree.env["TEST_INSTALL_RETURN_CODE"] = "43"
    result = run_wrapper(launch_tree)
    assert result.returncode == 43
    assert json.loads(launch_tree.output.read_text())["argv"] == [
        "--variant", "candidate", "--kv-offload", "--enable-prefix-caching"]


def test_env_helper_requires_source_instead_of_creating_an_unusable_subshell(launch_tree):
    result = subprocess.run(["bash", str(launch_tree.helper)], env=launch_tree.env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "source" in result.stderr
    assert not launch_tree.trace.exists() and not launch_tree.output.exists()
