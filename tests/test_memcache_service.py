"""Archive #94/#125/#175/#176/#177/#178: startup and nonconnecting observation.

Only temporary configuration/lock files and listener tables are real. Host changes,
SDK calls, and MetaService are isolated or replaced; no NPU/service is started.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from tools import memcache_service as service

META = "tcp://127.0.0.1:5000"
STORE = "tcp://127.0.0.1:6000"


@pytest.fixture
def config_files(tmp_path):
    local = tmp_path / "mmc-local.conf"
    meta = tmp_path / "mmc-meta.conf"
    local.write_text(
        f"ock.mmc.meta_service_url = {META}\n"
        f"ock.mmc.local_service.config_store_url = {STORE}\n"
        "ock.mmc.local_service.dram.size = 32GB\n"
        "ock.mmc.local_service.protocol = device_rdma\n"
        "ock.mmc.local_service.storage.enabled = true\n")
    meta.write_text(
        f"ock.mmc.meta_service_url = {META}\n"
        f"ock.mmc.meta_service.config_store_url = {STORE}\n"
        "ock.mmc.log_level = info\n"
        "ock.mmc.evict_threshold_high = 90\n"
        "ock.mmc.evict_threshold_low = 80\n")
    return local, meta


@pytest.fixture
def lifecycle(monkeypatch, config_files):
    local, meta = config_files
    state = NS(events=[], listeners={META: False, STORE: False}, acquired=True,
               sdk_held=False, host_rc=0, query_error=None, serve_error=None)
    original_prepare = service.prepare_node93
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", "before-local-test")
    monkeypatch.setenv("MMC_META_CONFIG_PATH", "before-meta-test")

    def prepare(directory, *, plan=False):
        state.events.append("plan" if plan else "prepare")
        return original_prepare(directory, plan=plan)

    @contextmanager
    def lock(meta_url, store_url):
        assert (meta_url, store_url) == (META, STORE)
        state.events.append("lock-enter")
        try:
            yield state.acquired
        finally:
            state.events.append("lock-exit")

    def listening(url):
        state.events.append(("listen", url))
        return state.listeners[url]

    def sdk_held():
        state.events.append("sdk-lock")
        return state.sdk_held

    def host(argv):
        assert argv == []
        state.events.append("host")
        return state.host_rc

    def query(path):
        assert path == local
        state.events.append("query")
        if state.query_error is not None:
            raise state.query_error

    def serve(path):
        assert path == meta
        state.events.append("serve")
        if state.serve_error is not None:
            raise state.serve_error
        return 0

    monkeypatch.setattr(service, "prepare_node93", prepare)
    monkeypatch.setattr(service, "launcher_lock", lock)
    monkeypatch.setattr(service, "endpoint_is_listening", listening)
    monkeypatch.setattr(service, "sdk_lock_held", sdk_held)
    monkeypatch.setattr(service.node93_host, "main", host)
    monkeypatch.setattr(service, "verify_metadata", query)
    monkeypatch.setattr(service, "serve_meta", serve)
    state.before = (local.read_bytes(), meta.read_bytes())
    state.files = (local, meta)
    return state


def unchanged(state):
    assert tuple(path.read_bytes() for path in state.files) == state.before
    assert not {"host", "prepare", "serve"}.intersection(
        event for event in state.events if isinstance(event, str))


@pytest.mark.parametrize("acquired", [False, True])
@pytest.mark.parametrize("prepare_host", [False, True])
def test_existing_service_query_reuses_without_host_config_or_start(lifecycle, acquired, prepare_host, capsys):
    lifecycle.listeners = {META: True, STORE: True}
    lifecycle.acquired = acquired
    lifecycle.sdk_held = True
    assert service.start_node93(lifecycle.files[0].parent, prepare_host=prepare_host) == 0
    unchanged(lifecycle)
    assert lifecycle.events.count("query") == 1
    assert lifecycle.events[0] == "plan" and lifecycle.events[-1] == "lock-exit"
    output = capsys.readouterr().out
    assert "owned by its original launcher" in output
    assert "keep that process running while OSCAR uses offload" in output


@pytest.mark.parametrize("prepare_host", [False, True])
def test_absent_service_changes_only_after_guard_and_starts_once(lifecycle, prepare_host):
    assert service.start_node93(lifecycle.files[0].parent, prepare_host=prepare_host) == 0
    names = [event for event in lifecycle.events if isinstance(event, str)]
    assert names == ["plan", "lock-enter", "sdk-lock", *(["host"] if prepare_host else []),
                     "prepare", "serve", "lock-exit"]
    assert lifecycle.events.index(("listen", STORE)) < lifecycle.events.index("prepare")
    assert "query" not in lifecycle.events
    assert b"40GB" in lifecycle.files[0].read_bytes()


@pytest.mark.parametrize("listeners", [{META: True, STORE: False}, {META: False, STORE: True}])
def test_partial_listener_never_repairs_or_starts_another_service(lifecycle, listeners):
    lifecycle.listeners = listeners
    with pytest.raises(RuntimeError):
        service.start_node93(lifecycle.files[0].parent, prepare_host=True)
    unchanged(lifecycle)
    assert "query" not in lifecycle.events


@pytest.mark.parametrize("busy", ["launcher", "sdk"])
def test_busy_lock_without_listeners_prevents_every_mutation(lifecycle, busy):
    lifecycle.acquired = busy != "launcher"
    lifecycle.sdk_held = busy == "sdk"
    with pytest.raises(RuntimeError):
        service.start_node93(lifecycle.files[0].parent, prepare_host=True)
    unchanged(lifecycle)
    assert "query" not in lifecycle.events


@pytest.mark.parametrize("error", [RuntimeError("query failed"),
    subprocess.CalledProcessError(9, ["metadata-client"]),
    subprocess.TimeoutExpired(["metadata-client"], 15)])
def test_open_ports_without_successful_metadata_query_are_not_reusable(lifecycle, error):
    lifecycle.listeners = {META: True, STORE: True}
    lifecycle.query_error = error
    with pytest.raises(type(error)) as caught:
        service.start_node93(lifecycle.files[0].parent, prepare_host=True)
    assert caught.value is error
    unchanged(lifecycle)


def test_host_failure_preserves_exit_code_and_does_not_write_config_or_start(lifecycle):
    lifecycle.host_rc = 23
    assert service.start_node93(lifecycle.files[0].parent, prepare_host=True) == 23
    assert "prepare" not in lifecycle.events and "serve" not in lifecycle.events
    assert tuple(path.read_bytes() for path in lifecycle.files) == lifecycle.before
    assert lifecycle.events[-1] == "lock-exit"


def test_config_error_before_guard_leaves_host_and_service_untouched(lifecycle):
    lifecycle.files[0].write_text("malformed configuration\n")
    before = tuple(path.read_bytes() for path in lifecycle.files)
    with pytest.raises(ValueError):
        service.start_node93(lifecycle.files[0].parent, prepare_host=True)
    assert lifecycle.events == ["plan"]
    assert tuple(path.read_bytes() for path in lifecycle.files) == before


def test_sdk_main_failure_after_cold_start_is_never_converted_to_reuse(lifecycle):
    lifecycle.serve_error = RuntimeError("MemCache MetaService failed: -1")
    with pytest.raises(RuntimeError) as caught:
        service.start_node93(lifecycle.files[0].parent)
    assert caught.value is lifecycle.serve_error
    assert lifecycle.events.count("serve") == 1 and "query" not in lifecycle.events
    assert lifecycle.events[-1] == "lock-exit"


TCP_HEADER = "  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"


def tcp_row(address_hex, port=5000, state="0A"):
    return (f"0: {address_hex}:{port:04X} {'0' * len(address_hex)}:0000 {state} "
            "00000000:00000000 00:00000000 00000000 0 0 1234 1 0000000000000000\n")


def ipv4_hex(address):
    return f"{int.from_bytes(bytes(map(int, address.split('.'))), sys.byteorder):08X}"


@pytest.fixture
def tcp_tables(monkeypatch, tmp_path):
    paths = tmp_path / "tcp", tmp_path / "tcp6"
    for path in paths:
        path.write_text(TCP_HEADER)
    monkeypatch.setattr(service, "TCP4_TABLE", paths[0])
    monkeypatch.setattr(service, "TCP6_TABLE", paths[1])
    def forbidden(*args, **kwargs):
        pytest.fail("listener observation must not connect, bind, or launch ss")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(service.subprocess, "run", forbidden)
    return paths


@pytest.mark.parametrize("host,port,state,expected", [
    ("127.0.0.1", 5000, "0A", True), ("0.0.0.0", 5000, "0A", True),
    ("127.0.0.2", 5000, "0A", False), ("192.0.2.1", 5000, "0A", False),
    ("127.0.0.1", 6000, "0A", False), ("127.0.0.1", 5000, "01", False),
    ("127.0.0.1", 5000, "06", False),
])
def test_readonly_ipv4_check_requires_matching_listen_address_and_port(tcp_tables, host, port, state, expected):
    tcp_tables[0].write_text(TCP_HEADER + tcp_row(ipv4_hex(host), port, state))
    assert service.endpoint_is_listening(META) is expected


def test_empty_valid_listener_tables_mean_absent_without_network_calls(tcp_tables):
    assert service.endpoint_is_listening(META) is False


@pytest.mark.parametrize("address", ["0" * 32, "0" * 24 + "01000000", "0" * 16 + "FFFF00000100007F"])
def test_same_port_ipv6_is_indeterminate_instead_of_permitting_cold_start(tcp_tables, address):
    tcp_tables[1].write_text(TCP_HEADER.replace("rem_address", "remote_address") + tcp_row(address))
    with pytest.raises(RuntimeError):
        service.endpoint_is_listening(META)


def test_ipv6_other_port_does_not_hide_absent_ipv4_endpoint(tcp_tables):
    tcp_tables[1].write_text(TCP_HEADER + tcp_row("0" * 32, port=6000))
    assert service.endpoint_is_listening(META) is False


def test_explicit_ipv4_listener_remains_proof_when_ipv6_shares_port(tcp_tables):
    tcp_tables[0].write_text(TCP_HEADER + tcp_row(ipv4_hex("127.0.0.1")))
    tcp_tables[1].write_text(TCP_HEADER + tcp_row("0" * 32))
    assert service.endpoint_is_listening(META) is True


def test_missing_ipv4_table_is_indeterminate(tcp_tables):
    tcp_tables[0].unlink()
    with pytest.raises(RuntimeError) as caught:
        service.endpoint_is_listening(META)
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_missing_ipv6_table_is_normal_when_ipv6_is_disabled(tcp_tables):
    tcp_tables[1].unlink()
    assert service.endpoint_is_listening(META) is False


@pytest.mark.parametrize("bad", ["", "unrecognized header\n", TCP_HEADER + "bad row\n",
    TCP_HEADER + tcp_row("0100007F").replace(":1388", ":ZZZZ", 1),
    TCP_HEADER + tcp_row("0100007F", state="GG"),
    TCP_HEADER + "0: 0100007F:1388 00000000:0000\n",
])
def test_malformed_tcp_table_never_becomes_absent(tcp_tables, bad):
    tcp_tables[0].write_text(bad)
    with pytest.raises(RuntimeError):
        service.endpoint_is_listening(META)


@pytest.mark.parametrize("missing", [False, True])
def test_matching_ipv4_does_not_depend_on_ipv6_table(tcp_tables, missing):
    tcp_tables[0].write_text(TCP_HEADER + tcp_row(ipv4_hex("127.0.0.1")))
    if missing:
        tcp_tables[1].unlink()
    else:
        tcp_tables[1].write_text(TCP_HEADER + "bad IPv6 row\n")
    assert service.endpoint_is_listening(META) is True


def test_malformed_ipv6_table_without_ipv4_match_is_indeterminate(tcp_tables):
    tcp_tables[1].write_text(TCP_HEADER + "bad IPv6 row\n")
    with pytest.raises(RuntimeError):
        service.endpoint_is_listening(META)


@pytest.mark.parametrize("tail", ["", " kernel-specific queue/timer columns"])
def test_listener_matching_does_not_validate_unrelated_columns(tcp_tables, tail):
    row = f"0: {ipv4_hex('127.0.0.1')}:1388 00000000:0000 0A{tail}\n"
    tcp_tables[0].write_text(TCP_HEADER + row)
    assert service.endpoint_is_listening(META) is True


def test_listener_table_permission_failure_preserves_cause(tcp_tables, monkeypatch):
    error = PermissionError("listener table inaccessible")
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == tcp_tables[1]:
            raise error
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(RuntimeError) as caught:
        service.endpoint_is_listening(META)
    assert caught.value.__cause__ is error


def test_listener_table_invalid_encoding_fails_closed(tcp_tables):
    tcp_tables[0].write_bytes(b"\xff")
    with pytest.raises(RuntimeError) as caught:
        service.endpoint_is_listening(META)
    assert isinstance(caught.value.__cause__, UnicodeError)


@pytest.mark.parametrize("url", ["tcp://localhost:5000", "tcp://192.0.2.1:5000", "tcp://[::1]:5000"])
def test_non_node93_address_is_explicitly_unsupported(tcp_tables, url):
    with pytest.raises(RuntimeError):
        service.endpoint_is_listening(url)


@pytest.fixture
def fake_sdk(monkeypatch, config_files):
    state = NS(calls=[], init_result=0, result=[0], query_error=None, close_error=None)
    class Store:
        def __init__(self):
            state.calls.append(("construct", os.environ.get("MMC_LOCAL_CONFIG_PATH")))
        def init(self, rank, *, init_bm):
            state.calls.append(("init", rank, init_bm))
            return state.init_result
        def batch_is_exist(self, keys, flags):
            state.calls.append(("query", keys, flags))
            if state.query_error is not None:
                raise state.query_error
            return state.result
        def close(self):
            state.calls.append(("close",))
            if state.close_error is not None:
                raise state.close_error
        def __getattr__(self, name):
            pytest.fail(f"metadata probe must not call other SDK operations: {name}")
    monkeypatch.setitem(sys.modules, "memcache_hybrid", NS(DistributedObjectStore=Store))
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", "before-test")
    return state


@pytest.mark.parametrize("status", [0, 1])
def test_metadata_query_is_scheduler_only_read_and_closes(fake_sdk, config_files, status):
    local, _ = config_files
    fake_sdk.result = [status]
    service.metadata_query(local)
    assert fake_sdk.calls[0] == ("construct", str(local))
    assert fake_sdk.calls[1] == ("init", 0, False)
    query = fake_sdk.calls[2]
    assert query[0] == "query" and query[2] == 0
    assert type(query[1]) is list and len(query[1]) == 1 and isinstance(query[1][0], str) and query[1][0]
    assert fake_sdk.calls[3:] == [("close",)]
    service.metadata_query(local)
    assert fake_sdk.calls[6][1] != query[1]


@pytest.mark.parametrize("result", [None, [], [0, 1], (0,), [False], [True], [-1], [2], ["0"], {"value": 0}])
def test_metadata_query_rejects_non_status_responses_and_closes(fake_sdk, config_files, result):
    fake_sdk.result = result
    with pytest.raises((ValueError, RuntimeError)):
        service.metadata_query(config_files[0])
    assert fake_sdk.calls[-1] == ("close",)


def test_metadata_query_failure_and_cleanup_failure_are_not_success(fake_sdk, config_files):
    error = RuntimeError("SDK existence failed")
    fake_sdk.query_error = error
    with pytest.raises(RuntimeError) as caught:
        service.metadata_query(config_files[0])
    assert caught.value is error and fake_sdk.calls[-1] == ("close",)
    fake_sdk.query_error = None
    fake_sdk.close_error = RuntimeError("SDK close failed")
    with pytest.raises(RuntimeError) as caught:
        service.metadata_query(config_files[0])
    assert caught.value is fake_sdk.close_error


@pytest.mark.parametrize("result", [-1, 1, None, False])
def test_metadata_init_failure_prevents_query(fake_sdk, config_files, result):
    fake_sdk.init_result = result
    with pytest.raises(RuntimeError):
        service.metadata_query(config_files[0])
    assert not any(call[0] == "query" for call in fake_sdk.calls)


def test_metadata_verification_uses_bounded_child_with_visible_output(monkeypatch, config_files):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return NS(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", run)
    service.verify_metadata(config_files[0])
    command, kwargs = calls[0]
    assert command == [sys.executable, "-m", "tools.memcache_service", "check-client",
                       "--local-config", str(config_files[0])]
    assert kwargs["check"] is True and kwargs["timeout"] == 15
    assert not kwargs.get("capture_output", False)
    assert kwargs.get("stdout") is None and kwargs.get("stderr") is None


@pytest.mark.parametrize("error", [subprocess.CalledProcessError(19, ["client"]),
                                  subprocess.TimeoutExpired(["client"], 15)])
def test_metadata_child_failure_is_preserved(monkeypatch, config_files, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(service.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as caught:
        service.verify_metadata(config_files[0])
    assert caught.value.__cause__ is error


def test_launcher_lock_is_nonblocking_endpoint_scoped_and_released(tmp_path):
    with service.launcher_lock(META, STORE, lock_root=tmp_path) as first:
        assert first is True
        with service.launcher_lock(META, STORE, lock_root=tmp_path) as duplicate:
            assert duplicate is False
        with service.launcher_lock(META, "tcp://127.0.0.1:6001", lock_root=tmp_path) as other:
            assert other is True
    with service.launcher_lock(META, STORE, lock_root=tmp_path) as after_release:
        assert after_release is True


def test_launcher_lock_excludes_an_independent_process_until_release(tmp_path):
    code = (
        "from pathlib import Path\n"
        "import sys\n"
        "from tools.memcache_service import launcher_lock\n"
        "with launcher_lock(sys.argv[1], sys.argv[2], lock_root=Path(sys.argv[3])) as acquired:\n"
        "    print(int(acquired))\n"
    )
    def child():
        return subprocess.run([sys.executable, "-c", code, META, STORE, str(tmp_path)],
            cwd=Path(__file__).resolve().parents[1], check=True, timeout=5,
            capture_output=True, text=True).stdout.strip()
    with service.launcher_lock(META, STORE, lock_root=tmp_path) as acquired:
        assert acquired is True and child() == "0"
    assert child() == "1"


@pytest.mark.parametrize("result", [None, 0, -1, 17])
def test_serve_meta_preserves_sdk_failure_semantics(monkeypatch, config_files, result):
    calls = []
    def main():
        calls.append(os.environ.get("MMC_META_CONFIG_PATH"))
        return result
    monkeypatch.setitem(sys.modules, "memcache_hybrid", NS(MetaService=NS(main=main)))
    monkeypatch.setenv("MMC_META_CONFIG_PATH", "before-test")
    if result in (None, 0):
        assert service.serve_meta(config_files[1]) == 0
    else:
        with pytest.raises(RuntimeError):
            service.serve_meta(config_files[1])
    assert calls == [str(config_files[1])]


def test_node93_cli_preserves_options_and_nonzero_host_result(monkeypatch, tmp_path):
    seen = []
    def start(directory, *, prepare_host):
        seen.append((directory, prepare_host))
        return 23
    monkeypatch.setattr(service, "start_node93", start)
    assert service.main(["node93", "--config-dir", str(tmp_path), "--prepare-host"]) == 23
    assert seen == [(tmp_path, True)]


def test_check_client_cli_runs_only_metadata_query(monkeypatch, config_files):
    seen = []
    monkeypatch.setattr(service, "metadata_query", lambda path: seen.append(path))
    monkeypatch.setattr(service, "start_node93", lambda *a, **k:
                        pytest.fail("metadata child must not start a service"))
    assert service.main(["check-client", "--local-config", str(config_files[0])]) == 0
    assert seen == [config_files[0]]
