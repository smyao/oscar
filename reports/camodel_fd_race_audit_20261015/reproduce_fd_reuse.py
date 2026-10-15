"""Archive #94/#95/#125: local FD reuse counterexample, never SDK/NPU evidence.

Darwin F_GETPATH supplies the same transient pathname observation as Linux
readlink(/proc/self/fd/N). Explicit legal close/open interleaving replaces an
unrelated profile descriptor before the delayed dup2. No project file is a
victim; all writes are confined to a TemporaryDirectory.
"""
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile


def observe_path(fd):
    if sys.platform == 'darwin':
        return fcntl.fcntl(fd, fcntl.F_GETPATH, b'\0' * 1024).split(b'\0', 1)[0].decode()
    return os.readlink(f'/proc/self/fd/{fd}')


def reproduce(redirect):
    with tempfile.TemporaryDirectory(prefix='oscar-fd-reuse-', dir='/tmp') as tmp:
        log = Path(tmp).resolve()
        sink = os.open('/dev/null', os.O_WRONLY)
        dump = os.open(log/'core0.cubecore0.instr_log.dump', os.O_WRONLY | os.O_CREAT, 0o600)
        observed = observe_path(dump)
        assert observed.startswith(str(log) + '/') and observed.endswith('.dump')
        os.close(dump)  # SDK thread closes the dump after quiet inspected it.
        profile = os.open(log/'profile_task_log0.toml', os.O_WRONLY | os.O_CREAT, 0o600)
        assert profile == dump  # Kernel reuses the now-lowest free FD number.
        before = observe_path(profile)
        if redirect:
            os.dup2(sink, dump)  # Delayed quiet operation now replaces profile.
        payload = b'[task]\nduration = 123\n'
        written = os.write(profile, payload)
        os.close(profile)
        os.close(sink)
        actual = (log/'profile_task_log0.toml').read_bytes()
        return {'redirect':redirect, 'same_numeric_fd_reused':True,
                'observed_suffix':Path(observed).suffix,
                'destination_before_dup2':Path(before).name,
                'write_returned_bytes':written,'actual_profile_bytes':len(actual),
                'profile_contents_preserved':actual == payload}


if __name__ == '__main__':
    control = reproduce(False)
    raced = reproduce(True)
    assert control['profile_contents_preserved']
    assert raced['write_returned_bytes'] > 0 and raced['actual_profile_bytes'] == 0
    print(json.dumps({'status':'deterministic_local_FD_reuse_counterexample',
                      'platform':sys.platform,'control':control,'raced':raced,
                      'C16_timeout_causality':'not_proven','SDK_execution':False,
                      'VM_accessed':False,'NPU_accessed':False},indent=2))
