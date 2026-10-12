"""Archive #132-135/#150: real host transport must not certify partial traces.

These tests execute ordinary producer processes and FIFO file descriptors.
They establish neither SDK FIFO compatibility nor kernel/phase performance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.experiments import stream_camodel_markers as stream


def marker(cycle, source, field, end=False):
    tag = (0x5a77c01d << 32) | (source << 8) | (field << 1) | end
    return f"[info] [{cycle}] MOV_XD_XN XD:X3={tag:#x}\n".encode()


def valid_trace():
    # More than a pipe buffer, ordinary instruction noise, fragmented writes,
    # and nested spans exercise the actual producer/reader transport.
    noise = b"[info] [0] ADD_XD_XN XD:X3=0x0\n" * 6000
    return (noise + marker(1, 3, 23) + marker(2, 0, 3) + marker(3, 0, 5)
            + marker(4, 0, 5, True) + marker(5, 0, 3, True)
            + marker(6, 3, 23, True))


PRODUCER = r"""
import json, os, pathlib, sys
names = json.loads(sys.argv[2]); payload = pathlib.Path(sys.argv[3]).read_bytes()
for name in names:
    with open(pathlib.Path(sys.argv[1]) / name, 'wb', buffering=0) as target:
        # Fragment markers across write boundaries; no test-only feed shortcut.
        for start in range(0, len(payload), 131):
            target.write(payload[start:start + 131])
"""


class FifoTransport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def produce(self, collector, payload):
        source = self.base / 'producer-input.bin'
        source.write_bytes(payload)
        result = subprocess.run(
            [sys.executable, '-c', PRODUCER, str(collector.log),
             json.dumps(list(collector.actors)), str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        return result.returncode

    @staticmethod
    def counts(collector):
        return {name: {(3, 23): 1, (0, 3): 1, (0, 5): 1}
                for name in collector.actors}

    def test_sixty_precreated_fifos_fragmented_real_producer(self):
        collector = stream.Collector(self.base / 'run')
        self.assertEqual(len(collector.actors), 60)
        self.assertTrue(all(actor.path.is_fifo() for actor in collector.actors.values()))
        payload = valid_trace()
        rc = self.produce(collector, payload)
        result = collector.finish(self.counts(collector), producer_rc=rc)
        self.assertEqual(result['status'], 'complete', result['failures'])
        self.assertEqual(result['raw_bytes'], len(payload) * 60)
        self.assertTrue(all(not actor.thread.is_alive() for actor in collector.actors.values()))
        for item in result['phase_marker_files']:
            self.assertEqual(item['raw_sha256'], hashlib.sha256(payload).hexdigest())
            evidence = json.loads((collector.output / item['path']).read_text())
            self.assertEqual(evidence['event_count'], 6)
            self.assertTrue(evidence['eof'])
            self.assertTrue(evidence['complete_for_actor'])
        with self.assertRaisesRegex(stream.MarkerError, 'already finished'):
            collector.finish(self.counts(collector))

    def test_incomplete_malformed_and_count_mismatch_fail(self):
        for name, payload, change in (
            ('missing_end', valid_trace().rsplit(b'\n', 2)[0] + b'\n', None),
            ('wrong_close', valid_trace().replace(marker(4, 0, 5, True), marker(4, 0, 6, True)), None),
            ('partial_line', valid_trace() + b'unterminated', None),
            ('count_mismatch', valid_trace(), 'count'),
            ('foreign_actor', valid_trace(), 'actor'),
        ):
            with self.subTest(name=name):
                collector = stream.Collector(self.base / name, cores=1)
                self.produce(collector, payload)
                expected = self.counts(collector)
                if change == 'count':
                    expected[next(iter(expected))][0, 5] = 2
                if change == 'actor':
                    expected['foreign'] = {}
                result = collector.finish(expected)
                self.assertEqual(result['status'], 'incomplete')
                self.assertTrue(result['failures'])

    def test_budget_exhaustion_keeps_draining_and_fails(self):
        collector = stream.Collector(self.base / 'budget', cores=1, max_marker_bytes=1)
        self.produce(collector, valid_trace())
        result = collector.finish(self.counts(collector))
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['raw_bytes'], len(valid_trace()) * 3)
        self.assertTrue(any('budget exhausted' in row for row in result['failures']))

    def test_missing_writer_and_failed_producer_never_pass(self):
        collector = stream.Collector(self.base / 'no-writer', cores=1)
        start = time.monotonic()
        result = collector.finish(self.counts(collector), producer_rc=7)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('producer returncode=7', result['failures'])

    def test_writer_remaining_open_and_continuous_data_are_bounded(self):
        collector = stream.Collector(self.base / 'leaked-writer', cores=1)
        actor = next(iter(collector.actors.values()))
        producer = subprocess.Popen([sys.executable, '-c', r"""
import os, sys
fd = os.open(sys.argv[1], os.O_WRONLY)
try:
    while True: os.write(fd, b'noise\n' * 10000)
except BrokenPipeError:
    pass
finally:
    os.close(fd)
""", str(actor.path)])
        try:
            deadline = time.monotonic() + 3
            while actor.raw_bytes == 0 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertGreater(actor.raw_bytes, 0)
            with patch.object(stream, 'DRAIN_SECONDS', .2):
                start = time.monotonic()
                result = collector.finish(self.counts(collector))
            self.assertLess(time.monotonic() - start, 2)
            self.assertEqual(result['status'], 'incomplete')
            self.assertTrue(all(not row.thread.is_alive() for row in collector.actors.values()))
            producer.wait(timeout=2)
        finally:
            if producer.poll() is None:
                producer.kill()
                producer.wait()

    def test_only_owned_fifo_inode_bypasses_quiet_dump_redirect(self):
        owned = self.base / 'owned'; foreign = self.base / 'foreign'
        for path in (owned, foreign):
            os.mkfifo(path)
        opened = []
        try:
            readers = [os.open(path, os.O_RDONLY | os.O_NONBLOCK) for path in (owned, foreign)]
            opened.extend(readers)
            writers = [os.open(path, os.O_WRONLY | os.O_NONBLOCK) for path in (owned, foreign)]
            opened.extend(writers)
            sink = os.open(os.devnull, os.O_WRONLY); opened.append(sink)
            regular = os.open(self.base / 'ordinary.dump', os.O_WRONLY | os.O_CREAT, 0o600)
            opened.append(regular)
            identity = owned.stat()
            proxy = stream.OwnedFifoOS({'actors': {'owned': [identity.st_dev, identity.st_ino]}})
            for fd in (*writers, regular):
                proxy.dup2(sink, fd)
                os.write(fd, b'witness')
            self.assertEqual(os.read(readers[0], 99), b'witness')
            self.assertEqual(os.read(readers[1], 99), b'')
            self.assertEqual((self.base / 'ordinary.dump').read_bytes(), b'')
        finally:
            for fd in reversed(opened):
                os.close(fd)

    def test_worker_adapts_precreated_paths_without_reporting_final_success(self):
        collector = stream.Collector(self.base / 'worker', cores=1)
        payload = self.base / 'worker-input.bin'; payload.write_bytes(valid_trace())
        replayer = self.base / 'stub_replayer.py'
        # Real import and subprocess exec exercise only the adapter boundary.
        # The stub is explicitly not a numerical or SDK substitute.
        replayer.write_text('''
import argparse, json, os
from pathlib import Path
def save_json(path, value): path.write_text(json.dumps(value))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--payload')
    a=p.parse_args(); output=Path(a.output); output.mkdir(parents=True,exist_ok=False)
    log=output/'log_ca'; log.mkdir()
    sink=os.open(os.devnull,os.O_WRONLY)
    try:
        for path in sorted(log.glob('*.dump')):
            fd=os.open(path,os.O_WRONLY)
            try:
                os.dup2(sink,fd)
                os.write(fd,Path(a.payload).read_bytes())
            finally: os.close(fd)
    finally: os.close(sink)
    save_json(output/'report.json',{'status':'passed','task_ticks':7,'test_stub':True})
    return 0
''')
        result = subprocess.run([
            sys.executable, str(Path(stream.__file__)), '--worker',
            '--fifo-manifest', str(collector.descriptor), '--replayer', str(replayer),
            '--', '--output', str(collector.output), '--payload', str(payload),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        report = json.loads((collector.output/'report.json').read_text())
        self.assertEqual(report['status'], 'numeric_passed_pending_stream_markers')
        self.assertEqual(report['replayer_numerical_status'], 'passed')
        self.assertTrue(report['test_stub'])
        collected = collector.finish(self.counts(collector), producer_rc=result.returncode)
        self.assertEqual(collected['status'], 'complete', collected['failures'])

    def test_supervisor_deadline_terminates_owned_child_and_returns_failure(self):
        replayer = self.base / 'hung_replayer.py'
        replayer.write_text('''
import os, signal, time
from pathlib import Path
def save_json(path, value): pass
def main():
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    while True: time.sleep(.02)
''')
        # The test worker calls no SDK. An empty preload is intentionally not
        # a substitute library; Linux ignores it with a loader warning.
        preload = self.base / 'not-an-sdk.so'; preload.write_bytes(b'')
        output = self.base / 'deadline'
        started = time.monotonic()
        result = subprocess.run([
            sys.executable, str(Path(stream.__file__)), '--replayer', str(replayer),
            '--runtime-preload', str(preload), '--timeout-seconds', '1',
            '--', '--output', str(output), '--profile-counters',
        ], env={k: v for k, v in os.environ.items() if k != 'LD_PRELOAD'},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertLess(time.monotonic() - started, 9)
        self.assertEqual(result.returncode, 2, result.stderr.decode())
        summary = json.loads((output/'stream_marker_summary.json').read_text())
        self.assertEqual(summary['status'], 'incomplete')
        self.assertIn('unchanged host deadline reached', summary['failures'])
        self.assertTrue(all(not row['complete_for_actor'] for row in summary['phase_marker_files']))
        owner = json.loads((output/'stream_owner.json').read_text())
        self.assertEqual(owner['runtime_pid'], owner['runtime_pgid'])
        self.assertEqual(owner['runtime_returncode'], -signal.SIGKILL)
        with self.assertRaises(ProcessLookupError):
            os.kill(owner['runtime_pid'], 0)

    def test_external_term_is_forwarded_before_supervisor_exits(self):
        replayer = self.base / 'term_replayer.py'
        ready = self.base / 'runtime-ready'
        replayer.write_text(f'''
import os, signal, time
from pathlib import Path
def save_json(path, value): pass
def main():
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    Path({str(ready)!r}).write_text(str(os.getpid()))
    while True: time.sleep(.02)
''')
        preload = self.base / 'not-an-sdk.so'; preload.write_bytes(b'')
        output = self.base / 'external-term'
        supervisor = subprocess.Popen([
            sys.executable, str(Path(stream.__file__)), '--replayer', str(replayer),
            '--runtime-preload', str(preload), '--timeout-seconds', '30',
            '--', '--output', str(output), '--profile-counters',
        ], env={k: v for k, v in os.environ.items() if k != 'LD_PRELOAD'},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            wait_until = time.monotonic() + 4
            while not ready.exists() and time.monotonic() < wait_until:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            runtime_pid = int(ready.read_text())
            started = time.monotonic(); supervisor.send_signal(signal.SIGTERM)
            _stdout, stderr = supervisor.communicate(timeout=12)
            self.assertLess(time.monotonic() - started, 10)
            self.assertEqual(supervisor.returncode, 2, stderr.decode())
            with self.assertRaises(ProcessLookupError): os.kill(runtime_pid, 0)
            owner = json.loads((output/'stream_owner.json').read_text())
            self.assertEqual(owner['runtime_returncode'], -signal.SIGKILL)
            summary = json.loads((output/'stream_marker_summary.json').read_text())
            self.assertIn('interrupted', summary['failures'])
        finally:
            if supervisor.poll() is None:
                supervisor.kill(); supervisor.communicate()


if __name__ == '__main__':
    unittest.main()
