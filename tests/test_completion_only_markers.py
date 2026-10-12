"""Archive #132-135/#150: bounded research logs never certify a truncated actor.

Host regular-file transport/policy only; no SDK, kernel or NPU validation.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from csrc.experiments.window_batch_decode import replay_camodel as replay
from tools.experiments.stream_camodel_markers import Intervals, actor_names


def event(cycle, source, field, end=False):
    tag=(0x5a77c01d<<32)|(source<<8)|(field<<1)|end
    return f'[info] [{cycle}] MOV_XD_XN XD:X3={tag:#x}\n'.encode()


def trace():
    return event(1,3,23)+event(2,0,3)+event(3,0,3,True)+event(4,3,23,True)


def test_default_caps_unchanged_and_research_needs_profile_twenty_cores_and_free_tmpfs(tmp_path):
    assert replay.trace_policy(0,20,False,False,tmp_path)['per_file_budget_bytes']==0
    assert replay.trace_policy(2*1024**2,20,False,False,tmp_path)['research_completion_only'] is False
    with pytest.raises(ValueError,match='2MiB cap'):
        replay.trace_policy(64*1024**2,20,True,False,tmp_path)
    for budget,cores,profile in ((0,20,True),(64*1024**2,2,True),(64*1024**2,20,False)):
        with pytest.raises(ValueError,match='profile counters, 20 cores'):
            replay.trace_policy(budget,cores,profile,True,tmp_path)
    with patch.object(Path,'stat',return_value=SimpleNamespace(st_dev=17)), patch.object(
            replay.os,'statvfs',return_value=SimpleNamespace(f_bavail=9*1024**3//2,f_frsize=1)):
        result=replay.trace_policy(64*1024**2,20,True,True,tmp_path)
        assert result['popped_logs']=='redirected_to_dev_null'
    with patch.object(Path,'stat',return_value=SimpleNamespace(st_dev=17)), patch.object(
            replay.os,'statvfs',return_value=SimpleNamespace(f_bavail=9*1024**3//2-1,f_frsize=1)):
        with pytest.raises(ValueError,match='4.5GiB'):
            replay.trace_policy(64*1024**2,20,True,True,tmp_path)
    with patch.object(Path,'stat',side_effect=[SimpleNamespace(st_dev=1),SimpleNamespace(st_dev=2)]):
        with pytest.raises(ValueError,match='/dev/shm'):
            replay.trace_policy(64*1024**2,20,True,True,tmp_path)


def test_research_only_completion_logs_are_retained():
    for name in actor_names():
        assert replay.retain_instruction(name,20,True)
        assert not replay.retain_instruction(name.replace('.instr_log.', '.instr_popped_log.'),20,True)
        assert replay.retain_instruction(name.replace('.instr_log.', '.instr_popped_log.'),20,False)
    assert not replay.retain_instruction('core20.veccore0.instr_log.dump',20,True)
    assert not replay.retain_instruction('core0.veccore2.instr_log.dump',20,True)


def test_sixty_regular_files_save_closed_markers_gzip_then_remove_owned_raw(tmp_path):
    log=tmp_path/'log_ca';log.mkdir();raw=trace()
    expected={name:{(3,23):1,(0,3):1} for name in actor_names()}
    for name in expected:(log/name).write_bytes(raw)
    unrelated=log/'not-an-instruction.dump';unrelated.write_bytes(b'keep')
    samples,files,failures=replay.collect_completion_markers(
        tmp_path,log,tmp_path/'samples',4096,set(),expected,Intervals)
    assert not failures and len(files)==60
    assert all(row['complete_for_actor'] and row['raw_removed'] for row in samples)
    for row in files:
        evidence=json.loads((tmp_path/row['path']).read_text())
        assert evidence['source_sha256']==hashlib.sha256(raw).hexdigest()
        assert len(evidence['events'])==4
        assert gzip.decompress((tmp_path/'samples'/(row['source']+'.gz')).read_bytes())==raw
        assert not (log/row['source']).exists()
    assert unrelated.read_bytes()==b'keep'


@pytest.mark.parametrize('failure',['cap','observed_cap','unclosed','wrong_count'])
def test_cap_or_incomplete_actor_never_passes_and_keeps_bounded_evidence(tmp_path,failure):
    log=tmp_path/'log_ca';log.mkdir();name=actor_names(1)[0];raw=trace()
    budget=4096;limited=set();counts={(3,23):1,(0,3):1}
    if failure=='cap':budget=len(raw)-1
    if failure=='observed_cap':limited.add(name)
    if failure=='unclosed':raw=raw[:raw.rfind(b'[info]')]
    if failure=='wrong_count':counts[0,3]=2
    (log/name).write_bytes(raw)
    samples,files,failures=replay.collect_completion_markers(
        tmp_path,log,tmp_path/'samples',budget,limited,{name:counts},Intervals)
    assert failures and files[0]['complete_for_actor'] is False
    assert samples[0]['captured_bytes']<=budget and samples[0]['raw_removed']
    assert gzip.decompress((tmp_path/'samples'/(name+'.gz')).read_bytes())==raw[:budget]


@pytest.mark.parametrize('kind',['symlink','fifo'])
def test_nonregular_or_unowned_file_is_never_deleted_or_blockingly_read(tmp_path,kind):
    log=tmp_path/'log_ca';log.mkdir();name=actor_names(1)[0];path=log/name
    original=tmp_path/'original';original.write_bytes(trace())
    if kind=='symlink':path.symlink_to(original)
    else:os.mkfifo(path)
    samples,files,failures=replay.collect_completion_markers(
        tmp_path,log,tmp_path/'samples',4096,set(),{name:{(3,23):1}},Intervals)
    assert failures and not files[0]['complete_for_actor']
    assert not samples[0]['raw_removed'] and path.exists()
    assert original.read_bytes()==trace()
