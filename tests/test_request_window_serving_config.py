"""Archive #34/#36/#127-129/#142/#153/#155: one-key research routing contracts.

Explicit research config is exercised without enabling the default preset or
claiming any device/graph evidence. Native A/B must share the omitted budget and unchanged native scheduler.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import install_serve, observe_serve
from tools.serving_variants import variant_config
from tools.serving_variants import FEATURE_FLAGS
from tools.phase import PhaseResult
from tools.target_cli import serve_argv


def configured():
    value=json.loads((Path(__file__).resolve().parents[1]/'configs/target.json').read_text())
    value.update({flag:True for flag in FEATURE_FLAGS.values()})
    value.update(experimental_request_windows=True,experimental_bounded_whole_prefill=True,
                 experimental_mdl_decode=False)
    return value


def test_unqualified_research_mdl_combination_is_rejected_by_both_entries(tmp_path,capsys):
    raw=configured();before=deepcopy(raw)
    path=tmp_path/'research.json';path.write_text(json.dumps(raw))
    # The fixed candidate now promises the qualified MDL q4 implementation.
    # A research-only request-window reader has no matching MDL qualification.
    for entry in (install_serve, observe_serve):
        with pytest.raises(ValueError,match='request-window MDL reader is not qualified'):
            entry.main(['--config',str(path),'--variant','candidate','--plan'])
    argv=serve_argv(variant_config(raw,None))
    assert '--max-num-batched-tokens' not in argv
    assert '--scheduler-cls' not in argv
    assert raw==before


def test_research_roundtrip_clears_every_owned_change_for_native_and_baseline():
    raw=configured()
    # The current candidate promises MDL; an unfinished request-window MDL
    # reader must be explicitly rejected, never silently mapped to its old op.
    with pytest.raises(ValueError,match='request-window MDL reader is not qualified'):
        variant_config(raw,'candidate')
    candidate=variant_config(raw,None)
    assert variant_config(candidate,None)==candidate
    for variant in ('native','baseline'):
        restored=variant_config(candidate,variant)
        assert 'max_num_batched_tokens' not in restored
        assert 'history_token_capacity' not in restored and 'scheduler_cls' not in restored
        assert not restored['experimental_request_windows']
        assert not restored['experimental_bounded_whole_prefill']


def test_explicit_config_without_variant_persists_normalized_capacity_for_every_phase(monkeypatch,tmp_path):
    raw=configured()
    path=tmp_path/'explicit-research.json';path.write_text(json.dumps(raw));before=path.read_bytes()
    logs=tmp_path/'launch';seen=[]
    def check(selected_path):
        assert Path(selected_path)==logs/'effective-target.json'
        selected=json.loads(Path(selected_path).read_text())
        assert 'max_num_batched_tokens' not in selected
        assert 'history_token_capacity' not in selected
        assert 'scheduler_cls' not in selected
        assert selected['use_native_token_budget'] is True
    def phase(name,command,*,env,**kwargs):
        seen.append(name);check(env['OSCAR_TARGET_CONFIG'])
        if '--config' in command:assert command[command.index('--config')+1]==env['OSCAR_TARGET_CONFIG']
        return PhaseResult(name,command,0,0.,False,False,str(logs/name),True)
    class HostExecIntercept(Exception):pass
    def launch(executable,command,env):
        check(command[-1]);assert command[-1]==env['OSCAR_TARGET_CONFIG']
        assert seen==list(install_serve.BUILD_PHASES)
        raise HostExecIntercept
    monkeypatch.setattr(install_serve,'run_phase',phase)
    monkeypatch.setattr(install_serve.os,'execvpe',launch)
    with pytest.raises(HostExecIntercept):
        install_serve.main(['--config',str(path),'--log-dir',str(logs)])
    assert path.read_bytes()==before


def test_legacy_whole_capacity_cannot_reenter_through_research_flags():
    raw=configured()
    raw.update(max_num_batched_tokens=66048, history_token_capacity=16896,
               scheduler_cls="oscar_ascend.integration.bounded_whole_prefill.BoundedWholePromptAsyncScheduler")
    selected=variant_config(raw,None)
    assert 'max_num_batched_tokens' not in selected
    assert 'history_token_capacity' not in selected
    assert 'scheduler_cls' not in selected
    assert not selected['experimental_bounded_whole_prefill']
    assert not selected['experimental_whole_prefill']
    assert selected['experimental_request_windows']
