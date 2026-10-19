"""Archive #157/#163/#165/#166/#167: pre-replay sync and queued dump boundaries.

Execute selected read-only native AST methods with host scheduling doubles.
These tests establish control flow, never the source of a target NPU Trap.
No native file, production guard or serving readback path is changed.
"""
import ast
from collections import deque
from contextlib import contextmanager, nullcontext
from enum import Enum
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from oscar_ascend.ops.cv_dispatch import select_cv_op
from tools import target_cli
from tools.serving_variants import variant_config

ROOT = Path(__file__).resolve().parents[1]
ASCEND = ROOT / 'references/vllm-ascend/vllm_ascend'
VLLM = ROOT / 'references/vllm/vllm'


class GraphMode(Enum):
    NONE = 0
    FULL = 1

    @staticmethod
    def valid_runtime_modes():
        return {GraphMode.NONE, GraphMode.FULL}


def native_method(path, owner, name, **globals_):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    namespace = {'__name__': 'native_host_contract', **globals_}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('prior_failure', [False, True])
def test_real_acl_graph_sync_precedes_replay_and_failure_never_submits_graph(prior_failure):
    events = []
    def synchronize():
        events.append('synchronize')
        if prior_failure:raise RuntimeError('previous async device failure')
    call = native_method(ASCEND/'compilation/acl_graph.py', 'ACLGraphWrapper', '__call__',
        CUDAGraphMode=GraphMode,
        get_forward_context=lambda: SimpleNamespace(batch_descriptor='small-decode', cudagraph_runtime_mode=GraphMode.FULL),
        _EXTRA_CTX=SimpleNamespace(is_draft_model=False),
        torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=synchronize))),
        logger=SimpleNamespace(info_once=lambda *a: events.append('replaying_log')))
    entry = SimpleNamespace(aclgraph=SimpleNamespace(replay=lambda: events.append('replay')), output='output')
    wrapper = SimpleNamespace(runtime_mode=GraphMode.FULL, enable_enpu=False,
        use_eagle=True, is_debugging_mode=False, concrete_aclgraph_entries={'small-decode': entry})
    if prior_failure:
        with pytest.raises(RuntimeError, match='previous async'):call(wrapper)
        assert events == ['replaying_log', 'synchronize']
    else:
        assert call(wrapper) == 'output'
        assert events == ['replaying_log', 'synchronize', 'replay']


def test_dumped_8232_token_chunk_cannot_be_the_512_max_graph_replay_batch():
    dispatch = native_method(VLLM/'v1/cudagraph_dispatcher.py', 'CudagraphDispatcher', 'dispatch',
        CUDAGraphMode=GraphMode, BatchDescriptor=lambda tokens: SimpleNamespace(num_tokens=tokens))
    dispatcher = SimpleNamespace(keys_initialized=True, cudagraph_mode=GraphMode.FULL,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=512))
    mode, descriptor = dispatch(dispatcher, num_tokens=8232, uniform_decode=False)
    assert mode == GraphMode.NONE and descriptor.num_tokens == 8232
    # This establishes conditional CV selection for that dumped geometry;
    # it does not say that this CV family caused the asynchronous failure.
    assert select_cv_op(16, 6, 1, 8232, 8232, fast_unpack=True, mixed_cv=True,
        striped_cache=True, decode_bundle=True, c4_member_pipeline=True,
        mdl_decode=True, bf16_history=True) == 'attention_bf16_history_window_range_cluster16_out'


def test_native_engine_dump_binds_oldest_awaited_future_not_newest_queued_batch():
    dumped = []
    error_detail = native_method(VLLM/'v1/engine/core.py', 'EngineCore', 'log_error_detail',
        contextmanager=contextmanager,
        dump_engine_exception=lambda config, output, stats: dumped.append((output, stats)))
    step = native_method(VLLM/'v1/engine/core.py', 'EngineCore', 'step_with_batch_queue')
    old = SimpleNamespace(context=16384, tokens=8232)
    newer = SimpleNamespace(context=24616, tokens=4)
    def failed_result():raise RuntimeError('asynchronous worker output failed')
    pending = deque([
        (SimpleNamespace(result=lambda: None), newer, None),
        (SimpleNamespace(result=failed_result), old, None)], maxlen=3)
    engine = SimpleNamespace(batch_queue=pending, batch_queue_size=3,
        scheduler=SimpleNamespace(has_requests=lambda: False, make_stats=lambda: 'current-live-stats'),
        vllm_config=object(), log_iteration_details=lambda _: nullcontext())
    engine.log_error_detail = MethodType(error_detail, engine)
    with pytest.raises(RuntimeError, match='asynchronous worker'):step(engine)
    assert dumped == [(old, 'current-live-stats')]
    assert pending[0][1] is newer


@pytest.mark.parametrize('draft_eager', [False, True])
def test_diagnose_trap_disables_both_native_target_and_drafter_graph_selection(draft_eager):
    selected = json.loads((ROOT/'configs/target.json').read_text())
    selected['diagnose_guard'] = True
    selected['speculative_config']['enforce_eager'] = draft_eager
    argv = target_cli.serve_argv(variant_config(selected, 'candidate'))
    assert '--enforce-eager' in argv
    native_use = native_method(ASCEND/'worker/model_runner_v1.py', 'NPUModelRunner', '_use_aclgraph',
        CUDAGraphMode=GraphMode, CompilationMode=SimpleNamespace(VLLM_COMPILE=3))
    runner = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=GraphMode.FULL, mode=3),
        model_config=SimpleNamespace(enforce_eager=True))
    runner._use_aclgraph = MethodType(native_use, runner)
    # Execute the actual native constructor expression, including the runner
    # term. It prevents draft graphs even with a custom draft_eager=False.
    tree = ast.parse((ASCEND/'spec_decode/llm_base_proposer.py').read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Attribute) and target.attr == 'use_cuda_graph' for target in node.targets))
    expression = ast.Expression(assignment.value)
    proposer = SimpleNamespace(runner=runner, speculative_config=SimpleNamespace(enforce_eager=draft_eager))
    assert native_use(runner) is False
    assert eval(compile(expression, '<native-use-cuda-graph>', 'eval'), {}, {'self': proposer}) is False
