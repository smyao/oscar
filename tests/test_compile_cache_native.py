"""Archive #169: execute the pinned native cache-miss method, without NPU."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from oscar_ascend.integration.compile_cache import load_compatible


def test_rejected_inner_cache_reenters_actual_native_load_and_keeps_aot_adapter(tmp_path):
    from torch._functorch.aot_autograd import aot_module_simplified
    from torch.fx import symbolic_trace

    class Model(torch.nn.Module):
        def forward(self, x, alias, z):
            x.add_(1)
            return (x + alias + z,)

    source = Path(__file__).resolve().parents[1] / 'references/vllm-ascend/vllm_ascend/compilation/compiler_interface.py'
    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'AscendCompiler')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'load')
    calls = []

    def native_backend(graph, inputs, config, vllm_config, ascend_config, compile_range, key, cache_dir):
        calls.append((graph, inputs, config, compile_range, key, cache_dir))
        return aot_module_simplified(graph, inputs, fw_compiler=lambda graph, inputs: graph.forward,
                                     keep_inference_input_mutations=True), (key, None)

    namespace = {'os': os, 'copy': copy, 'torch': torch,
                 'logger': SimpleNamespace(info=lambda *args: None),
                 'get_ascend_config': lambda: SimpleNamespace(ascend_compilation_config=object()),
                 'npugraph_ex_compile': native_backend}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), namespace)
    path = tmp_path / 'inner.py'
    path.write_text('def kernel(*args, **kwargs):\n    x, z = args\n    return (x + z,)\n')
    compiler = SimpleNamespace(vllm_config=object(), cache_dir=str(tmp_path))
    x, z = torch.ones(2), torch.ones(2)
    inputs = [x, x, z]
    graph = symbolic_trace(Model())
    runnable = load_compatible(namespace['load'], compiler, ('key', str(path)), graph, inputs, 0, 'range')
    assert len(calls) == 1 and calls[0][0] is not graph
    assert calls[0][1] is inputs and calls[0][2:] == ({}, 'range', 'key', str(tmp_path))
    torch.testing.assert_close(runnable(*inputs)[0], torch.full((2,), 5.0))
    torch.testing.assert_close(x, torch.full((2,), 2.0))
    assert path.is_file()
