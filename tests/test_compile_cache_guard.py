"""Archive #50/#108/#169: graph-cache input ABI, not operator return schema.

The generated-code forms come from the installed torch_npu npugraph_ex
fx2acl_converter._codegen_input/_codegen_fx_forward.  The real CPU AOT case
tests alias/mutation input deduplication; it is not NPU or service evidence.
"""
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from oscar_ascend.integration.compile_cache import (
    cached_input_contract,
    load_compatible,
)


def generated_source(arity, *, forward_arity=None):
    names = ", ".join(f"input_{i}" for i in range(arity))
    if arity == 1:
        names += ","
    source = (
        "_is_first_run = True\n"
        "def kernel(*args, **kwargs):\n"
        "    global _is_first_run\n"
        "    if _is_first_run:\n"
        "        _is_first_run = False\n"
        f"        {names} = args\n"
        "    return _inner(*args, **kwargs)\n"
    )
    if forward_arity is not None:
        forward_names = ", ".join(f"input_{i}" for i in range(forward_arity))
        if forward_arity == 1:
            forward_names += ","
        source += (
            "def forward(*args, node_info=[], is_capturing=False):\n"
            f"    {forward_names} = args\n"
            "    return args\n"
        )
    return source


@pytest.mark.parametrize("arity", [1, 2, 20, 27])
def test_native_symbolic_kernel_and_forward_have_exact_input_arity(arity):
    result = cached_input_contract(generated_source(arity, forward_arity=arity))
    assert result["status"] == "verified"
    assert result["arity"] == arity
    assert isinstance(result["reason"], str)


def test_static_kernel_uses_forward_contract_and_list_assignment_is_supported():
    source = (
        "def kernel(*args, **kwargs):\n"
        "    return forward(*args, **kwargs)\n"
        "def forward(*args, node_info=[], is_capturing=False):\n"
        "    [x, y] = args\n"
        "    return (x, y)\n"
    )
    result = cached_input_contract(source)
    assert result["status"] == "verified"
    assert result["arity"] == 2


def test_nested_function_is_not_mistaken_for_kernel_input_contract():
    source = (
        "def kernel(*args, **kwargs):\n"
        "    def inner(*args):\n"
        "        x, y, z, w = args\n"
        "        return x\n"
        "    a, b = args\n"
        "    return (a, b)\n"
    )
    result = cached_input_contract(source)
    assert result["status"] == "verified" and result["arity"] == 2


@pytest.mark.parametrize("source", [
    "not python !",
    "def kernel(*args, **kwargs):\n    return args\n",
    "def kernel(*args, **kwargs):\n    head, *tail = args\n    return head\n",
    "def kernel(*args, **kwargs):\n    (a, (b, c)) = args\n    return a\n",
    "def kernel(*args, **kwargs):\n    def inner(*args):\n        x, y = args\n    return args\n",
    generated_source(20, forward_arity=21),
])
def test_unknown_or_conflicting_codegen_is_not_accepted(source):
    result = cached_input_contract(source)
    assert result["status"] == "unknown"
    assert result["arity"] is None
    assert result["reason"]


def test_contract_parsing_does_not_execute_cached_code(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"open({str(marker)!r}, 'w').write('executed')\n" + generated_source(2)
    assert cached_input_contract(source)["arity"] == 2
    assert not marker.exists()


def call_load(original, path, inputs, *, compiler=None, graph=None):
    compiler = SimpleNamespace() if compiler is None else compiler
    graph = object() if graph is None else graph
    compile_range = object()
    result = load_compatible(original, compiler, ("cache-key", str(path)),
                             graph, inputs, 7, compile_range)
    return result, compiler, graph, compile_range


def test_matching_cache_reuses_exact_handle_inputs_and_file(tmp_path):
    path = tmp_path / "graph.py"
    source = generated_source(2)
    path.write_text(source)
    inputs = [object(), object()]
    compiler, graph = object(), object()
    calls = []
    sentinel = object()

    def original(*args):
        calls.append(args)
        return sentinel

    result, _, _, compile_range = call_load(original, path, inputs, compiler=compiler, graph=graph)
    assert result is sentinel
    assert len(calls) == 1
    assert calls[0] == (compiler, ("cache-key", str(path)), graph, inputs, 7, compile_range)
    assert calls[0][3] is inputs
    assert path.read_text() == source


@pytest.mark.parametrize("source", [generated_source(20), "broken syntax !", "def kernel(*args):\n    return args\n"])
def test_bad_input_abi_uses_native_cache_miss_without_deleting_or_editing(tmp_path, source):
    path = tmp_path / "graph.py"
    path.write_text(source)
    inputs = [object() for _ in range(21)]
    before = list(inputs)
    calls = []

    def original(*args):
        calls.append(args)
        return "fresh native compiled wrapper"

    result, compiler, graph, compile_range = call_load(original, path, inputs)
    assert result == "fresh native compiled wrapper"
    assert len(calls) == 1
    assert calls[0] == (compiler, ("cache-key", None), graph, inputs, 7, compile_range)
    assert calls[0][3] is inputs and inputs == before
    assert path.read_text() == source


@pytest.mark.parametrize("path_kind", ["missing", "none"])
def test_original_cache_miss_keeps_native_behavior(tmp_path, path_kind):
    path = str(tmp_path / "missing.py") if path_kind == "missing" else None
    inputs = [object()]
    calls = []
    compiler, graph, compile_range = object(), object(), object()
    handle = ("cache-key", path)
    sentinel = object()

    def original(*args):
        calls.append(args)
        return sentinel

    assert load_compatible(original, compiler, handle, graph, inputs, 7, compile_range) is sentinel
    assert calls == [(compiler, handle, graph, inputs, 7, compile_range)]


def test_native_compilation_error_propagates_once(tmp_path):
    path = tmp_path / "graph.py"
    path.write_text(generated_source(2))
    calls = []
    failure = RuntimeError("native compilation failed")

    def original(*args):
        calls.append(args)
        raise failure

    with pytest.raises(RuntimeError) as caught:
        call_load(original, path, [object(), object(), object()])
    assert caught.value is failure
    assert len(calls) == 1


def test_loaded_runtime_error_is_never_swallowed_or_retried(tmp_path):
    path = tmp_path / "graph.py"
    path.write_text(generated_source(2))
    loads, executions = [], []
    failure = ValueError("a later real graph error")

    def runtime(*args):
        executions.append(args)
        raise failure

    def original(*args):
        loads.append(args)
        return runtime

    inputs = [object(), object()]
    runnable, *_ = call_load(original, path, inputs)
    with pytest.raises(ValueError) as caught:
        runnable(*inputs)
    assert caught.value is failure
    assert len(loads) == len(executions) == 1


def test_real_cpu_aot_alias_mutation_requires_outer_wrapper(tmp_path):
    """Torch's actual AOT inner signature differs from its public call signature.

    This is the native-cache save-inner/load-outer failure mode, reproduced
    without importing torch_npu or running an NPU/benchmark.
    """
    from torch._functorch.aot_autograd import aot_module_simplified
    from torch.fx import symbolic_trace

    class MutatingModel(torch.nn.Module):
        def forward(self, x, alias, z):
            x.add_(1)
            return (x + alias + z,)

    inner = {}

    def compiler(graph, example_inputs):
        inner["arity"] = len(example_inputs)
        inner["runnable"] = graph.forward
        return graph.forward

    x, z = torch.ones(2), torch.ones(2)
    inputs = [x, x, z]
    graph = symbolic_trace(MutatingModel())
    fresh_wrapper = aot_module_simplified(
        graph, inputs, fw_compiler=compiler,
        keep_inference_input_mutations=True,
    )
    assert len(inputs) == 3 and inner["arity"] == 2

    source = generated_source(inner["arity"])
    namespace = {"_inner": inner["runnable"]}
    exec(source, namespace)
    with pytest.raises(ValueError, match="too many values to unpack.*expected 2"):
        namespace["kernel"](*inputs)
    torch.testing.assert_close(x, torch.ones(2))

    path = tmp_path / "graph.py"
    path.write_text(source)
    calls = []

    def native_load(compiler_obj, handle, passed_graph, passed_inputs, index, compile_range):
        calls.append(handle)
        assert passed_graph is graph and passed_inputs is inputs
        if handle[1] is None:
            return fresh_wrapper
        pytest.fail("raw inner AOT artifact must not accept the outer input tuple")

    runnable, *_ = call_load(native_load, path, inputs, graph=graph)
    output = runnable(*inputs)
    assert calls == [("cache-key", None)]
    assert inputs[0] is inputs[1] is x
    torch.testing.assert_close(output[0], torch.full((2,), 5.0))
    torch.testing.assert_close(x, torch.full((2,), 2.0))
    torch.testing.assert_close(z, torch.ones(2))
    assert path.read_text() == source


@pytest.mark.parametrize("passive", [False, True])
def test_compiler_hook_installs_only_for_oscar_and_restores_exact_descriptor(tmp_path, monkeypatch, passive):
    from oscar_ascend import plugin
    from oscar_ascend.integration.runtime_api import clear_runtime

    plugin.unregister()
    clear_runtime()
    module = types.ModuleType("vllm_ascend.compilation.compiler_interface")
    seen = []

    class AscendCompiler:
        def load(self, handle, graph, example_inputs, graph_index, compile_range):
            seen.append(handle)
            return "native"

    module.AscendCompiler = AscendCompiler
    original = AscendCompiler.__dict__["load"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # Limit this fixture to the compiler seam; unrelated native modules are
    # neither imported nor simulated by these cache-ABI tests.
    monkeypatch.setattr(plugin, "_callbacks", {module.__name__: plugin._patch_compiler_cache})
    monkeypatch.setenv("OSCAR_ENABLED", "0" if passive else "1")
    monkeypatch.setenv("OSCAR_PASSIVE_TIMING_CONTROL", str(tmp_path / "control.json"))
    path = tmp_path / "cache.py"
    path.write_text(generated_source(2))
    try:
        if passive:
            plugin.register_passive_observer()
            assert AscendCompiler.__dict__["load"] is original
        else:
            plugin.register()
            replacement = AscendCompiler.__dict__["load"]
            assert replacement is not original
            plugin.register()
            assert AscendCompiler.__dict__["load"] is replacement
        assert AscendCompiler().load(("key", str(path)), object(), [1, 2, 3], 0, object()) == "native"
        assert seen == [("key", str(path) if passive else None)]
    finally:
        plugin.unregister()
        clear_runtime()
    assert AscendCompiler.__dict__["load"] is original
