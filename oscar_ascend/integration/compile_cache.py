"""Archive #169/#28/#34: validate cached FX input ABI before native loading.

AscendCompiler persists inner AOT code, while load receives outer graph inputs.
AOT can legally deduplicate aliased inputs. A bare inner kernel must not be
called with the outer argument list. Rejected cache entries use the native
cache-miss compiler path, retaining its full AOT wrapper and NPU graph backend.
No execution exception is caught, no arguments are dropped, no eager fallback.
"""
from __future__ import annotations

import ast
from pathlib import Path


def cached_input_contract(source):
    """Read current native codegen's fixed ``a, b, ... = args`` contract."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError) as error:
        return {"status": "unknown", "arity": None, "reason": f"invalid_python:{type(error).__name__}"}
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in {"kernel", "forward"}}
    if "kernel" not in functions:
        return {"status": "unknown", "arity": None, "reason": "kernel_missing"}
    counts = set()
    unsupported = False

    class Inputs(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # An inner helper's args are not the generated kernel's inputs.
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Assign(self, node):
            nonlocal unsupported
            if isinstance(node.value, ast.Name) and node.value.id == "args":
                for target in node.targets:
                    if isinstance(target, (ast.Tuple, ast.List)):
                        if not target.elts or not all(isinstance(item, ast.Name) for item in target.elts):
                            unsupported = True
                        else:
                            counts.add(len(target.elts))
            self.generic_visit(node)

    for function in functions.values():
        if function.args.vararg is None or function.args.vararg.arg != "args":
            unsupported = True
        visitor = Inputs()
        for statement in function.body:
            visitor.visit(statement)
    if unsupported or len(counts) != 1:
        return {"status": "unknown", "arity": None, "reason": "ambiguous_or_missing_input_unpack"}
    return {"status": "verified", "arity": counts.pop(), "reason": "fixed_args_unpack"}


def load_compatible(original, compiler, handle, graph, example_inputs, graph_index, compile_range):
    """Keep native cache hits, rebuild incompatible cache before executing it."""
    if not isinstance(handle, (tuple, list)) or len(handle) != 2:
        return original(compiler, handle, graph, example_inputs, graph_index, compile_range)
    key, path = handle
    if not path or not Path(path).is_file():
        return original(compiler, handle, graph, example_inputs, graph_index, compile_range)
    try:
        source = Path(path).read_text()
    except FileNotFoundError:
        contract = {"status": "unknown", "arity": None, "reason": "cache_disappeared"}
    except UnicodeDecodeError:
        contract = {"status": "unknown", "arity": None, "reason": "invalid_encoding"}
    else:
        contract = cached_input_contract(source)
    expected = len(example_inputs)
    if contract["status"] == "verified" and contract["arity"] == expected:
        return original(compiler, handle, graph, example_inputs, graph_index, compile_range)
    reason = "input_arity" if contract["status"] == "verified" else contract["reason"]
    print(f"[oscar] COMPILE_CACHE_REBUILD reason={reason} cached_inputs={contract['arity']} "
          f"current_inputs={expected} graph={graph_index} path={path} "
          "route=native_compiler graph=retained", flush=True)
    # Native AscendCompiler.load explicitly recompiles an absent handle path.
    # It deep-copies the FX graph, re-wraps FakeTensors and restores AOT's
    # argument adapter. Leave the existing file and all input objects intact.
    return original(compiler, (key, None), graph, example_inputs, graph_index, compile_range)
