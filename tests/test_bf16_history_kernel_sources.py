"""Archive G4/#81-83/#126/#145/#148/#154/#157: source/resource contracts.

These checks prove generation identity, typed publication boundaries and bounded
address arithmetic. They do not validate MatmulImpl Reset/re-Init, CANN, NPU
ordering or BF16 quality; the separate stage1 numerical gate remains failed.
"""
from __future__ import annotations

import ast
import operator
import re
from pathlib import Path

import pytest

from tools.experiments.generate_bf16_history_kernels import ROOT, SPECS, derive


def body(source, signature):
    start = source.index("{", source.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1:end - 1]


def integer_expression(expression, names):
    """Only the integer grammar used by actual InitBuffer size expressions."""
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.floordiv}
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        raise AssertionError(f"unreviewed kernel size expression: {expression}")
    return visit(ast.parse(expression, mode="eval"))


# Independent D256 allocation expectations, including the 32-byte-aligned
# explicit vector buffers. TPipe/SDK-reserved UB is excluded by the 184KiB cap.
EXPECTED = {
    "mdl": (1146880, 1720320, 158656, 175040),
    "member_pipeline_cluster4": (2494464, 3741696, 157760, 174144),
    "window_range_cluster16": (4997120, 7495680, 157632, 174016),
    "window_range": (917504, 1376256, 157632, 174016),
    "window_range_balanced": (917504, 1376256, 157632, 174016),
}


@pytest.mark.parametrize("name,mother,sha", SPECS)
def test_inverse_generation_preserves_exact_fixed_mother(name, mother, sha):
    generated, _ = derive(name, mother, sha)  # Includes reverse-transform identity.
    path = ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp"
    assert path.read_text() == generated
    # Error/tag checks and ownership classification are untouched whole bodies.
    original = (ROOT / "csrc/kernels" / mother).read_text()
    for signature in ("void LoadTask(", "bool TaskValid(", "int64_t LogicalPosition(",
                      "void PublishRows(", "void PublishStatus("):
        assert body(generated, signature) == body(original, signature)


@pytest.mark.parametrize("name,mother,sha", SPECS)
def test_only_history_operands_round_and_fp32_result_rotation_are_retained(name, mother, sha):
    source = (ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp").read_text()
    publisher = body(source, "void PublishOperand(")
    assert "if(kind==0)oscar_bf16_history::Publish(operandWork[offset],source,operandBuf,elements);" in publisher
    assert "else DataCopy(work[offset],source,elements);" in publisher
    assert "PublishOperand(qOffset+row*D,dst,D)" in source
    assert "PublishOperand(qOffset+row*D,zeros,count*D)" in source
    assert "PublishOperand((value?vOffset:kOffset)+slot*D,src,kElements)" in source
    assert "PublishOperand(pOffset+rowBase*kKvRows,scores,activeRows*kKvRows)" in source
    assert "DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D)" in source
    assert not re.search(r"Publish(?:History)?Operand\([^;]*\bacc\b", source)
    assert "using HistoryA=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,false>" in source
    assert "using HistoryB=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,true>" in source
    assert "GetMatmulApiTiling<HistoryA,HistoryB,Matrix,Matrix>" in source
    assert "using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>" in source
    dot = body(source, "void Matmul(")
    assert "const bool history=kind==0 && !rotation;" in dot
    assert dot.index("SelectMatmul(history,") < dot.index("if(history)")
    assert "historyMm.template IterateAll<true>(work[c]);" in dot
    assert "if(rotation) mm.SetTensorB(rv,true);else mm.SetTensorB(work[b],transposeB);" in dot
    assert "mm.template IterateAll<true>(work[c]);" in dot
    if name == "member_pipeline_cluster4":
        atomic = body(source, "void MatmulAtomic(")
        assert "SelectMatmul(kind==0," in atomic
        assert "historyMm.template IterateAll<true>(work[c],static_cast<uint8_t>(1));" in atomic
        assert "mm.template IterateAll<true>(work[c],static_cast<uint8_t>(1));" in atomic


def test_cluster4_history_publisher_preserves_both_old_lane_domains():
    source = (ROOT / "csrc/kernels/attention_bf16_history_member_pipeline_cluster4.cpp").read_text()
    publish = body(source, "void PublishHistoryOperand(")
    assert "PublishOperand(offset,source,kElements);" in publish
    assert "PublishOperand(offset+2*kElements,source[kElements],kElements);" in publish
    assert "else DataCopy(work[offset],source,originalCopy);" in publish
    coverage = {}
    for subtile in range(4):
        for lane in range(2):
            for row in range(32):
                # One lane's input order is its old H16 block, then the next
                # old H16 block; restore the global natural [KV,D] order.
                destination = subtile * 64 + lane * 16 + (row // 16) * 32 + row % 16
                assert destination not in coverage
                coverage[destination] = (subtile, lane, row)
    assert sorted(coverage) == list(range(256))


@pytest.mark.parametrize("name,mother,sha", SPECS)
def test_actual_buffers_and_per_core_arenas_fit_the_declared_resource_limits(name, mother, sha):
    source = (ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp").read_text()
    init = body(source, "void Init(")
    values = {"D": 256, "kHalfKv": 64 if name == "mdl" else 16,
              "kHalfRows": 16 if name == "mdl" else 32, "kKvRows": 256,
              "kLaneRows": 16 if name == "mdl" else 64,
              "kHistoryHalfKv": 32, "kHistoryElements": 8192, "kPackedStride": 160}
    values["kElements"] = values["kHalfKv"] * 256
    values["kWords"] = values["kElements"] // 8
    sizes = re.findall(r"pipe.InitBuffer\(\w+,([^;]*)\);", init)
    total = 0
    for expression in sizes:
        expression = expression.replace("oscar_ascend_fast::Geometry<D>::planeBytes",
                                         str(8 * (values["kWords"] + 16) * 2))
        expression = expression.replace("oscar_bf16_history::kOperandScratchBytes", "16384")
        size = integer_expression(expression, values)
        assert size % 32 == 0
        total += size
    old_bytes, candidate_bytes, old_ub, candidate_ub = EXPECTED[name]
    assert total == candidate_ub == old_ub + 16384
    assert total <= 184 * 1024
    assert "kOriginalVectorBytes+oscar_bf16_history::kOperandScratchBytes<=184*1024" in source
    assert "auto coreWorkspace=workspace+core*floatsPerCore*6;" in init
    assert "work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coreWorkspace));" in init
    assert "coreWorkspace+floatsPerCore*4" in init
    floats = old_bytes // 4
    assert candidate_bytes == floats * 6
    for core in range(20):
        fp32 = (core * candidate_bytes, core * candidate_bytes + old_bytes)
        bf16 = (fp32[1], fp32[1] + floats * 2)
        assert fp32[1] == bf16[0]
        assert bf16[1] == (core + 1) * candidate_bytes
        assert all(offset % 32 == 0 for offset in (*fp32, *bf16))
    # The full original index space is mirrored, including C4 ping/pong slots.
    last_bf16_element_end = old_bytes + (floats - 1) * 2 + 2
    assert last_bf16_element_end == candidate_bytes


def test_shared_helper_rounding_and_resource_switch_order_are_explicit():
    source = (ROOT / "csrc/kernels/attention_bf16_history_operands.h").read_text()
    assert "RoundMode::CAST_RINT" in body(source, "void Publish(")
    select = body(source, "void SelectMatmul(")
    assert select.index("if (history == historyActive) return;") < select.index(".End()")
    assert select.index(".End()") < select.index("PipeBarrier<PIPE_ALL>()") < select.index("pipe.Reset()")
    assert select.index("pipe.Reset()") < select.index("bf16.Init(")
    assert "else fp32.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);" in select
    # Native source establishes this mode/type call exists; no inference from
    # enum spelling or unrelated GDN code is used as compiler acceptance.
    native = (ROOT / "references/vllm-ascend/csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h").read_text()
    assert re.search(r"IsSameType<OUT_T, bfloat16_t>::value[^}]+RoundMode::CAST_RINT", native)


def test_every_new_project_local_include_exists_and_workspace_families_are_declared():
    paths = [ROOT / "csrc/kernels" / f"attention_bf16_history_{name}.cpp" for name, _, _ in SPECS]
    paths.extend((ROOT / "csrc/kernels/attention_bf16_history_operands.h",
                  ROOT / "csrc/include/oscar_bf16_history_launch.h"))
    for path in paths:
        for include in re.findall(r'^#include "([^"]+)"', path.read_text(), re.MULTILINE):
            if include.startswith(("lib/", "adv_api/")):
                continue  # Official SDK is unavailable here, explicitly unvalidated.
            assert (path.parent / include).is_file(), (path, include)
    header = paths[-1].read_text()
    for family, expected in (("", 1376256), ("mdl_", 1720320),
                             ("member_pipeline_", 3741696), ("cluster16_", 7495680)):
        assert f"attention_bf16_history_{family}workspace_per_core(256)=={expected}" in header
