"""Archive #164: compile the real helper's C++ templates with declaration stubs.

This checks language syntax only, not CANN API overloads, code generation,
official CPU-debug or device execution. The negative control must reproduce
the target's missing-dependent-template diagnostic.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "csrc/kernels/attention_striped_unpack_bf16.h"

# No SDK or standard-library include is needed for the language regression.
# Variadic vector declarations deliberately do not claim API compatibility.
DECLARATIONS = r"""
#define __aicore__
using int16_t = short;
using int32_t = int;
using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
struct half {};
struct bfloat16_t {};
namespace AscendC {
enum class TPosition { VECCALC };
enum class HardEvent { S_V, V_S };
enum class RoundMode { CAST_NONE, CAST_RINT, CAST_ROUND };
enum class CMPMODE { GT, LT, GE };
enum Pipe { PIPE_V };
template<class T> struct LocalTensor {
  LocalTensor<T> operator[](int32_t) const;
  template<class U> LocalTensor<U> ReinterpretCast() const;
  T GetValue(int32_t) const;
  void SetValue(int32_t, T);
};
template<TPosition> struct TBuf {
  template<class T> LocalTensor<T> Get();
};
struct UnaryRepeatParams { uint16_t dst, src; uint8_t dstRep, srcRep; };
struct BinaryRepeatParams { uint8_t dst, src0, src1, dstRep, src0Rep, src1Rep; };
template<Pipe> void PipeBarrier();
template<class... A> void Duplicate(A...);
template<class... A> void ShiftRight(A...);
template<class... A> void And(A...);
template<class... A> void Gather(A...);
template<class... A> void Cast(A...);
template<class T, class U, class... A> void Compares(A...);
template<class... A> void Mul(A...);
template<class... A> void Add(A...);
template<class... A> void Muls(A...);
template<class... A> void Adds(A...);
}
namespace oscar_ascend_device {
template<AscendC::HardEvent> void Fence();
}
"""


@pytest.fixture(scope="module")
def clangxx():
    compiler = os.environ.get("OSCAR_TEST_CLANGXX") or shutil.which("clang++")
    if not compiler:
        pytest.skip("clang++ unavailable: C++ language check not run")
    version = subprocess.run([compiler, "--version"], capture_output=True,
                             text=True, timeout=30)
    if version.returncode:
        pytest.skip("clang++ cannot run: " + version.stderr.strip())
    return compiler


@pytest.mark.parametrize("restore_missing_keyword", [False, True],
                         ids=["fixed-source", "old-error-negative-control"])
def test_real_unpack_templates_parse(clangxx, restore_missing_keyword):
    source = HELPER.read_text()
    if restore_missing_keyword:
        source = source.replace("offsets.template ReinterpretCast<uint32_t>()",
                                "offsets.ReinterpretCast<uint32_t>()")
    assert source.count('#include "oscar_common.h"') == 1
    source = source.replace('#include "oscar_common.h"', DECLARATIONS)
    source = source.replace("#pragma once", "")
    # Taking each specialization's address instantiates its complete body.
    for rows in (16, 32, 64):
        for name in ("Unpack", "InitIndices"):
            source += (f"\nauto instantiate_{name}_{rows} = "
                       f"&oscar_ascend_striped_bf16::{name}<256,{rows}>;\n")
    result = subprocess.run(
        [clangxx, "-std=c++17", "-fsyntax-only", "-x", "c++", "-"],
        input=source, capture_output=True, text=True, timeout=30)
    if restore_missing_keyword:
        assert result.returncode != 0, "old target error was not reproduced"
        assert "use 'template' keyword" in result.stderr, result.stderr
        assert "'ReinterpretCast'" in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stderr
