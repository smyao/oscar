"""Archive #13/#126/#145/#148/#154/#155: precise row-check batching.

Compile the actual helper with a synchronous host API model. Compare every
ReduceSum input/order and scalar status to the original one-row schedule.
This is neither a CANN overload check nor asynchronous NPU evidence.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "csrc/kernels/attention_bf16_precise_check.h"

STUB = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#define __aicore__
struct bfloat16_t { uint16_t bits; };
struct Trace {
  int casts=0, barriers=0, scalarIn=0, scalarOut=0;
  std::vector<std::vector<uint32_t>> reductions;
} trace;
namespace AscendC {
enum class TPosition { VECCALC };
enum class HardEvent { V_S, S_V };
enum class RoundMode { CAST_NONE };
enum Pipe { PIPE_V };
template<class T> struct LocalTensor {
  T* ptr;
  LocalTensor<T> operator[](int32_t n) const { return {ptr+n}; }
  T GetValue(int32_t n) const { return ptr[n]; }
};
template<TPosition> struct TBuf {
  std::vector<float> data;
  explicit TBuf(int32_t count):data(count+8,1234567.0F) {}
  template<class T> LocalTensor<T> Get() { return {reinterpret_cast<T*>(data.data()+4)}; }
  void canary() const {
    for(int i=0;i<4;++i)assert(data[i]==1234567.0F && data[data.size()-1-i]==1234567.0F);
  }
};
template<Pipe> void PipeBarrier() { ++trace.barriers; }
void Cast(LocalTensor<float> out,LocalTensor<bfloat16_t> in,RoundMode,int32_t n) {
  ++trace.casts;
  for(int i=0;i<n;++i) {uint32_t bits=uint32_t(in.ptr[i].bits)<<16;std::memcpy(out.ptr+i,&bits,4);}
}
void ReduceSum(LocalTensor<float> out,LocalTensor<float> in,LocalTensor<float> work,int32_t n) {
  assert(n==256 && work.ptr==out.ptr+16);
  std::vector<uint32_t> values(n);std::memcpy(values.data(),in.ptr,n*4);
  trace.reductions.push_back(values);
  // Same deterministic FP32 model for both schedules. Input identity, not
  // this host summation tree, proves the unchanged device reduction call.
  float sum=0.0F;
  for(int i=0;i<n;++i)sum=sum+in.ptr[i];
  out.ptr[0]=sum;
}
}
namespace oscar_ascend_device {
template<AscendC::HardEvent event> void Fence() {
  if(event==AscendC::HardEvent::V_S)++trace.scalarIn;else ++trace.scalarOut;
}
bool Finite(float x) { return x==x && x<=3.402823466e38F && x>=-3.402823466e38F; }
}
"""

HARNESS = r"""
using namespace AscendC;
using oscar_ascend_device::Fence;
using oscar_ascend_device::Finite;
int main(int argc,char** argv) {
  assert(argc==3);const int rows=std::atoi(argv[1]);const bool tagged=std::atoi(argv[2]);
  assert(rows>=0 && rows<=64);
  for(int incoming: {0,1,2,3,4,5})for(int seed=0;seed<8;++seed) {
    std::vector<bfloat16_t> source(64*256);
    std::vector<int64_t> tags(64);
    for(int r=0;r<64;++r) {
      tags[r]=(seed>=4 && ((r+seed)%3==0))?-1:91+r;
      for(int d=0;d<256;++d) {
        // The first four seeds span every raw BF16 encoding. Tails are
        // poison and must not be reduced, even when the last tile is short.
        uint16_t bits=uint16_t(seed*16384+r*256+d);
        if(seed==4)bits=0x7f7f; // finite inputs with an overflowing FP32 sum
        if(seed==5)bits=(d%2)?0xff7f:0x7f7f;
        if(seed==6)bits=(d%2)?0x8000:0;
        if(seed==7)bits=(r%2)?0x7fc1:0x3f80;
        if(r>=rows)bits=0x7fc1;
        if(tagged && tags[r]<0)bits=0; // producer does not DMA invalid tags
        source[r*256+d].bits=bits;
      }
    }
    const auto original=source;
    TBuf<TPosition::VECCALC> rowBuf(256),scratch(512),natural(16*272/2);
    auto one=rowBuf.Get<float>(),check=scratch.Get<float>();
    int expected=incoming;trace={};
    for(int r=0;r<rows;++r) {
      if(tagged && tags[r]!=91+r) {expected=4;continue;}
      Cast(one,{source.data()+r*256},RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();
      ReduceSum(check,one,check[16],256);Fence<HardEvent::V_S>();
      if(!Finite(check.GetValue(0)))expected=2;
      Fence<HardEvent::S_V>();
    }
    const auto old=trace;trace={};int actual=incoming;
    if(tagged)oscar_bf16_precise_check::CheckTaggedRows<256,16>(
        {source.data()},{tags.data()},91,rows,natural,scratch,actual);
    else oscar_bf16_precise_check::CheckRows<256,16>(
        {source.data()},rows,natural,scratch,actual);
    assert(actual==expected);
    assert(trace.reductions==old.reductions);
    assert(trace.scalarIn==old.scalarIn && trace.scalarOut==old.scalarOut);
    assert(trace.casts==(rows+3)/4 && trace.barriers==trace.casts);
    if(!tagged && rows>1)assert(trace.casts<old.casts);
    assert(std::memcmp(source.data(),original.data(),source.size()*sizeof(bfloat16_t))==0);
    natural.canary();scratch.canary();rowBuf.canary();
    // Batch Cast may touch only actual rows, never the poisoned tile tail.
    const int count=rows==0?0:(rows-1)%4+1;
    for(int i=count*256;i<int(natural.data.size())-8;++i)
      if(rows<=4)assert(natural.Get<float>().ptr[i]==1234567.0F);
    if(!tagged && seed==4 && rows)assert(actual==2);
    if(rows==0)assert(actual==incoming);
  }
}
"""


@pytest.fixture(scope="module")
def helper_program(tmp_path_factory):
    compiler = os.environ.get("OSCAR_TEST_CLANGXX") or shutil.which("clang++")
    if not compiler:
        pytest.skip("clang++ unavailable: real-header host schedule not run")
    source = HEADER.read_text().replace('#include "oscar_common.h"', STUB).replace("#pragma once", "")
    path = tmp_path_factory.mktemp("bf16-precise") / "check"
    result = subprocess.run([compiler, "-std=c++17", "-O0", "-fno-fast-math",
                             "-x", "c++", "-", "-o", str(path)],
                            input=source + HARNESS, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return path


@pytest.mark.parametrize("rows", [0, 1, 3, 4, 5, 15, 16, 17, 31, 32, 63, 64])
@pytest.mark.parametrize("tagged", [False, True], ids=["current", "sink-recent-tags"])
def test_actual_header_preserves_every_reduction_status_and_bounds(helper_program, rows, tagged):
    result = subprocess.run([str(helper_program), str(rows), str(int(tagged))],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_minimum_existing_natural_buffers_fit_without_new_allocation():
    for natural_rows in (16, 32, 64):
        assert 4 * 256 * 4 <= natural_rows * (256 + 16) * 2
    source = HEADER.read_text()
    assert "InitBuffer" not in source and "GlobalTensor" not in source
    assert "ReduceSum(check,row,check[16],D)" in source
    assert "if(!Finite(check.GetValue(0)))error=2;" in source
