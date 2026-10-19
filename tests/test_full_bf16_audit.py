"""Archive #13-22/#126/#154/#167: audit the real direct-decode LUT/index tail.

The extracted source is unmodified for positive controls. Vector primitives
are scalar host models: this checks arithmetic, scratch aliasing and byte
addresses, not CANN overloads, A2 rounding/fences, or NPU/graph completion.
Frozen attention acceptance is checked separately by the failing screen; this
file never reclassifies an acceptance failure as an expected pytest success.
"""
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tools.experiments.bf16_direct_decode import independent_level_bits

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'csrc/kernels/attention_striped_unpack_bf16.h'

STUB = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <stdexcept>
struct half{uint16_t bits;};
struct bfloat16_t{uint16_t bits;};
constexpr int PIPE_V=0;
template<int>void PipeBarrier(){}
enum class RoundMode{CAST_NONE,CAST_RINT,CAST_ROUND};
struct UnaryRepeatParams{uint16_t dst,src;uint8_t dstRep,srcRep;};
template<class T>struct Tensor{
 uint8_t*p;size_t bytes;
 Tensor operator[](size_t i)const{
  if(i>bytes/sizeof(T))throw std::runtime_error("tensor offset");
  return{p+i*sizeof(T),bytes-i*sizeof(T)};}
 template<class U>Tensor<U>ReinterpretCast()const{return{p,bytes};}
 T GetValue(size_t i)const{
  if(i>=bytes/sizeof(T))throw std::runtime_error("tensor read");
  T v;std::memcpy(&v,p+i*sizeof(T),sizeof(T));return v;}
 void SetValue(size_t i,T v){
  if(i>=bytes/sizeof(T))throw std::runtime_error("tensor write");
  std::memcpy(p+i*sizeof(T),&v,sizeof(T));}
};
struct Buffer{
 std::vector<uint8_t>data;
 Buffer(size_t bytes):data(bytes,0xcd){}
 template<class T>Tensor<T>Get(){return{data.data(),data.size()};}
};
float half_float(half h){
 unsigned sign=h.bits>>15,exp=(h.bits>>10)&31,m=h.bits&1023;
 if(exp==31)throw std::runtime_error("nonfinite fixture metadata");
 float v=exp?std::ldexp(float(1024+m),int(exp)-25):std::ldexp(float(m),-24);
 return sign?-v:v;
}
uint16_t bf16_rne(float v){
 uint32_t b;std::memcpy(&b,&v,4);
 if((b&0x7fffffff)>0x7f800000)return uint16_t((b>>16)|0x40);
 return uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
void Cast(Tensor<float>d,Tensor<half>s,RoundMode,int n){
 for(int i=0;i<n;++i)d.SetValue(i,half_float(s.GetValue(i)));}
void Cast(Tensor<bfloat16_t>d,Tensor<float>s,RoundMode mode,int n){
 if(mode!=RoundMode::CAST_RINT)throw std::runtime_error("round mode");
 for(int i=0;i<n;++i)d.SetValue(i,{bf16_rne(s.GetValue(i))});}
void Cast(Tensor<int32_t>d,Tensor<float>s,RoundMode mode,int n){
 if(mode!=RoundMode::CAST_ROUND)throw std::runtime_error("index round mode");
 for(int i=0;i<n;++i){float v=s.GetValue(i);
  if(v<0 || v>3 || std::floor(v)!=v)throw std::runtime_error("index code");
  d.SetValue(i,int32_t(v));}}
void Cast(Tensor<float>d,Tensor<int16_t>s,RoundMode,uint64_t mask,uint8_t reps,UnaryRepeatParams p){
 if(p.dst!=1 || p.src!=1)throw std::runtime_error("unsupported block stride");
 for(int r=0;r<reps;++r)for(size_t i=0;i<mask;++i)
  d.SetValue(r*p.dstRep*32/sizeof(float)+i,float(s.GetValue(r*p.srcRep*32/sizeof(int16_t)+i)));}
void Duplicate(Tensor<float>d,float x,int n){for(int i=0;i<n;++i)d.SetValue(i,x);}
void Mul(Tensor<float>d,Tensor<float>a,Tensor<float>b,int n){
 for(int i=0;i<n;++i)d.SetValue(i,a.GetValue(i)*b.GetValue(i));}
void Add(Tensor<float>d,Tensor<float>a,Tensor<float>b,int n){
 for(int i=0;i<n;++i)d.SetValue(i,a.GetValue(i)+b.GetValue(i));}
void Muls(Tensor<int32_t>d,Tensor<int32_t>s,int32_t v,int n){
 for(int i=0;i<n;++i)d.SetValue(i,s.GetValue(i)*v);}
void Adds(Tensor<int32_t>d,Tensor<int32_t>s,int32_t v,int n){
 for(int i=0;i<n;++i)d.SetValue(i,s.GetValue(i)+v);}
void Gather(Tensor<half>d,Tensor<half>s,Tensor<uint32_t>offset,uint32_t base,int n){
 for(int i=0;i<n;++i){uint32_t byte=base+offset.GetValue(i);
  if(byte%2)throw std::runtime_error("unaligned gather");
  d.SetValue(i,s.GetValue(byte/2));}}
'''


def extracted_tail(mutation):
    source = HELPER.read_text()
    geometry = source[source.index('template<int32_t D,int32_t HalfKv=16> struct Geometry'):
                      source.index('\n};') + 3]
    start = source.index('  Cast(metadata,halfMetadata,RoundMode::CAST_NONE,2*HalfKv);')
    tail = source[start:source.index('\n}\n} // namespace', start)]
    if mutation == 'wrong-row-offset':
        old = 'static_cast<int32_t>(2*(begin+row))'
        assert tail.count(old) == 1
        tail = tail.replace(old, 'static_cast<int32_t>(2*row)')
    elif mutation == 'live-metadata-alias':
        old = 'auto table=metadata[4*HalfKv];'
        assert tail.count(old) == 1
        tail = tail.replace(old, 'auto table=metadata[HalfKv];')
    elif mutation != 'none':
        raise ValueError(mutation)
    return geometry, tail


def executable_source(rows, mutation):
    geometry, tail = extracted_tail(mutation)
    return STUB + geometry + f'\nconstexpr int D=256,HalfKv={rows};\n' + r'''
int main(){try{
 using G=Geometry<D,HalfKv>;
 Buffer scratchBuf(G::scratchBytes),indexScratchBuf(G::indexScratchBytes);
 Buffer halfBuf(4*HalfKv),naturalBuf(G::naturalBytes),output(G::outputBytes+32);
 auto metadata=scratchBuf.Get<float>();auto halfMetadata=halfBuf.Get<half>();
 auto natural=naturalBuf.Get<int16_t>();auto values=output.Get<bfloat16_t>();
 constexpr int naturalStride=G::naturalStride;
 for(;;){
  if(std::fread(halfBuf.data.data(),1,halfBuf.data.size(),stdin)!=halfBuf.data.size())break;
  for(int row=0;row<HalfKv;++row)for(int dim=0;dim<D;++dim){
   uint8_t code;if(std::fread(&code,1,1,stdin)!=1)return 8;
   natural.SetValue(row*naturalStride+dim,code);}
''' + tail + r'''
  for(size_t i=G::outputBytes;i<output.data.size();++i)if(output.data[i]!=0xcd)return 9;
  if(std::fwrite(output.data.data(),1,G::outputBytes,stdout)!=G::outputBytes)return 10;
 }
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 11;}}
'''


def fixture(rows):
    # Fixed finite metadata across all exponents, signed zeros, subnormals,
    # cancellation, and exact BF16 halfway cases; deliberately distinct rows.
    rng = np.random.default_rng(719)
    size = 2048
    scales = rng.integers(1, 0x7c00, size, dtype=np.uint16)
    zeros = rng.integers(0, 0x7c00, size, dtype=np.uint16)
    zeros |= rng.integers(0, 2, size, dtype=np.uint16) << 15
    scales[:8] = [1, 0x3ff, 0x400, 0x3c00, 0x3e00, 0x7bff, 0x3c00, 0x3c00]
    zeros[:8] = [0, 0x8000, 0x8001, 0xbc00, 0x7bff, 0xfbff, 0x8000, 0x3c04]
    codes = ((np.arange(size)[:, None]*13 + np.arange(256)*7) % 4).astype(np.uint8)
    table = independent_level_bits(scales, zeros)
    expected = table[np.arange(size)[:, None], codes]
    binary = bytearray()
    for first in range(0, size, rows):
        binary.extend(scales[first:first+rows].tobytes())
        binary.extend(zeros[first:first+rows].tobytes())
        binary.extend(codes[first:first+rows].tobytes())
    return bytes(binary), expected


@pytest.mark.parametrize('rows', [16, 32, 64])
@pytest.mark.parametrize('mutation', ['none', 'wrong-row-offset', 'live-metadata-alias'])
def test_actual_cpp_lookup_tail_matches_independent_bit_oracle(tmp_path, rows, mutation):
    compiler = os.environ.get('OSCAR_TEST_CLANGXX') or shutil.which('clang++')
    if not compiler:
        pytest.skip('host C++ compiler unavailable; actual-source test not run')
    source, binary = tmp_path/'lookup.cpp', tmp_path/'lookup'
    source.write_text(executable_source(rows, mutation))
    compiled = subprocess.run([compiler, '-std=c++17', '-O1', '-ffp-contract=off',
                               str(source), '-o', str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    payload, expected = fixture(rows)
    result = subprocess.run([str(binary)], input=payload, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    actual = np.frombuffer(result.stdout, dtype=np.uint16).reshape(expected.shape)
    if mutation == 'none':
        np.testing.assert_array_equal(actual, expected)
    else:
        assert np.count_nonzero(actual != expected) > 0, 'negative control failed to detect a broken lookup'
