"""Execute the actual RotateOnly producer with host DMA/rotation stubs.

This checks publication, pre-round checksum/status and padding, not CANN
rotation arithmetic, asynchronous fences, NPU completion or graph execution.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]

STUB = r'''
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <cstdio>
#include <stdexcept>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
constexpr int kRowTile=8,kMaxDim=256;
struct bfloat16_t{uint16_t bits;};
uint16_t round_bf16(float value){uint32_t b;std::memcpy(&b,&value,4);
 if((b&0x7fffffffU)>0x7f800000U)return uint16_t((b>>16)|0x40);
 return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
template<class T>struct Tensor{T*p=nullptr;size_t size=0;
 Tensor operator[](size_t i)const{if(i>size)throw std::runtime_error("offset");return{p+i,size-i};}
 T GetValue(size_t i)const{if(i>=size)throw std::runtime_error("read");return p[i];}
 void SetValue(size_t i,T v){if(i>=size)throw std::runtime_error("write");p[i]=v;}
 void SetGlobalBuffer(T*v){p=v;size=1000000;}
};
template<class T>using LocalTensor=Tensor<T>;
template<class T>using GlobalTensor=Tensor<T>;
enum class TPosition{VECCALC};
template<TPosition>struct TBuf{std::vector<uint32_t> data;
 template<class T>Tensor<T> Get(size_t=0){return{reinterpret_cast<T*>(data.data()),data.size()*4/sizeof(T)};}};
struct TPipe{template<class B>void InitBuffer(B&b,size_t bytes){b.data.resize((bytes+3)/4);}};
enum class HardEvent{MTE2_S,MTE2_V,S_MTE2,S_V,V_S,V_MTE3,S_MTE3,MTE3_V,MTE3_S};
enum PIPE{PIPE_V};template<HardEvent>void Fence(){}template<PIPE>void PipeBarrier(){}
enum class RoundMode{CAST_RINT};
struct DataCopyExtParams{int a;uint32_t bytes;int c,d,e;};
template<class T>struct DataCopyPadExtParams{bool a;int b,c,d;};
template<class T>void Duplicate(Tensor<T>x,T v,size_t n){for(size_t i=0;i<n;++i)x.SetValue(i,v);}
template<class T>void DataCopy(Tensor<T>a,Tensor<T>b,size_t n){for(size_t i=0;i<n;++i)a.SetValue(i,b.GetValue(i));}
template<class T>void DataCopyPad(Tensor<T>a,Tensor<T>b,DataCopyExtParams p){DataCopy(a,b,p.bytes/sizeof(T));}
template<class T>void DataCopyPad(Tensor<T>a,Tensor<T>b,DataCopyExtParams p,DataCopyPadExtParams<T>){DataCopy(a,b,p.bytes/sizeof(T));}
void Cast(Tensor<bfloat16_t>d,Tensor<float>s,RoundMode,size_t n){for(size_t i=0;i<n;++i)d.SetValue(i,{round_bf16(s.GetValue(i))});}
void ReduceSum(Tensor<float>d,Tensor<float>s,Tensor<float>,size_t n){float r=0;for(size_t i=0;i<n;++i)r+=s.GetValue(i);d.SetValue(0,r);}
bool Finite(float x){return std::isfinite(x);}
int block=0,blocks=3;int GetBlockIdx(){return block;}int GetBlockNum(){return blocks;}
std::vector<float> prepared;
template<class T>struct RotationEngine{int dim=0;int64_t off=0;std::vector<float> result;bool active[8];
 void Init(TPipe*,int d,bool){dim=d;result.resize(8*d);}
 void Load(GlobalTensor<T>,int64_t o,int){off=o;for(auto&a:active)a=true;}
 void ZeroInputRow(int row){active[row]=false;}
 void Transform(GlobalTensor<float>,int rows){for(int r=0;r<rows;++r)for(int j=0;j<dim;++j)result[r*dim+j]=active[r]?prepared[off+r*dim+j]:0;}
 Tensor<float> Result(){return{result.data(),result.size()};}
};
'''

MAIN = r'''
int main(){
 for(int d:{64,128,256})for(int heads:{1,6})for(bool slotsEnabled:{false,true})for(int replay=0;replay<2;++replay){
  const int tokens=19,rows=tokens*heads;
  prepared.resize(rows*d);std::vector<int64_t>slots(tokens);
  for(int t=0;t<tokens;++t)slots[t]=(t%3==0)?-1:t;
  for(int r=0;r<rows;++r)for(int j=0;j<d;++j){float v=float((r*29+j*17)%101-50)/19;
   if(replay)v=-v;if(r%11==0)v=std::numeric_limits<float>::max();
   if(r%13==0 && j==3)v=std::numeric_limits<float>::quiet_NaN();
   if(r%17==0 && j==4)v=-std::numeric_limits<float>::infinity();
   if(r%7==0 && j==0)v=-0.0F;prepared[r*d+j]=v;}
  std::vector<bfloat16_t>input(rows*d),out(rows*(d+16)+32,{0x5a5a});
  std::vector<float>old(rows*d,-99),matrix(d*d);
  std::vector<int32_t>oldstatus(rows,-88),status(rows,-88);
  for(block=0;block<blocks;++block){
   RotateOnly<bfloat16_t> a;RotateOnly<bfloat16_t,true>b;
   a.Run((GM_ADDR)input.data(),(GM_ADDR)matrix.data(),(GM_ADDR)old.data(),(GM_ADDR)oldstatus.data(),rows,d,true,slotsEnabled?(GM_ADDR)slots.data():nullptr,heads);
   b.Run((GM_ADDR)input.data(),(GM_ADDR)matrix.data(),(GM_ADDR)out.data(),(GM_ADDR)status.data(),rows,d,true,slotsEnabled?(GM_ADDR)slots.data():nullptr,heads);}
  for(int r=0;r<rows;++r){if(oldstatus[r]!=status[r])return 2;
   for(int j=0;j<d;++j)if(out[r*(d+16)+j].bits!=round_bf16(old[r*d+j]))return 3;
   float checksum;std::memcpy(&checksum,&out[r*(d+16)+d],4);
   float expected=0;for(int j=0;j<d;++j)expected+=old[r*d+j];
   if(std::isnan(expected)?!std::isnan(checksum):std::memcmp(&expected,&checksum,4)!=0)return 4;
   if((Finite(checksum)?0:2)!=oldstatus[r])return 5;
   for(int j=d+2;j<d+16;++j)if(out[r*(d+16)+j].bits)return 6;}
  for(int i=rows*(d+16);i<int(out.size());++i)if(out[i].bits!=0x5a5a)return 7;
 }
 std::puts("producer publication, checksum, padding, tail, replay data: passed");return 0;
}
'''


def producer_source():
    source = (ROOT / 'csrc/kernels/rotate_vector_init.cpp').read_text()
    start = source.index('template <typename T, bool PackedBf16 = false>')
    end = source.index('template <typename T>\nclass RotateStore', start)
    return source[start:end]


@pytest.mark.parametrize('missing_checksum', [False, True])
def test_actual_producer_preserves_rounding_and_pre_round_error(tmp_path, missing_checksum):
    compiler = os.environ.get('OSCAR_TEST_CLANGXX') or shutil.which('clang++')
    if not compiler:
        pytest.skip('host C++ compiler unavailable; no CANN execution implied')
    source = producer_source()
    if missing_checksum:
        statement = 'packedBuf_.template Get<float>().SetValue((row*(dim+16)+dim)/2,sum);'
        assert statement in source
        source = source.replace(statement, '')
    cpp = tmp_path / 'producer.cpp'
    binary = tmp_path / 'producer'
    cpp.write_text(STUB + source + MAIN)
    result = subprocess.run([compiler, '-std=c++17', '-ffp-contract=off', str(cpp), '-o', str(binary)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == (4 if missing_checksum else 0), result.stdout + result.stderr
