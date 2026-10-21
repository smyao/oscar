"""Archive #173/#171: paired BF16 publication and Cube term ownership.

These host checks exercise generated C++ term loops and workspace bounds,
not CANN execution, device atomic visibility or numerical qualification.
"""
import os
from pathlib import Path
import subprocess

import pytest
from tests.test_bf16_history_kernel_sources import body

ROOT = Path(__file__).resolve().parents[1]
KERNELS = tuple((ROOT / 'csrc/kernels').glob('attention_bf16_history_*.cpp'))

@pytest.mark.parametrize('path', KERNELS, ids=lambda p:p.stem)
@pytest.mark.parametrize('kind,transpose,expected', [(0, True,4),(1, True,1),(2, False,2)])
def test_actual_cube_loop_has_source_specific_terms(path, kind, transpose, expected, tmp_path):
    compiler=os.environ.get('OSCAR_TEST_CLANGXX')
    if not compiler:
        pytest.skip('set OSCAR_TEST_CLANGXX for host C++ execution')
    kernel=path.read_text()
    code=body(kernel,'void Matmul(')
    prefix=r'''
using int64_t=long long; using int32_t=int;using uint8_t=unsigned char;
extern "C" int printf(const char*,...);
struct Tensor {int64_t index=0; Tensor operator[](int64_t x)const{return{x};}};
struct MM {int64_t a=0,b=0; void SetOrgShape(int,int,int){} void SetSingleShape(int,int,int){}
void SetTensorA(Tensor t,bool){a=t.index;} void SetTensorB(Tensor t,bool){b=t.index;}
template<bool> void IterateAll(Tensor t,uint8_t mode=0){printf("%lld %lld %lld %d\n",a,b,t.index,mode);} void End(){}};
namespace oscar_bf16_history {template<class...T>void SelectMatmul(T&&...){}}
void SetHF32Mode(bool){}
struct Kernel {int kind=0,pipe=0;bool historyMatmulActive=true;int D=256,kKvRows=256;
int64_t residualOffset=1000000; Tensor operandWork,work,rv;MM historyMm,mm;
void Matmul(int64_t a,int64_t b,int64_t c,int32_t m,int32_t n,int32_t k,bool rotation=false,bool transposeB=true){
'''
    code=prefix+code+'}};\nint main(){Kernel x;x.kind='+str(kind)+';x.Matmul(11,22,33,24,256,256,false,'+str(transpose).lower()+');}\n'
    cpp=tmp_path/'term.cpp';exe=tmp_path/'term';cpp.write_text(code)
    built=subprocess.run([compiler,'-std=c++17',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    rows=[list(map(int,x.split())) for x in subprocess.check_output([str(exe)],text=True).splitlines()]
    assert len(rows)==expected
    assert rows[0]==[11,22,33,0]
    assert all(row[3]==1 for row in rows[1:])
    assert len({tuple(row[:2]) for row in rows})==expected
    if kind:
        assert all(row[1]==22 for row in rows) # exact V/K never reads stale low plane


def test_c4_accumulation_keeps_every_residual_term_atomic():
    source=(ROOT/'csrc/kernels/attention_bf16_history_member_pipeline_cluster4.cpp').read_text()
    code=body(source,'void MatmulAtomic(')
    assert 'ai<2' in code and 'bi<2' in code
    assert 'IterateAll<true>(work[c],static_cast<uint8_t>(1))' in code
    assert '(ai||bi)' not in code
    publication=body(source,'void PublishHistoryOperand(')
    assert 'source[kHistoryHalfKv*D]' in publication
    assert 'source[kHistoryHalfKv*D+kElements]' in publication


def test_small_level_tables_and_p_split_scratch_are_disjoint():
    # Actual table layout: high [0,8H), low [8H,16H), FP32 [16H,32H).
    for h in (16,32,64):
        regions=[set(range(0,8*h)),set(range(8*h,16*h)),set(range(16*h,32*h))]
        assert all(not regions[i]&regions[j] for i in range(3) for j in range(i))
        assert 4*h*4<=8192 # before codeFloat lifetime, index scratch holds level residual
    source=(ROOT/'csrc/kernels/attention_bf16_history_operands.h').read_text()
    assert 'constexpr int32_t chunk=2048' in source
    assert 'auto low=high[chunk]' in source
    assert 'auto reconstructed=scratch.Get<float>()[chunk]' in source
    assert 'Sub(reconstructed,source[begin],reconstructed,count)' in source
    assert 'Sub(source' not in source # preserve softmax denominator input


def test_real_pair_publisher_keeps_source_and_covers_chunk_tails(tmp_path):
    from test_bf16_query_producer import STUB
    compiler=os.environ.get('OSCAR_TEST_CLANGXX')
    if not compiler:
        pytest.skip('set OSCAR_TEST_CLANGXX for host C++ execution')
    stub=STUB.replace('CAST_RINT};','CAST_RINT,CAST_NONE};')
    stub+=r'''
void Cast(Tensor<float>d,Tensor<bfloat16_t>s,RoundMode,size_t n){for(size_t i=0;i<n;++i){
 uint32_t bits=uint32_t(s.GetValue(i).bits)<<16;float v;std::memcpy(&v,&bits,4);d.SetValue(i,v);}}
void Sub(Tensor<float>d,Tensor<float>a,Tensor<float>b,size_t n){for(size_t i=0;i<n;++i)d.SetValue(i,a.GetValue(i)-b.GetValue(i));}
constexpr int32_t kOperandScratchBytes=16384;
void Publish(GlobalTensor<bfloat16_t> destination,GlobalTensor<bfloat16_t> residualDestination,
 LocalTensor<float> source,TBuf<TPosition::VECCALC>& scratch,int32_t elements){
'''
    helper=(ROOT/'csrc/kernels/attention_bf16_history_operands.h').read_text()
    code=stub+body(helper,'void Publish(')+r'''}
int main(){for(int n:{256,2048,2304,4096,8192}){
 std::vector<float>source(n);std::vector<bfloat16_t>hi(n+16,{0x5a5a}),lo(n+16,{0x5a5a});
 for(int i=0;i<n;++i)source[i]=float(i%101-50)/31;
 auto saved=source;TPipe pipe;TBuf<TPosition::VECCALC>scratch;pipe.InitBuffer(scratch,16384);
 Publish({hi.data(),hi.size()},{lo.data(),lo.size()},{source.data(),source.size()},scratch,n);
 if(source!=saved)return 1;
 for(int i=0;i<n;++i){uint16_t expected=round_bf16(source[i]);
  if(hi[i].bits!=expected)return 2;uint32_t bits=uint32_t(expected)<<16;float recovered;
  std::memcpy(&recovered,&bits,4);if(lo[i].bits!=round_bf16(source[i]-recovered))return 3;}
 for(int i=n;i<n+16;++i)if(hi[i].bits!=0x5a5a||lo[i].bits!=0x5a5a)return 4;
}return 0;}
'''
    cpp=tmp_path/'publisher.cpp';exe=tmp_path/'publisher';cpp.write_text(code)
    result=subprocess.run([compiler,'-std=c++17','-ffp-contract=off',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert subprocess.run([str(exe)]).returncode==0
