"""Actual five LoadQueries bodies on host; memory/finite contract, not NPU."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from test_bf16_query_producer import STUB, producer_source
from test_bf16_reader_protocol import _function, _without_comments, FAMILIES

ROOT = Path(__file__).resolve().parents[1]


def model(filename):
    source = _without_comments((ROOT/'csrc/kernels'/filename).read_text())
    constants = source[source.index('constexpr int32_t kQueryRows='):
                       source.index('constexpr SoftmaxConfig')]
    return constants + r'''
struct Model {
 static constexpr int D=256;
 struct{int hq=12,hk=2;}g;
 int qcount=1,qbegin=3,kvhead=1,lane=0,kind=0,error=1,qOffset=32;
 TPipe pipe;TBuf<TPosition::VECCALC>qFloatBuf,qBf16Buf,scratchBuf,dequantBuf;
 GlobalTensor<bfloat16_t>q,qr;GlobalTensor<float>qrChecksum;
 std::vector<bfloat16_t> published;
 Model(){pipe.InitBuffer(qFloatBuf,D*4);pipe.InitBuffer(qBf16Buf,D*2);
  pipe.InitBuffer(scratchBuf,(D+16)*4);pipe.InitBuffer(dequantBuf,kHalfKv*D*2);
  published.resize(qOffset+kQueryRows*D+32,{0x5a5a});}
 void PublishBf16Operand(int64_t off,LocalTensor<bfloat16_t>src,int elements){
  DataCopy(Tensor<bfloat16_t>{published.data(),published.size()}[off],src,elements);}
''' + _function(source,'LoadQueries') + r'''
};
int check(){
 constexpr int d=256,tokens=32,heads=12,rows=tokens*heads;
 prepared.resize(rows*d);std::vector<bfloat16_t>input(rows*d),packed(rows*(d+16));
 std::vector<float> matrix(d*d);std::vector<int>status(rows);
 for(int repeat=0;repeat<2;++repeat){
  for(int r=0;r<rows;++r)for(int j=0;j<d;++j){
   float v=float((r*19+j*23)%109-54)/7;
   if(repeat && r==3*heads+6)v=std::numeric_limits<float>::max();
   if(repeat && r==4*heads+7 && j==9)v=std::numeric_limits<float>::quiet_NaN();
   prepared[r*d+j]=v;input[r*d+j]={round_bf16(v)};}
  for(block=0;block<blocks;++block){RotateOnly<bfloat16_t,true>p;
   p.Run((GM_ADDR)input.data(),(GM_ADDR)matrix.data(),(GM_ADDR)packed.data(),
     (GM_ADDR)status.data(),rows,d,true,nullptr,heads);}
  for(int count:{1,4})for(int kind:{0,1,2})for(int lane:{0,1}){
   Model m;m.qcount=count;m.kind=kind;m.lane=lane;
   m.q={input.data(),input.size()};m.qr={packed.data(),packed.size()};
   m.qrChecksum={reinterpret_cast<float*>(packed.data()),packed.size()/2};m.LoadQueries();
   int expectedError=1;
   for(int r=lane*kLaneRows;r<(lane+1)*kLaneRows;++r){
    bool live=r<count*6;int inputrow=(m.qbegin+r/6)*heads+6+r%6;
    if(live){float sum=0;
     if(kind==0)std::memcpy(&sum,&packed[inputrow*(d+16)+d],4);
     else for(int j=0;j<d;++j){uint32_t bits=uint32_t(input[inputrow*d+j].bits)<<16;
       float value;std::memcpy(&value,&bits,4);sum+=value;}
     if(!std::isfinite(sum))expectedError=2;}
    for(int j=0;j<d;++j){uint16_t expected=live?(kind==0?packed[inputrow*(d+16)+j].bits:input[inputrow*d+j].bits):0;
     if(m.published[m.qOffset+r*d+j].bits!=expected)return 12;}}
   if(m.error!=expectedError)return 13;
   for(int i=0;i<m.qOffset;++i)if(m.published[i].bits!=0x5a5a)return 14;
   for(size_t i=m.qOffset+kQueryRows*d;i<m.published.size();++i)if(m.published[i].bits!=0x5a5a)return 15;
  }
 }
 return 0;
}
'''


def test_all_reader_query_paths_consume_the_actual_producer_output(tmp_path):
    compiler = os.environ.get('OSCAR_TEST_CLANGXX') or shutil.which('clang++')
    if not compiler:
        pytest.skip('host C++ unavailable')
    stub = STUB.replace('MTE3_S};','MTE3_S,MTE3_MTE2};')
    stub = stub.replace('CAST_RINT};','CAST_RINT,CAST_NONE};')
    stub = stub.replace('T GetValue(size_t i)const',
        'template<class U>Tensor<U> ReinterpretCast(){return{reinterpret_cast<U*>(p),size*sizeof(T)/sizeof(U)};}\n T GetValue(size_t i)const')
    stub += r'''
int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
int64_t Max64(int64_t a,int64_t b){return a>b?a:b;}
namespace oscar_ascend{constexpr int kAttentionQueryRows=128,kAttentionKvRows=256;}
void Cast(Tensor<float>d,Tensor<bfloat16_t>s,RoundMode,size_t n){for(size_t i=0;i<n;++i){
 uint32_t b=uint32_t(s.GetValue(i).bits)<<16;float v;std::memcpy(&v,&b,4);d.SetValue(i,v);}}
'''
    cpp = stub + producer_source()
    unique = list(dict.fromkeys(f[1] for f in FAMILIES))
    for i, filename in enumerate(unique):
        cpp += f'namespace f{i}{{\n' + model(filename) + '}\n'
    cpp += 'int main(){int r;'
    for i in range(len(unique)):
        cpp += f'if((r=f{i}::check()))return r;'
    cpp += 'return 0;}'
    source,binary=tmp_path/'reader.cpp',tmp_path/'reader'
    source.write_text(cpp)
    result=subprocess.run([compiler,'-std=c++17','-ffp-contract=off',str(source),'-o',str(binary)],
                          text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(binary)],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
