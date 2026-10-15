"""Archive #134/#140/#145/#148/#154: actual Softmax-only CPU stress harness.

Extract the exact two production-shaped Softmax/AddVisibleRange methods.
This isolates changed validation/scratch ordering; it never replaces the
complete C320 full-CV CPU/CAModel pair or claims hardware qualification.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c16_b1_followup_20261015.generate import generate_arm
from csrc.experiments.c16_anchor_skip_20261015.generate import method,sha

TEMPLATE='''// GENERATED actual Softmax CPU isolation. Archive #134/#140/#145/#148/#154.
// D.4: bounded FIA score validation, no history restoration. Original per-row
// ReduceSum and full V2/mask/alpha body are tested; helper is not full CV,
// hardware/graph or speed evidence. Zero production resource change.
#include "../../../kernels/oscar_common.h"
#include "../../../kernels/attention_range_mask.h"
#include "adv_api/activation/softmaxflashv2.h"
using namespace oscar_ascend_device;
namespace {
constexpr int32_t D=256,kKvRows=256,kHalfRows=32,kLaneRows=64,kElements=4096;
constexpr float kNegativeInfinity=-__builtin_inff(),kEmptyMax=-__FLT_MAX__;
constexpr SoftmaxConfig kCvSoftmaxConfig={false,0,0,SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
__aicore__ inline int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
__aicore__ inline int64_t Max64(int64_t a,int64_t b){return a>b?a:b;}
class Probe {
 public:
  __aicore__ void Run(GM_ADDR w,GM_ADDR ia,GM_ADDR is,GM_ADDR oa,GM_ADDR os,GM_ADDR oe,
      int64_t source,int64_t ctx,int64_t queries,int64_t group,int64_t start,int64_t end,
      int64_t sink,int64_t recent,int64_t rowbase,int64_t initialError) {
    g={group,1,sink,recent,1.0F};kind=source;context=ctx;qcount=queries;
    qbegin=requestBegin=0;kvend=end;lane=static_cast<int32_t>(rowbase/64);error=initialError;
    scoreOffset=pOffset=0;
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
    GlobalTensor<float> inAcc,inStats,outAcc,outStats;GlobalTensor<int32_t> outError;
    inAcc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ia));inStats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(is));
    outAcc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(oa));outStats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(os));
    outError.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(oe));
    pipe.InitBuffer(naturalBuf,17408);pipe.InitBuffer(dequantBuf,32768);
    pipe.InitBuffer(scoreBuf,32768);pipe.InitBuffer(accBuf,65536);
    pipe.InitBuffer(statsBuf,768);pipe.InitBuffer(scratchBuf,1024);pipe.InitBuffer(addressBuf,128);
    DataCopy(accBuf.Get<float>(),inAcc,64*D);DataCopy(statsBuf.Get<float>(),inStats,192);
    Fence<HardEvent::MTE2_V>();
    Softmax(start,static_cast<int32_t>(rowbase));
    Fence<HardEvent::V_MTE3>();DataCopy(outAcc,accBuf.Get<float>(),64*D);
    DataCopy(outStats,statsBuf.Get<float>(),192);Fence<HardEvent::MTE3_V>();
    auto result=addressBuf.Get<int32_t>();result.SetValue(0,error);
    Fence<HardEvent::S_MTE3>();DataCopyExtParams copy{1,4,0,0,0};DataCopyPad(outError,result,copy);
    Fence<HardEvent::MTE3_S>();
  }
 private:
@@VISIBLE@@
@@SOFTMAX@@
  TPipe pipe;
  struct Geometry {int64_t hq,hk,sink,recent;float scale;} g;
  TBuf<TPosition::VECCALC> naturalBuf,dequantBuf,scoreBuf,accBuf,statsBuf,scratchBuf,addressBuf;
  GlobalTensor<float> work;
  int64_t qcount,qbegin,requestBegin,context,kvend,kind,scoreOffset,pOffset;
  int32_t lane,error;
};
}
extern "C" __global__ __aicore__ void finite_probe_@@ARM@@(
    GM_ADDR w,GM_ADDR ia,GM_ADDR is,GM_ADDR oa,GM_ADDR os,GM_ADDR oe,
    int64_t source,int64_t ctx,int64_t queries,int64_t group,int64_t start,int64_t end,
    int64_t sink,int64_t recent,int64_t rowbase,int64_t initialError) {
  Probe op;op.Run(w,ia,is,oa,os,oe,source,ctx,queries,group,start,end,sink,recent,rowbase,initialError);
}
'''


def generate(arm):
    source,identity=generate_arm(arm)
    if (HERE/f'c16_{arm}.cpp').read_text()!=source:raise ValueError('combined source drift')
    visible=source[slice(*method(source,'AddVisibleRange'))];softmax=source[slice(*method(source,'Softmax'))]
    body=TEMPLATE.replace('@@VISIBLE@@',visible).replace('@@SOFTMAX@@',softmax).replace('@@ARM@@',arm)
    return body,{'arm':arm,'full_CV_source_sha256':identity['generated_sha256'],'Softmax_sha256':sha(softmax),
        'AddVisibleRange_sha256':sha(visible),'probe_sha256':sha(body),'exact_methods_extracted':True,
        'scope':'actual isolated Softmax CPU numerical/error/scratch qualification only; full C320 pair remains mandatory'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();folder=HERE/'finite_probe';folder.mkdir(exist_ok=True);manifest={}
    for arm in ('baseline','candidate'):
        text,manifest[arm]=generate(arm);path=folder/f'{arm}.cpp'
        if a.check:
            if path.read_text()!=text:raise ValueError('probe source drift')
        else:path.write_text(text)
    body=json.dumps(manifest,indent=2)+'\n';path=folder/'manifest.json'
    if a.check:
        if path.read_text()!=body:raise ValueError('probe manifest drift')
    else:path.write_text(body)
    print(json.dumps({'status':'checked' if a.check else 'generated','CPU':'not_run','full_CV_qualification':'separate'}))
