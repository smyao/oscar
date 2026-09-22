// Archive #12/#27/#44/#91: every device status must be checked without a D2H sync.
// D.4 prepare/host: fuse four eq/all/assert chains into one device launch;
// O(status elements), fixed 12KiB UB/core, no KV read/write or history recovery.
// The latency target is below the 0.6-1.1ms decode phase; hardware timing pending.
// CANN9.1 kernel_operator_sys_var_intf.h:54 declares AscendC::Trap().
#include "oscar_common.h"
#ifdef ASCENDC_CPU_DEBUG
#include <cstdio>
#endif
using namespace oscar_ascend_device;
extern "C" __global__ __aicore__ void oscar_status_guard_kernel(GM_ADDR a,GM_ADDR b,
    GM_ADDR c,GM_ADDR d,int64_t na,int64_t nb,int64_t nc,int64_t nd) {
    TPipe pipe;TBuf<TPosition::VECCALC> integersBuf,floatsBuf,reduceBuf,scalarBuf;
    pipe.InitBuffer(integersBuf,4096);pipe.InitBuffer(floatsBuf,4096);
    pipe.InitBuffer(reduceBuf,4096);pipe.InitBuffer(scalarBuf,32);
    auto integers=integersBuf.Get<int32_t>();auto floats=floatsBuf.Get<float>();
    auto reduce=reduceBuf.Get<float>();auto scalar=scalarBuf.Get<float>();
    GM_ADDR pointers[4]={a,b,c,d};int64_t counts[4]={na,nb,nc,nd};
    for(int32_t segment=0;segment<4;++segment) {
        GlobalTensor<int32_t> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pointers[segment]));
        for(int64_t start=GetBlockIdx()*1024;start<counts[segment];start+=GetBlockNum()*1024) {
            const int32_t count=counts[segment]-start<1024 ? counts[segment]-start : 1024;
            DataCopyExtParams copy{1,static_cast<uint32_t>(count*4),0,0,0};
            DataCopyPadExtParams<int32_t> padding{false,0,0,0};
            DataCopyPad(integers,source[start],copy,padding);Fence<HardEvent::MTE2_V>();
            Cast(floats,integers,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
            Abs(floats,floats,count);PipeBarrier<PIPE_V>();
            ReduceMax(scalar,floats,reduce,count);Fence<HardEvent::V_S>();
            if(scalar.GetValue(0)!=0.0F) {
#ifdef ASCENDC_CPU_DEBUG
                // CPU debugger diagnostics do not add a print-workspace
                // allocation to production NPU graph launches.
                for(int32_t i=0;i<count;++i) if(integers.GetValue(i)!=0) {
                    std::fprintf(stderr,"OSCAR_GUARD segment=%d index=%ld status=%d\n",
                        segment,static_cast<long>(start+i),integers.GetValue(i));
                    break;
                }
#endif
                Trap();
            }
            Fence<HardEvent::V_MTE2>();
        }
    }
    // Declare and perform an idempotent mutation so graph compilers preserve
    // the validation side effect even though the operator returns no tensors.
    if(GetBlockIdx()==0 && na>0) {
        GlobalTensor<int32_t> output;output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a));
        integers.SetValue(0,0);Fence<HardEvent::S_MTE3>();
        DataCopyExtParams copy{1,4,0,0,0};DataCopyPad(output,integers,copy);Fence<HardEvent::MTE3_S>();
    }
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void oscar_status_guard_launch(void* stream,void* a,void* b,void* c,void* d,
    int64_t na,int64_t nb,int64_t nc,int64_t nd) {
    oscar_status_guard_kernel<<<4,nullptr,stream>>>(static_cast<uint8_t*>(a),static_cast<uint8_t*>(b),
        static_cast<uint8_t*>(c),static_cast<uint8_t*>(d),na,nb,nc,nd);
}
#endif
