"""Archive #111/#126/#145/#148/#164: explicit FP32-metadata BF16 readers.

D.4: retain bounded tiles and original cross-core ownership; no history-sized
restore. New 144-byte row metadata and dequant precision are a versioned codec.
"""
from pathlib import Path
import argparse

ROOT=Path(__file__).resolve().parents[1]


def generate(cluster):
    text=(ROOT/f'csrc/kernels/attention_cv_bf16_cluster{cluster}.cpp').read_text()
    def edit(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError(('aligned cluster anchor',old,text.count(old),count))
        text=text.replace(old,new)
    edit('#include "attention_striped_unpack32.h"','#include "attention_aligned_unpack.h"')
    text=text.replace('oscar_ascend_unpack32::','oscar_ascend_aligned::')
    text=text.replace(f'attention_cv_bf16_cluster{cluster}',f'attention_cv_aligned_cluster{cluster}')
    edit('"../include/oscar_bf16_cluster_launch.h"','"../include/oscar_aligned_cluster_launch.h"')
    edit('qr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queryRot));',
         'qr.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(queryRot));')
    edit('GlobalTensor<bfloat16_t> q,ck,cv,wk,wv,rv,operands;',
         'GlobalTensor<bfloat16_t> q,qr,ck,cv,wk,wv,rv,operands;')
    edit('GlobalTensor<float> qr,out,outLse,work;', 'GlobalTensor<float> out,outLse,work;')
    edit('if(kind==0) {DataCopy(dst,qr[off],D);Fence<HardEvent::MTE2_V>();}',
         'if(kind==0) {DataCopy(src,qr[off],D);Fence<HardEvent::MTE2_V>();Cast(dst,src,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();}')
    edit('static constexpr int32_t kSlotBytes=D/2+8;','static constexpr int32_t kSlotBytes=D/2+16;')
    edit('  float scale;\n};','  float scale;\n  int32_t dequantMode;\n};')
    edit('metadataHalfBuf,dequantBuf,scratchBuf);','metadataHalfBuf,dequantBuf,scratchBuf,g.dequantMode);')
    edit('int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale',
         'int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale, int32_t dequantMode')
    edit('pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale};',
         'pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale,dequantMode};')
    edit('int64_t speculative,int64_t splits,float scale,uint32_t cores)',
         'int64_t speculative,int64_t splits,float scale,int32_t dequantMode,uint32_t cores)')
    edit('pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale);',
         'pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale,dequantMode);')
    return ('// Versioned aligned codec: FP32 scale/normalized zero, 144B/D256.\n'
            '// dequantMode=0: FP32 prefill reconstruction; 1: BF16 decode steps.\n'
            '// Archive #111/#126/#145/#164; D.4 bounded tile protocol retained.\n'+text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    for cluster in (4,16):
        suffix='_control' if cluster==16 else ''
        path=ROOT/f'csrc/kernels/attention_cv_aligned_cluster{cluster}{suffix}.cpp';text=generate(cluster)
        if cluster==16:text=text.replace('attention_cv_aligned_cluster16','attention_cv_aligned_cluster16_control')
        if args.check:
            if path.read_text()!=text:raise ValueError('aligned cluster source drift')
        else:path.write_text(text)
    from tools.generate_c16_tuned import generate as tuned
    path=ROOT/'csrc/kernels/attention_cv_aligned_cluster16.cpp'
    text=tuned().replace('attention_cv_aligned_cluster16_tuned','attention_cv_aligned_cluster16')
    if args.check:
        if path.read_text()!=text:raise ValueError('targeted C16 source drift')
    else:path.write_text(text)


if __name__=='__main__':main()
