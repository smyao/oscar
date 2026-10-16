"""Archive G4/#126/#145/#148/#154/#157; full D.4 reread for BF16 branch.

Derive independent C4/C16 readers from main7047ce36, preserving task/error/
flag ownership. Actual BF16 operand storage is distinct from FP32 score/state.
The old FP32 kernels and historical qualification remain byte-for-byte intact.
"""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c16_anchor_skip_20261015.generate import method
SOURCES={
    4:('attention_cv_member_pipeline_cluster4.cpp','0795a07b1db0ba3d12e7ae762929b752d3bfb6d7475a925cf9e25e62ddeeb4bc'),
    16:('attention_cv_l1_reuse_cluster16.cpp','d6b8ee9040678e57a21bc72215b7cdc1b9cf67b2b395c57a574d4377d27621b3'),
}
HEADER='''// SPDX-License-Identifier: Apache-2.0
// GENERATED BF16 LONG READER from main7047ce36. Archive G4/#126/#145/#148/#154/#157.
// Complete startup D.4 reread; four questions:
// 1. Bounded fused history/precise QK/PV and final value rotation operands.
// 2. Old whole-history dequant6499.8-6655.1ms/store209-215ms are forbidden
//    historical structures, not current measured phase percentages.
// 3. INT2 bytes/FP16 metadata and FP32 unpack/error/math remain original;
//    Q/K/V and each online-exp P round once to BF16. Normalized history
//    and immutable natural Rv also use BF16 operands with FP32 Cube output.
//    FP32 score/max/sum/alpha/acc/LSE/merge stay FP32. No history tensor.
//    Each old Q float slot is explicitly Q-BF16 first half and independent
//    P-BF16 second half; P NEVER aliases FP32 score across AIV lanes.
//    BF16 ND/ND2NZ uses C0=16; no float-NZ8 reinterpretation. Cast reuses
//    dead natural UB under original/local MTE3 ownership fences.
// 4. GM upper bounds unchanged: C4=2494464,C16=7487488B/core; UB157760B.
//    Source counts are not speed. New explicit BF16 oracle/CPU/CANN/device/
//    graph/model gates replace only the old FP32-only arithmetic comparison.
// Historical mother comments below describe its original FP32 arithmetic.
'''

NORMALIZED='''  __aicore__ void PublishNormalizedQueries() {
    auto acc=accBuf.Get<float>();auto rounded=naturalBuf.Get<bfloat16_t>();
    // #126/#145: natural holds at most32 rows of BF16; never cast64 rows
    // into that buffer. Finish each MTE3 read before reusing it on Vector.
    Fence<HardEvent::MTE3_V>();
    for(int32_t row=0;row<kLaneRows;row+=kHistoryHalfKv) {
      const int32_t count=static_cast<int32_t>(Min64(kHistoryHalfKv,kLaneRows-row));
      Cast(rounded,acc[row*D],RoundMode::CAST_RINT,count*D);
      Fence<HardEvent::V_MTE3>();
      DataCopy(operands[2*qOffset+(lane*kLaneRows+row)*D],rounded,count*D);
      Fence<HardEvent::MTE3_V>();
    }
  }
  __aicore__ int64_t ProbabilityOffset(int64_t queryBase) const {
    // Offsets are32-bit address words; operands[] indexes genuine16-bit
    // values. Q takes first half of its old float slot; P takes second half.
    return queryBase+kQueryRows*D/2;
  }
'''


def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def generate(cluster):
    name,pin=SOURCES[cluster];path=ROOT/'csrc/kernels'/name;original=path.read_text()
    if sha(original)!=pin:raise ValueError('fixed main FP32 mother changed '+name)
    text=original;edits=[]
    def edit(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError(('strict source anchor',old[:90],text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count))
    def body(name,transform):
        old=text[slice(*method(text,name))];edit(old,transform(old))
    edit('using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;',
         'using Matrix=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,false>;\nusing ResultMatrix=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;')
    edit('CubeFormat::'+('NZ' if cluster==16 else 'ND')+',float,true>;',
         'CubeFormat::'+('NZ' if cluster==16 else 'ND')+',bfloat16_t,true>;')
    edit('GetMatmulApiTiling<Matrix,TransposedMatrix,Matrix,Matrix>',
         'GetMatmulApiTiling<Matrix,TransposedMatrix,ResultMatrix,ResultMatrix>')
    edit('matmul::MatmulImpl<Matrix,TransposedMatrix,Matrix,Matrix,kCubeConfig>',
         'matmul::MatmulImpl<Matrix,TransposedMatrix,ResultMatrix,ResultMatrix,kCubeConfig>')
    if cluster==16:
        edit('kCubeConfig.shareL1Size+256*1024<=512*1024','kCubeConfig.shareL1Size+128*1024<=512*1024')
        edit('single external B1 plus SDK A/reserve exceeds physical L1','single BF16 B1 plus SDK A/reserve exceeds physical L1')
        edit('pipe.InitBuffer(residentB,256*1024);','pipe.InitBuffer(residentB,128*1024);')
        edit('residentB.Get<float>()','residentB.Get<bfloat16_t>()',2)
        edit('if(rotation)DataCopy(local,rv,copy);else DataCopy(local,work[b],copy);',
             'if(rotation)DataCopy(local,rv,copy);else DataCopy(local,operands[2*b],copy);')
    edit('rv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotation));',
         'rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rotation));')
    set_work='work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)+core*floatsPerCore);'
    edit(set_work,set_work+'\n    operands.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace)+core*2*floatsPerCore);')
    edit('pOffset=scoreOffset;pvOffset=pOffset+kQueryRows*kKvRows;rotOffset=pvOffset;',
         'pOffset=kQueryRows*D/2;pvOffset=scoreOffset+kQueryRows*kKvRows;rotOffset=pvOffset;')
    edit(';pOffset=scoreOffset;','; // P lives in each Q slot second half, never in FP32 score.')
    edit('mm.SetTensorA(work[a],false);','mm.SetTensorA(operands[2*a],false);',1 if cluster==16 else 2)
    if cluster==4:
        edit('mm.SetTensorB(work[b],transposeB)','mm.SetTensorB(operands[2*b],transposeB)')
        edit('mm.SetTensorB(work[b],false)','mm.SetTensorB(operands[2*b],false)')
        edit('MatmulAtomic(pOffset,vOffset,accStateOffset+member*kQueryRows*D,',
             'MatmulAtomic(ProbabilityOffset(member*kQueryRows*D),vOffset,accStateOffset+member*kQueryRows*D,')
    else:
        edit('MatmulResidentB(pOffset,accStateOffset+member*kQueryRows*D,',
             'MatmulResidentB(ProbabilityOffset(member*kQueryRows*D),accStateOffset+member*kQueryRows*D,')
    # Source0 FP32 QR and source1/2 native BF16 Q retain old finite checks;
    # publishing now performs an actual RNE Cast into disjoint BF16 storage.
    edit('Fence<HardEvent::V_MTE3>();\n      DataCopy(work[qOffset+row*D],dst,D);Fence<HardEvent::MTE3_MTE2>();',
         'auto rounded=naturalBuf.Get<bfloat16_t>();\n      Cast(rounded,dst,RoundMode::CAST_RINT,D);Fence<HardEvent::V_MTE3>();\n      DataCopy(operands[2*qOffset+row*D],rounded,D);Fence<HardEvent::MTE3_MTE2>();')
    edit('Duplicate(zeros,0.0F,count*D);Fence<HardEvent::V_MTE3>();\n      DataCopy(work[qOffset+row*D],zeros,count*D);Fence<HardEvent::MTE3_V>();',
         'Duplicate(zeros,0.0F,count*D);PipeBarrier<PIPE_V>();\n      auto rounded=naturalBuf.Get<bfloat16_t>();\n      Cast(rounded,zeros,RoundMode::CAST_RINT,count*D);Fence<HardEvent::V_MTE3>();\n      DataCopy(operands[2*qOffset+row*D],rounded,count*D);Fence<HardEvent::MTE3_V>();')
    def history(old):
        new=old.replace('kHalfKv*D/8','kHalfKv*D/16')
        new=new.replace('    Fence<HardEvent::V_MTE3>();',
            '    auto rounded=naturalBuf.Get<bfloat16_t>();\n    Cast(rounded,src,RoundMode::CAST_RINT,kHistoryElements);\n    Fence<HardEvent::V_MTE3>();',1)
        return new.replace('DataCopy(work[(value?vOffset:kOffset)+slot*D],src,copy);',
            'DataCopy(operands[2*(value?vOffset:kOffset)+slot*D],rounded,copy);')
    body('PublishHistoryKv',history)
    def precise(old):
        new=old.replace('    Fence<HardEvent::V_MTE3>();',
            '    auto rounded=naturalBuf.Get<bfloat16_t>();\n    Cast(rounded,src,RoundMode::CAST_RINT,kElements);\n    Fence<HardEvent::V_MTE3>();',1)
        return new.replace('DataCopy(work[(value?vOffset:kOffset)+slot*D],src,kElements);',
            'DataCopy(operands[2*(value?vOffset:kOffset)+slot*D],rounded,kElements);')
    body('PublishKv',precise)
    edit('    DataCopy(work[pOffset+rowBase*kKvRows],scores,activeRows*kKvRows);',
         '    auto roundedP=naturalBuf.Get<bfloat16_t>();\n    Cast(roundedP,scores,RoundMode::CAST_RINT,activeRows*kKvRows);\n    Fence<HardEvent::V_MTE3>();\n    DataCopy(operands[2*ProbabilityOffset(qOffset)+rowBase*kKvRows],roundedP,activeRows*kKvRows);')
    edit('Fence<HardEvent::V_MTE3>();DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D);',
         'PublishNormalizedQueries(); // independent C1 final rotation input')
    edit('      Fence<HardEvent::V_MTE3>();\n      DataCopy(work[qOffset+lane*kLaneRows*D],acc,kLaneRows*D);',
         '      PublishNormalizedQueries(); // clustered final rotation input')
    edit('  __aicore__ void LoadQueries() {',NORMALIZED+'  __aicore__ void LoadQueries() {')
    edit('GlobalTensor<bfloat16_t> q,ck,cv,wk,wv;','GlobalTensor<bfloat16_t> q,ck,cv,wk,wv,rv,operands;')
    edit('GlobalTensor<float> qr,rv,out,outLse,work;','GlobalTensor<float> qr,out,outLse,work;')
    # Fixed existing allocations are intentionally retained as an upper bound.
    check='''  static_assert(D==256 && kKvRows==D,"BF16 Q/P half-slot layout requires D256/KV256");
  static_assert(kHistoryHalfKv*(D+16)*2>=kHalfRows*kKvRows*2,
      "dead natural UB must fit one BF16 P block");
  static constexpr int32_t kUbBytes=128+kHistoryHalfKv*kPackedStride+32+
      kHistoryHalfKv*(D+16)*2+32+D/8*2+32+32+2*kHistoryHalfKv*4+2*kHistoryHalfKv*2+
      kHistoryElements*4+kHalfRows*kKvRows*4+kLaneRows*D*4+D*4+D*2+
      (2*kLaneRows+2*kHalfRows)*4+kHistoryHalfKv*8*4+128;
  static_assert(kUbBytes==157760 && kUbBytes<=184*1024,"BF16 long-reader UB budget drift");
'''
    edit(' public:\n  __aicore__ void Init(',check+' public:\n  __aicore__ void Init(')
    old_launch='attention_cv_'+('l1_reuse_cluster16' if cluster==16 else 'member_pipeline_cluster4')+'_launch'
    old_kernel='oscar_attention_cv_'+('l1_reuse_cluster16' if cluster==16 else 'member_pipeline_cluster4')+'_kernel'
    edit(old_launch,f'attention_cv_bf16_cluster{cluster}_launch')
    edit(old_kernel,f'oscar_attention_cv_bf16_cluster{cluster}_kernel')
    old_header='oscar_c16_l1_reuse_launch.h' if cluster==16 else 'oscar_member_pipeline_launch.h'
    edit(f'"../include/{old_header}"','"../include/oscar_bf16_cluster_launch.h"')
    restored=text
    for old,new,count in reversed(edits):
        if restored.count(new)!=count:raise ValueError(('inverse ambiguity',new[:90],restored.count(new),count))
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('full original FP32 source not restored')
    flags=lambda s:re.findall(r'CrossCore(?:WaitFlag|SetFlag)[^;]*;',s)
    if flags(text)!=flags(original):raise ValueError('original crosscore protocol changed')
    unchanged={}
    for name in ('Process','LoadTask','CurrentTask','UseTask','IsClusterMember','TaskValid',
                 'LoadPacked','Unpack','LoadWindowRuns','LoadPrecise','SpillState','RestoreState',
                 'PublishRows','PublishEmpty','PublishStatus'):
        before=original[slice(*method(original,name))];after=text[slice(*method(text,name))]
        if before!=after:raise ValueError('unexpected task/metadata/error/state change '+name)
        unchanged[name]=sha(before)
    text=HEADER+text
    return text,{'cluster':cluster,'source':str(path.relative_to(ROOT)),'source_sha256':pin,
        'candidate_sha256':sha(text),'full_inverse_mother_bytes':True,'unchanged_methods':unchanged,
        'crosscore_flags_unchanged':True,'Q_K_V_P_Rv_operands':'BF16_RNE','Cube_output_accumulator':'FP32',
        'softmax_online_LSE_merge':'FP32','workspace_upper_bound_per_core':7487488 if cluster==16 else 2494464,
        'UB_per_AIV':157760,'external_B1_bytes':131072 if cluster==16 else 0,
        'P_layout':'independent second half of each original Q float slot, never aliasFP32score',
        'FP32_baseline_not_modified':True,'CANN':'not_run','CPU':'not_run','NPU':'not_run'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();manifest={}
    for cluster in SOURCES:
        body,manifest[str(cluster)]=generate(cluster);path=ROOT/f'csrc/kernels/attention_cv_bf16_cluster{cluster}.cpp'
        if a.check:
            if path.read_text()!=body:raise ValueError('generated BF16 cluster drift')
        else:path.write_text(body)
    path=ROOT/'reports/bf16_attention_20261016/cluster_source_manifest.json';body=json.dumps(manifest,indent=2)+'\n'
    if a.check:
        if path.read_text()!=body:raise ValueError('cluster source manifest drift')
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body)
    print(json.dumps({'status':'checked' if a.check else 'generated','FP32_sources_modified':False,'device_validation':'not_run'}))


if __name__=='__main__':main()
