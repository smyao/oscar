"""Isolated dual-segment / request-window reader, no production selection.

Archive #17-20/#34/#36/#37-49/#69/#126-129/#145/#154/#155; startup D.4.
Keep the delivered FP32 attention body and change only cache addressing.
The checked substitutions deliberately fail when the source structure drifts.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

HEADER = '''// EXPERIMENT: dual native-owned INT2 segments, request-stable precise windows.
// Archive #17-20/#34/#36/#37-49/#69/#126-129/#145/#154/#155; D.4 four questions:
// (1) source0 history and source1 precise attention; source2 is unchanged.
// (2) D.4 full restore 6499.8-6655.1ms must not return. Per-page BF16 copies
//     waste the INT2 capacity while request identity is not a batch row.
// (3) Each virtual128 read selects one of two native SoA segments. Window
//     map carries stable slot+epoch, and tags also verify logical position.
//     FP32 QK/PV/online state, error ordering and math are unchanged.
// (4) This reclaims storage; no attention speed or NPU accuracy is claimed.
//     Bounded map/tag reads add work. CPU, graph and lifecycle gates required.
'''

MAP_HELPER = '''  __aicore__ bool LoadWindowIdentity() {
    if(request<0 || request>=g.requests || request>=g.windowCapacity || g.windowCapacity<=0 ||
        g.positionBits<1 || g.positionBits>62) {error=1;return false;}
    auto identity=addressBuf.Get<int64_t>()[8];
    Fence<HardEvent::S_MTE2>();
    DataCopyExtParams copy{1,16,0,0,0};
    DataCopyPadExtParams<int64_t> pad{false,0,0,0};
    DataCopyPad(identity,windowMap[request*2],copy,pad);
    Fence<HardEvent::MTE2_S>();
    windowSlot=identity.GetValue(0);windowEpoch=identity.GetValue(1);
    const int64_t maxEpoch=INT64_MAX>>g.positionBits;
    if(windowSlot<0 || windowSlot>=g.windowCapacity ||
        windowEpoch<=0 || windowEpoch>maxEpoch) {error=1;return false;}
    Fence<HardEvent::S_MTE2>();
    return true;
  }
  __aicore__ int64_t ExpectedWindowTag(int64_t position) {
    if(position<0 || position>=(int64_t(1)<<g.positionBits))return -1;
    return (windowEpoch<<g.positionBits)|position;
  }
  __aicore__ bool ValidWindowTag(int64_t tag,int64_t position) {
    const int64_t expected=ExpectedWindowTag(position);
    return expected>=0 && tag==expected;
  }
'''


def replace(text, before, after, count=1):
    actual = text.count(before)
    if actual != count:
        raise ValueError(f"source drift ({actual}, expected {count}): {before!r}")
    return text.replace(before, after)


def transform(source):
    text = source
    text = replace(text, '  float scale;\n',
                   '  float scale;\n  int64_t windowCapacity,positionBits;\n')
    text = replace(text, 'GM_ADDR windowValue,GM_ADDR tags,GM_ADDR tasks,',
                   'GM_ADDR windowValue,GM_ADDR tags,GM_ADDR requestMap,GM_ADDR tasks,')
    text = replace(text, '    wt.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tags));',
                   '    wt.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tags));\n'
                   '    windowMap.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(requestMap));')
    text = replace(text, 'GlobalTensor<int64_t> wt,taskGm',
                   'GlobalTensor<int64_t> windowMap;\n  GlobalTensor<int64_t> wt,taskGm')
    text = replace(text, '  int64_t coreIndex,qbegin,',
                   '  int64_t windowSlot=-1,windowEpoch=0;\n  int64_t coreIndex,qbegin,')
    text = replace(text, 'const int64_t off=g.ssmOffset+block*g.pageStride+\n'
                   '          (inpage*g.hk+kvhead)*kSlotBytes;',
                   'const int64_t segmentTokens=g.blockTokens/2;\n'
                   '      const int64_t off=g.ssmOffset+block*g.pageStride+\n'
                   '          (inpage/segmentTokens)*g.blocks*g.pageStride+\n'
                   '          ((inpage%segmentTokens)*g.hk+kvhead)*kSlotBytes;')
    text = replace(text, '  __aicore__ void LoadWindowRuns(int64_t start,bool value) {',
                   MAP_HELPER + '  __aicore__ void LoadWindowRuns(int64_t start,bool value) {\n'
                   '    if(!LoadWindowIdentity())return;')
    # Only the sidecar row mapping changes. The physical table check remains
    # before tag/finite checks, preserving rejection of malformed live pages.
    text = replace(text, 'ring-inpage%ring', 'ring-position%ring')
    text = replace(text, 'windowRow=g.sink+inpage%ring;',
                   'windowRow=g.sink+position%ring;')
    text = replace(text, 'wt[block*g.tagStride+windowRow]',
                   'wt[windowSlot*g.tagStride+windowRow]')
    text = replace(text, 'tags.GetValue(begin)!=inpage+begin',
                   '!ValidWindowTag(tags.GetValue(begin),position+begin)')
    text = replace(text, 'tags.GetValue(end)==inpage+end',
                   'ValidWindowTag(tags.GetValue(end),position+end)')
    text = replace(text, 'tags.GetValue(row)!=inpage+row',
                   '!ValidWindowTag(tags.GetValue(row),position+row)')
    text = replace(text, 'block*g.windowStride+((windowRow+begin)*g.hk+kvhead)*D',
                   'windowSlot*g.windowStride+((windowRow+begin)*g.hk+kvhead)*D')
    # Old scalar source1 branch is unreachable after LoadWindowRuns' return;
    # remove it instead of leaving an obsolete page-snapshot reader in the ABI.
    start = text.index('    for(int32_t j=0;j<kHalfKv;++j) {', text.index('if(kind==1) {LoadWindowRuns'))
    end = text.index('  __aicore__ void PublishKv', start)
    # The qualified H32/member pipeline has a history publisher immediately
    # after LoadPrecise. Never delete it together with the obsolete scalar
    # source1 fallback; only the address protocol is being replaced here.
    history_end = text.find('  __aicore__ void PublishHistoryKv', start)
    if history_end >= 0:
        end = min(end, history_end)
    text = text[:start] + '    error=1; // source must be current or request-window here.\n  }\n' + text[end:]
    text = replace(text, 'GM_ADDR rotation, GM_ADDR raw, GM_ADDR table, GM_ADDR wk, GM_ADDR wv, GM_ADDR tags,',
                   'GM_ADDR rotation, GM_ADDR raw, GM_ADDR table, GM_ADDR wk, GM_ADDR wv, GM_ADDR tags, GM_ADDR requestMap,')
    text = replace(text, 'int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale\n',
                   'int64_t sink, int64_t recent, int64_t speculative, int64_t splits, float scale, \\\n'
                   '    int64_t windowCapacity, int64_t positionBits\n')
    text = replace(text, 'pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale};',
                   'pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale,windowCapacity,positionBits};')
    text = replace(text, 'op.Init(query,queryRot,key,value,rotation,raw,table,wk,wv,tags,',
                   'op.Init(query,queryRot,key,value,rotation,raw,table,wk,wv,tags,requestMap,')
    text = replace(text, 'void* rotation,void* raw,void* table,void* wk,void* wv,void* tags,void* tasks,',
                   'void* rotation,void* raw,void* table,void* wk,void* wv,void* tags,void* requestMap,void* tasks,')
    text = replace(text, 'int64_t speculative,int64_t splits,float scale,uint32_t cores)',
                   'int64_t speculative,int64_t splits,float scale,int64_t windowCapacity,int64_t positionBits,uint32_t cores)')
    text = replace(text, 'static_cast<uint8_t*>(tags),static_cast<uint8_t*>(tasks),',
                   'static_cast<uint8_t*>(tags),static_cast<uint8_t*>(requestMap),static_cast<uint8_t*>(tasks),')
    text = replace(text, 'pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale);',
                   'pageStride,windowStride,tagStride,sink,recent,speculative,splits,scale,windowCapacity,positionBits);')
    text = text.replace('attention_cv_decode_bundle', 'attention_request_window_decode')
    text = text.replace('attention_cv_member_pipeline_cluster4',
                        'attention_request_window_member_pipeline_cluster4')
    text = text.replace('attention_cv_window_range', 'attention_request_window')
    text = text.replace('oscar_attention_window_range', 'oscar_attention_request_window')
    # Original include search directory is supplied by the isolated CMake.
    text = text.replace('"../include/', '"../../include/')
    # CANN precompile is a separate CMake project and does not inherit a
    # later target_include_directories call. Resolve our own headers directly.
    for header in ('oscar_common.h', 'attention_fast_unpack.h',
                   'attention_striped_unpack.h', 'attention_striped_unpack_padded.h',
                   'attention_striped_unpack32.h',
                   'attention_range_mask.h'):
        text = text.replace('"'+header+'"', '"../../kernels/'+header+'"')
    return HEADER + text


def main():
    entries = []
    for suffix in ('decode_bundle', 'window_range', 'window_range_balanced',
                   'window_range_cluster4', 'window_range_cluster16',
                   'member_pipeline_cluster4'):
        path = ROOT / 'csrc/kernels' / f'attention_cv_{suffix}.cpp'
        source = path.read_text()
        out = HERE / f'{suffix}.cpp'
        out.write_text(transform(source))
        entries.append({'source': str(path.relative_to(ROOT)),
                        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                        'candidate': out.name,
                        'candidate_sha256': hashlib.sha256(out.read_bytes()).hexdigest()})
    (HERE / 'sources.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(f'generated {len(entries)} isolated request-window readers')


if __name__ == '__main__':
    main()
