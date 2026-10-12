"""Archive #13-22/#126/#144-145/#148/#151/#154-156; startup D.4 four questions.

Source0 reconstruction only; no full-history restore. Batch two OLD per-lane
16-row sets into one32-row UB. Exact FP32 math/softmax capacity/status domains
survive; helper-call reduction is not a measured full-operator speedup.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c4_unpack32.generate import helper,PINS,sha
from csrc.experiments.decode_phase_profile.generate import function,masked
SOURCE='csrc/kernels/attention_cv_window_range_cluster4.cpp'


def generate(candidate):
    helper();original=(ROOT/SOURCE).read_text();text=original;edits=[]
    def replace(old,new,count=1,label=''):
        nonlocal text
        if text.count(old)!=count:raise ValueError((label,text.count(old),count))
        text=text.replace(old,new);edits.append((old,new,count,label))
    def method(name,change):
        a,b=function(text,name);old=text[a:b];new=change(old)
        if old==new:raise ValueError('no change '+name)
        replace(old,new,label=name)
    if candidate:
        replace('constexpr int32_t kKvSubtiles=kKvRows/(2*kHalfKv);',
            'constexpr int32_t kKvSubtiles=kKvRows/(2*kHalfKv);\nconstexpr int32_t kHistoryHalfKv=32;\nconstexpr int32_t kHistorySubtiles=kKvRows/(2*kHistoryHalfKv);',label='history geometry only')
        replace('static constexpr int32_t kElements=kHalfKv*D;',
            'static constexpr int32_t kElements=kHalfKv*D;\n  static constexpr int32_t kHistoryElements=kHistoryHalfKv*D;',label='history capacity')
        replace('pipe.InitBuffer(packedBuf,kHalfKv*kPackedStride);','pipe.InitBuffer(packedBuf,kHistoryHalfKv*kPackedStride);')
        replace('pipe.InitBuffer(planeBuf,oscar_ascend_fast::Geometry<D>::planeBytes);','pipe.InitBuffer(planeBuf,32);')
        replace('pipe.InitBuffer(naturalBuf,kElements*2);','pipe.InitBuffer(naturalBuf,kHistoryHalfKv*(D+16)*2);')
        replace('pipe.InitBuffer(wordBuf,kWords*2);pipe.InitBuffer(maskBuf,kWords*2);','pipe.InitBuffer(wordBuf,32);pipe.InitBuffer(maskBuf,D/8*2);')
        replace('pipe.InitBuffer(wordIndexBuf,kWords*4);pipe.InitBuffer(laneIndexBuf,kElements*4);','pipe.InitBuffer(wordIndexBuf,32);pipe.InitBuffer(laneIndexBuf,32);')
        replace('pipe.InitBuffer(metadataIndexBuf,128);pipe.InitBuffer(metadataHalfBuf,64);','pipe.InitBuffer(metadataIndexBuf,2*kHistoryHalfKv*4);pipe.InitBuffer(metadataHalfBuf,2*kHistoryHalfKv*2);')
        replace('pipe.InitBuffer(dequantBuf,kElements*4);','pipe.InitBuffer(dequantBuf,kHistoryElements*4);')
        replace('pipe.InitBuffer(scratchBuf,512);','pipe.InitBuffer(scratchBuf,kHistoryHalfKv*8*4);')
        replace('oscar_ascend_fast::InitIndices<D>','oscar_ascend_unpack32::InitIndices<D,kHistoryHalfKv>')
        replace('oscar_ascend_fast::Unpack<D>','oscar_ascend_unpack32::Unpack<D,kHistoryHalfKv>')
        def packed(body):
            body=body.replace('liveKvRows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-start-lane*kHalfKv)));',
                'const int64_t first=start+lane*kHalfKv;\n    liveKvRows=static_cast<int32_t>(Max64(0,Min64(kHalfKv,kvend-first))+\n        Max64(0,Min64(kHalfKv,kvend-first-2*kHalfKv)));')
            body=body.replace('kHalfKv*kPackedStride/2','kHistoryHalfKv*kPackedStride/2')
            body=body.replace('const int64_t candidate=start+lane*kHalfKv+j;',
                'const int64_t candidate=start+(j/kHalfKv)*(2*kHalfKv)+lane*kHalfKv+j%kHalfKv;')
            body=body.replace('Min64(liveKvRows-j,128-candidate%128)',
                'Min64(Min64(liveKvRows-j,kHalfKv-j%kHalfKv),128-candidate%128)')
            return body
        method('LoadPacked',packed)
        history_publish='''  __aicore__ void PublishHistoryKv(bool value,int32_t subtile) {
    auto src=dequantBuf.Get<float>();
    const int64_t slot=subtile*(2*kHistoryHalfKv)+lane*kHalfKv;
    // Same lane owns rows0..15 and32..47 (or16..31 and48..63).
    // Restore natural GM order without moving any error into the other AIV.
    const uint16_t blocks=static_cast<uint16_t>(kHalfKv*D/8);
    DataCopyParams copy{2,blocks,0,blocks};
    Fence<HardEvent::V_MTE3>();
    DataCopy(work[(value?vOffset:kOffset)+slot*D],src,copy);
    Fence<HardEvent::MTE3_V>();
  }
'''
        replace('  __aicore__ void PublishKv(bool value,int32_t subtile) {',history_publish+'  __aicore__ void PublishKv(bool value,int32_t subtile) {',label='lane-preserving history publication')
        old='''      for(int32_t subtile=0;subtile<kKvSubtiles;++subtile) {
        const int64_t subStart=start+subtile*(2*kHalfKv);
        if(kind==0) {LoadPacked(subStart);Unpack(false);}
        else LoadPrecise(subStart,false);
        PublishKv(false,subtile);
        if(kind==0)Unpack(true);else LoadPrecise(subStart,true);
        PublishKv(true,subtile);
      }'''
        new='''      if(kind==0) {
        for(int32_t subtile=0;subtile<kHistorySubtiles;++subtile) {
          const int64_t subStart=start+subtile*(2*kHistoryHalfKv);
          LoadPacked(subStart);Unpack(false);PublishHistoryKv(false,subtile);
          Unpack(true);PublishHistoryKv(true,subtile);
        }
      } else {
        for(int32_t subtile=0;subtile<kKvSubtiles;++subtile) {
          const int64_t subStart=start+subtile*(2*kHalfKv);
          LoadPrecise(subStart,false);PublishKv(false,subtile);
          LoadPrecise(subStart,true);PublishKv(true,subtile);
        }
      }'''
        replace(old,new,label='C1 history only')
        def cluster(body):
            body=body.replace('subtile<kKvSubtiles','subtile<kHistorySubtiles')
            body=body.replace('subtile*(2*kHalfKv)','subtile*(2*kHistoryHalfKv)')
            return body.replace('PublishKv(', 'PublishHistoryKv(')
        method('VectorCluster',cluster)
        # Capacity views remain exactly the original size. Native precedent:
        # moe_custom_sort_one_core.h:82 uses TBuf.Get<float>(element_count).
        def softmax(body):
            return body.replace('auto softWork=dequantBuf.Get<float>();','auto softWork=dequantBuf.Get<float>(kElements);')
        method('Softmax',softmax)
        replace('scratchBuf.Get<float>();','scratchBuf.Get<float>(128);',6,
                'retain every non-unpack scratch view at original512B')
        replace('#include "attention_striped_unpack.h"','#include "attention_striped_unpack.h"\n#include "unpack32.h"',label='research helper')
    name='candidate' if candidate else 'baseline'
    replace('attention_cv_window_range_cluster4_launch','c4_unpack32_'+name+'_launch')
    replace('oscar_attention_window_range_cluster4_kernel','oscar_c4_unpack32_'+name+'_kernel')
    for header,count in (('attention_fast_unpack.h',2),('attention_striped_unpack.h',1),('attention_range_mask.h',1),('oscar_common.h',1)):
        replace('#include "'+header+'"','#include "../../kernels/'+header+'"',count)
    replace('"../include/','"../../include/',2)
    restored=text
    for old,new,count,label in reversed(edits):
        if restored.count(new)!=count:raise ValueError('inverse edit drift '+label)
        restored=restored.replace(new,old)
    if restored!=original:raise ValueError('not reversible')
    unchanged=('Matmul','LoadQueries','LoadWindowRuns','LoadPrecise','SpillState','RestoreState','CubeTask','CubeCluster','PublishRows','PublishStatus','TaskValid')
    for name2 in unchanged:
        body=text[slice(*function(text,name2))].replace('scratchBuf.Get<float>(128)','scratchBuf.Get<float>()')
        if body!=original[slice(*function(original,name2))]:raise ValueError('unrelated math changed '+name2)
    for token in ('CrossCoreSetFlag<','CrossCoreWaitFlag(','SetHF32Mode('):
        if masked(text).count(token)!=masked(original).count(token):raise ValueError('protocol changed '+token)
    return '// RESEARCH C4 UNPACK32 ONLY; archive #13-22/#126/#144-145/#148/#151/#154-156 and D.4.\n'+text,{
        'arm':name,'symbol':'oscar_c4_unpack32_'+name+'_kernel','source_sha256':PINS[SOURCE],
        'unchanged_math_methods':unchanged,'original_bytes_restored':True,'GM_workspace_unchanged':True,
        'source1_current_PV_rows':16,'history_half_rows':32 if candidate else 16,
        'AIV_explicit_UB_bytes':157760 if candidate else 157632,'softmax_temp_bytes':16384,
        'status_lane_mapping':'old alternating16-row ownership preserved','execution':'not_run'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();manifest={}
    for candidate in (False,True):
        text,entry=generate(candidate);path=HERE/('c4_'+entry['arm']+'.cpp');entry['sha256']=sha(text);manifest[entry['arm']]=entry
        if a.check:
            if path.read_text()!=text:raise ValueError('stale C4 clone')
        else:path.write_text(text)
    encoded=json.dumps(manifest,indent=2)+'\n';path=HERE/'c4_manifest.json'
    if a.check:
        if path.read_text()!=encoded:raise ValueError('stale manifest')
    else:path.write_text(encoded)
    print(json.dumps({'status':'checked' if a.check else 'generated','production_changed':False,'execution':'not_run'}))
