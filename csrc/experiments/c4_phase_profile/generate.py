"""Archive #126/#132-135/#140/#145/#148/#154/#155; full startup D.4 read.

Same-source C4 normal/profile clones. Reversible counted instrumentation only;
no new completion fence, math, task filtering, production symbol or selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
PINS = {
    'csrc/kernels/attention_cv_window_range_cluster4.cpp': '14b28f9181220432a15bcc3f23679e539473b12e01d858bbb00ff3345522ba65',
    'csrc/kernels/attention_fast_unpack.h': 'f2366ee6fa895139524bb49e85f7d522a65d1f11c1d11e690f6d97099a1f2566',
    'csrc/kernels/attention_striped_unpack.h': '2f03998b9ac66474acaab70563b1608c4f3646369b7767a76ccd171eecbf3e21',
    'csrc/kernels/attention_range_mask.h': 'c1608937a57a0d74159f236b823a38682024bb3ea1893d75a9d1f3652788aa84',
    'csrc/kernels/oscar_common.h': '20574a83f919df5bd6450cd581d1dba5a5a6f169768e7e237ba62f51b7499964',
    'csrc/include/oscar_attention_launch.h': 'd8731742b2d928b3f8d9a9e330767df449ace11af6808917a3b41721dd2707df',
    'csrc/include/oscar_cv_schedule.h': '36c9bb6276e2ac066057d3d30be0f22f83bc65d6b3aef9497f6ceca946313db8',
    'csrc/experiments/decode_phase_profile/generate.py': '49298a943ce3c3c3a54abf9fdd69d07044b5d31cb5ec264ad622b319d23bdc44',
    'csrc/kernels/attention_tasks.cpp': '1e6c7c4e4d5f71d8c8679573bcc3c4d523361f93ae4e16b2fa3606acb7bc3952',
}
SOURCE = next(iter(PINS))


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def generate(profile):
    for path, pin in PINS.items():
        if sha((ROOT/path).read_bytes()) != pin:
            raise ValueError('source drift; re-audit before repinning: '+path)
    from csrc.experiments.decode_phase_profile.generate import ENUM, HELPERS, FIELDS, function, masked
    fields = [*FIELDS, 'state_spill', 'state_restore', 'clusters', 'member_kv_units',
              'state_bytes', 'c1_tasks', 'cluster_members', 'finite_reduce']
    enum = ENUM.replace('schema2: same [core,3,4,24]', 'schema3: C4-only [core,3,4,32]')
    enum = enum.replace('kComputeSpan=23,', 'kComputeSpan=23,\n  kStateSpill=24,kStateRestore=25,kClusters=26,\n'
                        '  kMemberKvUnits=27,kStateBytes=28,kC1Tasks=29,kClusterMembers=30,kFiniteReduce=31,')
    helpers = HELPERS.replace('oscar_ascend::kAttentionProfileFields', '32')
    helpers = helpers.replace('32==24', '32==32').replace('field<24', 'field<32').replace('source*24', 'source*32')
    original = (ROOT/SOURCE).read_text()
    text, edits = original, []

    def replace(old, new, count=1, label=''):
        nonlocal text
        if text.count(old) != count: raise ValueError((label, text.count(old), count))
        text = text.replace(old, new); edits.append((old,new,count,label))

    def method(name, change):
        a,b = function(text,name); before=text[a:b]; after=change(before)
        if before == after: raise ValueError('unchanged '+name)
        replace(before,after,label='method '+name)

    def scope(name, field, extra=''):
        method(name,lambda b:b[:1]+'\n    ProfileScope profileMethod(*this,'+field+');'+extra+b[1:])

    def waits(name, fields):
        flag='kVectorReady' if name.startswith('Cube') else 'kCubeReady'
        token='CrossCoreWaitFlag('+flag+');'
        def edit(body):
            if body.count(token)!=len(fields): raise ValueError((name,'wait count'))
            for field in fields: body=body.replace(token,'ProfileWait('+field+','+flag+');',1)
            return body
        method(name,edit)

    if profile:
        replace('template<int32_t D> class AttentionCv {',enum+'\ntemplate<int32_t D> class AttentionCv {',label='enum')
        replace('GM_ADDR workspace,GM_ADDR clusterStatsPtr,','GM_ADDR workspace,GM_ADDR clusterStatsPtr,GM_ADDR profile,',label='Init ABI')
        replace('    g=geometry;','    TraceMarker(kComputeSpan,false);\n    profileStart=GetSystemCycle();\n    g=geometry;',label='actor begin')
        replace('    coreIndex=core;', '''    coreIndex=core;
    engineIndex=0;
    if ASCEND_IS_AIV {engineIndex=lane+1;}
    profileGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(profile));
    for(int32_t source=0;source<4;++source)
      for(int32_t field=0;field<32;++field)profileCounters[source][field]=0;''',label='profile init')
        replace('  __aicore__ void Process() {',helpers+'  __aicore__ void Process() {',label='profile helpers')
        replace('    }\n  }\n private:', '''    }
    profileCounters[3][kComputeSpan]=static_cast<int64_t>(GetSystemCycle()-profileStart);
    TraceMarker(kComputeSpan,true);
    ProfileFlush();
  }
 private:''',label='actor end')
        replace('  Geometry g;', '''  Geometry g;
  GlobalTensor<int64_t> profileGm;
  int64_t profileCounters[4][32];uint64_t profileStart;int32_t engineIndex=0;''',label='profile storage')
        for name,field in (('LoadTask','kTaskMetadata'),('LoadQueries','kQueryStage'),
            ('LoadPacked','kPackedDma'),('Unpack','kUnpack'),('LoadPrecise','kPreciseLoad'),
            ('PublishKv','kKvPublish'),('Softmax','kSoftmaxP'),('PublishRows','kFinalWriteback')):
            scope(name,field)
        def finite_scope(body):
            token=('      ReduceSum(tmp,scores[r*kKvRows],tmp[16],kKvRows);Fence<HardEvent::V_S>();\n'
                   '      if(!Finite(tmp.GetValue(0)))error=2;')
            if body.count(token)!=1:raise ValueError('finite ReduceSum/fence/check anchor drift')
            return body.replace(token,'      { ProfileScope profileFinite(*this,kFiniteReduce);\n'+token+'\n      }')
        method('Softmax',finite_scope)
        for name,field in (('SpillState','kStateSpill'),('RestoreState','kStateRestore')):
            scope(name,field,'\n    ProfileAdd(kStateBytes,(kLaneRows*D+2*kLaneRows)*sizeof(float));')
        scope('Matmul','rotation?kAicRotation:(transposeB?kAicQk:kAicPv)')
        for name in ('CubeTask','VectorTask'):
            scope(name,'kTaskSpan','\n    ProfileAdd(kTasks,1);ProfileAdd(kC1Tasks,1);')
        for name in ('CubeCluster','VectorCluster'):
            scope(name,'kTaskSpan','\n    ProfileAdd(kTasks,1);ProfileAdd(kClusters,1);ProfileAdd(kClusterMembers,4);')
        waits('CubeTask',['kAicKvWait','kAicPWait','kAicAccWait','kRotationWait','kRotationWait'])
        waits('VectorTask',['kAivQkWait','kAivPvWait','kRotationWait'])
        waits('CubeCluster',['kAicKvWait','kAicPWait','kAicAccWait','kRotationWait','kRotationWait'])
        waits('VectorCluster',['kAivQkWait','kAivPvWait','kRotationWait'])
        for loop,end,members in (
            ('for(int64_t start=kvbegin;start<kvend;start+=kKvRows) {','kvend',1),
            ('for(int64_t start=first;start<last;start+=kKvRows) {','last',4)):
            replace(loop,loop+'\n      ProfileAdd(kKvUnits,1);\n'
                    '      ProfileAdd(kKvRowsObserved,Min64(kKvRows,'+end+'-start));\n'
                    f'      ProfileAdd(kMemberKvUnits,{members});',2,'tile work counters')
        for name in ('VectorTask','VectorCluster'):
            def pv_scope(body):
                begin=body.index('      auto result=dequantBuf.Get<float>();') if name=='VectorTask' else body.index('        auto result=dequantBuf.Get<float>();')
                finish=body.index('      CrossCoreSetFlag<2,PIPE_MTE2>(kVectorReady);',begin) if name=='VectorTask' else body.index('        memberErrors[member]=error;',begin)
                return body[:begin]+'      { ProfileScope profilePv(*this,kPvAcc);\n'+body[begin:finish]+'      }\n'+body[finish:]
            method(name,pv_scope)
        def finish_task(body):
            token='    Fence<HardEvent::V_S>();\n    for(int32_t r=0;r<kLaneRows;++r) {'
            if body.count(token)!=1: raise ValueError('task finalize anchor drift')
            return body.replace(token,'    ProfileScope profileFinalize(*this,kAivFinalize);\n'+token)
        method('VectorTask',finish_task)
        def finish_cluster(body):
            token='      error=memberErrors[member];RestoreState(member);\n      Fence<HardEvent::V_S>();'
            if body.count(token)!=1: raise ValueError('cluster finalize anchor drift')
            return body.replace(token,'      ProfileScope profileFinalize(*this,kAivFinalize);\n'+token)
        method('VectorCluster',finish_cluster)
        replace('    GM_ADDR clusterStats, \\\n','    GM_ADDR clusterStats, GM_ADDR profile, \\\n',label='kernel ABI')
        replace('status,workspace,clusterStats,g);','status,workspace,clusterStats,profile,g);',label='Init call')
        replace('void* workspace,void* clusterStats,','void* workspace,void* clusterStats,void* profile,',label='launch ABI')
        replace('      static_cast<uint8_t*>(clusterStats),','      static_cast<uint8_t*>(clusterStats),static_cast<uint8_t*>(profile),',label='launch arg')
    arm='profile' if profile else 'normal'
    replace('attention_cv_window_range_cluster4_launch','c4_'+arm+'_launch',label='isolated launch macro')
    replace('oscar_attention_window_range_cluster4_kernel','oscar_c4_'+arm+'_kernel',label='isolated kernel macro')
    for header,count in (('attention_fast_unpack.h',2),('attention_striped_unpack.h',1),
                         ('attention_range_mask.h',1),('oscar_common.h',1)):
        replace('#include "'+header+'"','#include "../../kernels/'+header+'"',count,'relative header')
    replace('"../include/','"../../include/',2,'relative common includes')
    restored=text
    for old,new,count,label in reversed(edits):
        if restored.count(new)!=count: raise ValueError(('inverse drift',label))
        restored=restored.replace(new,old)
    if restored!=original: raise ValueError('source byte identity failed')
    for token in ('Fence<HardEvent::','PipeBarrier<','CrossCoreSetFlag<','SetHF32Mode('):
        if masked(text).count(token)!=masked(original).count(token): raise ValueError('sync/math drift '+token)
    if profile and masked(original).count('CrossCoreWaitFlag(')!=masked(text).count('ProfileWait(')-1:
        raise ValueError('wait count drift')
    header='''// GENERATED DIAGNOSTIC C4 ONLY. Archive #126/#132-135/#140/#145/#148/#154/#155.
// D.4: fused FIA attribution; no 6.5s full-history restore or new data path.
// Reversible markers preserve all original math/fences/flags/error handling.
// Spill/restore scopes are nested; do not add actor totals or count waits twice.
// Q1344/R1 is a prefill partition, not K32 whole-step/end-to-end evidence.
'''
    return header+text,{'arm':arm,'symbol':'oscar_c4_'+arm+'_kernel',
        'inverse_source_bytes_identical':True,'new_completion_fences':0,
        'pointer_count':17 if profile else 16,'edits':[
            {'label':label,'count':count,'before_sha256':sha(old),'after_sha256':sha(new)}
            for old,new,count,label in edits]},fields


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    manifest={'schema_version':1,'diagnostic_only':True,'source':SOURCE,'dependency_sha256':PINS,
        'cache_format':'production striped_v1 per-page window, not request-window layout',
        'abi':{'profile_schema':3,'profile_shape':['cube_cores',3,4,32],
               'profile_pointer_position_0based':16,'int64_attributes':17,'float_attributes':1,
               'profile_bytes_per_core':3072,'cluster_stats_bytes_per_core':64,
               'workspace_bytes_per_core':1839104},'arms':{},
        'validation':{name:'not_run' for name in ('cann_compile','cpu_debug','camodel','npu','graph','whole_service')}}
    for profile in (False,True):
        text,entry,fields=generate(profile);dest=HERE/(entry['arm']+'.cpp')
        entry['sha256']=sha(text);manifest['arms'][entry['arm']]=entry
        if args.check:
            if dest.read_text()!=text: raise ValueError('generated source stale')
        else: dest.write_text(text)
    manifest['abi']['fields']=dict(enumerate(fields))
    text=json.dumps(manifest,indent=2)+'\n';dest=HERE/'manifest.json'
    if args.check:
        if dest.read_text()!=text: raise ValueError('manifest stale')
    else: dest.write_text(text)
    print(json.dumps({'status':'checked' if args.check else 'generated','source_identity':True,'device_execution':'not_run'}))


if __name__=='__main__': main()
