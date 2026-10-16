"""Archive #126/#145/#148/#154/#157; real prepare/BF16 CV/Rv/merge checker.

Adapt the current independent BF16 decode checker, adding complete cluster
counts and declared original lane error domains; no candidate-derived oracle.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'csrc/experiments/bf16_decode_20261016/probe.cpp'
PIN='b265fb67fb5465fb8e62635924cc7ceb1788afbe4237838264bda87f4493d060'


def sha(b):return hashlib.sha256(b.encode() if isinstance(b,str) else b).hexdigest()


def generate():
    old=SOURCE.read_text()
    if sha(old)!=PIN:raise ValueError('independent BF16 checker changed')
    s=old;edits=[]
    def edit(a,b,count=1):
        nonlocal s
        if s.count(a)!=count:raise ValueError(('checker anchor',a[:90],s.count(a),count))
        s=s.replace(a,b);edits.append((a,b,count))
    edit('    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \\\n',
         '    uint8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t, \\\n')
    edit('oscar_attention_cv_bf16_decode_q1_kernel','oscar_attention_cv_bf16_cluster16_kernel',2)
    edit('oscar_attention_cv_bf16_decode_kernel','oscar_attention_cv_bf16_cluster4_kernel',2)
    edit('"q4"','"c4"',s.count('"q4"'));edit('"q1"','"c16"',s.count('"q1"'))
    edit('oscar_bf16_decode_cpu q4|q1','oscar_bf16_clusters_cpu c4|c16')
    edit('hq/hk>8','hq/hk>16')
    edit('Gm work(cores*1146880),merged',
         'Gm stats(cores*64);\n    Gm work(cores*(mode=="c16"?7487488:2494464)),merged')
    edit('&status,&work,&merged','&status,&work,&stats,&merged')
    edit('const bool slotContext=mode=="c16";','const bool slotContext=false;')
    anchor='    const int64_t segments=3*splits,taskCount=n*hk*segments,windowRows=sink+recent+spec;'
    edit(anchor,'''    int64_t suppressCurrent,expectedClusters,badMetadata,badQr;
    std::ifstream clusterFlags(dir+"/cluster_flags.txt");
    if(!(clusterFlags>>suppressCurrent>>expectedClusters>>badMetadata>>badQr))throw std::runtime_error("cluster flags missing");
'''+anchor)
    edit('    for(int64_t id=0;id<taskCount;++id)if(At<int64_t>(tasks.ptr,id*16+1)>(mode=="c16"?1:4))throw std::runtime_error("task owner exceeds decode family");',
         '''    for(int64_t id=0;id<taskCount;++id) {
      auto* row=reinterpret_cast<int64_t*>(tasks.ptr)+id*16;
      if(row[1]>128/(hq/hk))throw std::runtime_error("task exceeds long query group");
      if(suppressCurrent && row[7]==2 && row[10]==0 && row[4]>=row[3])row[4]=row[3];
    }''')
    edit('tasks.ptr,partial.ptr,lse.ptr,status.ptr,work.ptr,n,hq,hk,d,requests,columns,taskCount,b,blocks,prefix,stride, \\\n',
         'tasks.ptr,partial.ptr,lse.ptr,status.ptr,work.ptr,stats.ptr,n,hq,hk,d,requests,columns,taskCount,b,blocks,prefix,stride, \\\n')
    edit('    guards();immutable();CheckStatuses(status,dir,fault!=0);',
         '''    guards();immutable();CheckStatuses(status,dir,fault!=0);
    int64_t clusters=0,members=0,skips=0;
    for(int64_t core=0;core<cores;++core){clusters+=At<int64_t>(stats.ptr,core*8);members+=At<int64_t>(stats.ptr,core*8+1);skips+=At<int64_t>(stats.ptr,core*8+6);}
    const int clusterSize=mode=="c16"?16:4;
    if(clusters!=expectedClusters || members!=expectedClusters*clusterSize || skips!=members)throw std::runtime_error("actual mature cluster ownership differs");
    for(int64_t id=0;id<taskCount;++id)for(int lane=0;lane<2;++lane) {
      auto* row=reinterpret_cast<int64_t*>(tasks.ptr)+id*16;int expected=0;
      if(row[1]>0 && row[7]==0) {
        if(badMetadata>=row[3] && badMetadata<row[4] && ((badMetadata-row[3])%32)/16==lane)expected=3;
        if(badQr>=row[0] && badQr<row[0]+row[1] && ((badQr-row[0])*(hq/hk))/64==lane)expected=2;
      }
      if(At<int32_t>(status.ptr,id*2+lane)!=expected)throw std::runtime_error("complete status/lane oracle at "+std::to_string(id)+":"+std::to_string(lane));
    }
    Write(output+"/cluster_stats.bin",stats.ptr,stats.size);''')
    edit('<<",\\"gm_guards\\":true,',
         '<<",\\"eligible_clusters\\":"<<clusters<<",\\"members\\":"<<members<<",\\"gm_guards\\":true,')
    restored=s
    for a,b,n in reversed(edits):
        if restored.count(b)!=n:raise ValueError('inverse checker ambiguity '+b[:60])
        restored=restored.replace(b,a)
    if restored!=old:raise ValueError('checker full inverse failed')
    return s,{'template':str(SOURCE.relative_to(ROOT)),'template_sha256':PIN,'generated_sha256':sha(s),
        'full_inverse':True,'oracle':'predeclared BF16v2 from actual tasks','source2':'optional declared production prefill suppression, every task retained'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();text,meta=generate()
    for name,body in [('probe.cpp',text),('harness.json',json.dumps(meta,indent=2)+'\n')]:
        path=HERE/name
        if a.check:
            if path.read_text()!=body:raise ValueError('cluster checker drift')
        else:path.write_text(body)
    print(json.dumps({'status':'checked' if a.check else 'generated','CPU':'not_run'}))
