"""Exact decode Softmax extraction; no substitute math or full-CV claims.

Reuse the qualified actual-method CPU template with decode's real capacities,
lane16 and stats64. Independently bind q4/q1 complete source/method hashes.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.decode_finite_20261015.generate import generate,sha,digest,method
from csrc.experiments.c16_b1_followup_20261015.finite_probe import TEMPLATE

def artifacts():
    template=TEMPLATE
    for old,new in [('kHalfRows=32,kLaneRows=64','kHalfRows=16,kLaneRows=16'),
       ('kElements=4096','kElements=16384'),('dequantBuf,32768','dequantBuf,65536'),
       ('rowbase/64','rowbase/16'),('naturalBuf,17408','naturalBuf,34816'),
       ('scoreBuf,32768','scoreBuf,16384'),('accBuf,65536','accBuf,16384'),
       ('statsBuf,768','statsBuf,256'),('scratchBuf,1024','scratchBuf,2048'),
       ('64*D','16*D'),('192','64')]:
        if old not in template:raise ValueError('actual template anchor '+old)
        template=template.replace(old,new)
    template=template.replace('int64_t rowbase,int64_t initialError)', 'int64_t rowbase,int64_t initialError,int64_t postCommit)')
    template=template.replace('source,ctx,queries,group,start,end,sink,recent,rowbase,initialError);',
        'source,ctx,queries,group,start,end,sink,recent,rowbase,initialError,postCommit);')
    template=template.replace('    Softmax(start,static_cast<int32_t>(rowbase));',
        '''    Softmax(start,static_cast<int32_t>(rowbase));
    if(postCommit) {pendingKeyError[0]=3;pendingKeyError[1]=0;
      pendingValueError[0]=pendingValueError[1]=0;CommitPrefetchErrors();}''')
    template=template.replace('@@SOFTMAX@@','@@SOFTMAX@@\n@@COMMIT@@')
    template=template.replace('  int32_t lane,error;',
        '  static constexpr int32_t kKvSubtiles=2;\n  int32_t pendingKeyError[2],pendingValueError[2];\n  int32_t lane,error;')
    result={};manifest={}
    for family in ('q4','q1'):
        for arm in ('baseline','candidate'):
            source,binding=generate(family,arm)
            if (HERE/f'{family}_{arm}.cpp').read_text()!=source:raise ValueError('complete source drift')
            softmax=source[slice(*method(source,'Softmax'))];visible=source[slice(*method(source,'AddVisibleRange'))]
            commit=source[slice(*method(source,'CommitPrefetchErrors'))]
            text=template.replace('@@VISIBLE@@',visible).replace('@@SOFTMAX@@',softmax).replace('@@COMMIT@@',commit).replace('@@ARM@@',family+'_'+arm)
            name=f'{family}_{arm}.cpp';result[name]=text
            manifest[name]={'source_sha256':binding['generated_sha256'],'Softmax_sha256':sha(softmax),
               'AddVisibleRange_sha256':sha(visible),'CommitPrefetchErrors_sha256':sha(commit),'extracted_methods_exact':True,'probe_sha256':sha(text)}
    original=(ROOT/'csrc/experiments/c16_b1_followup_20261015/finite_probe/main.cpp').read_text();text=original
    for old,new in [('finite_probe_baseline','FINITE_BASELINE'),('finite_probe_candidate','FINITE_CANDIDATE'),
        ('128*256','32*256'),('64*256','16*256'),('192','64'),('i<64?','i<16?'),('i<128?','i<32?'),
        ('r<32','r<16'),('rowbase+7','rowbase+3'),('rowbase+31','rowbase+15'),
        ('int64_t{21}','int64_t{FINITE_QUERIES}'),('1,96,0,0','1,16,0,0')]:
        if old not in text:raise ValueError('actual main anchor '+old)
        text=text.replace(old,new)
    needle=' {"normal_full",0,0,320,64,320,1,0,0,0},'
    if text.count(needle)!=1:raise ValueError('main case anchor')
    text=text.replace(needle,needle+'\n {"second_lane_full_or_empty",0,0,320,64,320,1,16,0,0},')
    text=text.replace('int expected;};','int expected;int64_t postCommit=0;};')
    text=text.replace('int64_t,int64_t\nextern', 'int64_t,int64_t,int64_t\nextern')
    text=text.replace('c.rowbase,c.initial)','c.rowbase,c.initial,c.postCommit)')
    needle=' {"existing_metadata_error_preserved",0,0,320,64,320,1,0,3,3},'
    if text.count(needle)!=1:raise ValueError('incoming error case anchor')
    text=text.replace(needle,needle+'''\n {"score_NaN_overwrites_incoming_metadata",4,0,320,64,320,1,0,3,2,0},
 {"score_NaN_then_pending_metadata_commit",4,0,320,64,320,1,0,3,3,1},''')
    result['main.cpp']=text;manifest['main.cpp']={'template_sha256':sha(original),'generated_sha256':sha(text),
       'cases_per_family':15,'scope':'actual Softmax exceptional values, active6/16/8/0 rows, original precise fallback, exact Commit error order'}
    result['manifest.json']=json.dumps(manifest,indent=2)+'\n'
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();folder=HERE/'finite_probe';folder.mkdir(exist_ok=True)
    for name,text in artifacts().items():
        path=folder/name
        if a.check:
            if path.read_text()!=text:raise ValueError('probe drift '+name)
        else:path.write_text(text)
    print(json.dumps({'status':'checked' if a.check else 'generated','CPU':'not_run'}))
