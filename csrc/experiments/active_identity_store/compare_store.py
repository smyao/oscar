"""Archive #140/#148/#155/D.4: compare completed, byte-identical store tasks.

No model tick is treated as NPU ms; a large store speedup needs measured decode weights.
"""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--grid',type=int,default=16);a=p.parse_args()
root=Path('reports/active_identity_store_camodel_20261006')
variants=['old','active'] if 'hadamard' in a.case else ['old','active','identity']
reports={}
for v in variants:
 stem=f'{a.case}-g{a.grid}-{v}';directory=root/stem
 j=json.loads((directory/'report.json').read_text());h=json.loads((root/(stem+'-host.json')).read_text())
 c=json.loads((directory/'completion.json').read_text())
 assert j['status']=='passed' and not j['failures'] and c['status']=='device_complete_outputs_saved' and not c['failures']
 assert h['process_rc']==0 and h['fetch_rc']==0 and h['extract_rc']==0 and not h['timed_out'] and not h['interrupted']
 assert c['outputs']==j['outputs'] and c['input_sha256']==j['input_sha256']
 for name,m in j['outputs'].items():
  data=(directory/m['path']).read_bytes()
  assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256']==j['oracle_sha256'][name]
 reports[v]=j
base=reports['old']
for j in reports.values():
 for key in ('case','geometry','grid','actors','scope','input_sha256','initial_sha256','oracle_sha256','library_sha256','outputs'):
  assert j[key]==base[key],key
result={'status':'completed_matched_store_comparison','case':a.case,'grid':a.grid,
 'ticks':{v:j['task_ticks'] for v,j in reports.items()},
 'speedup_vs_old':{v:base['task_ticks']/j['task_ticks'] for v,j in reports.items()},
 'all_output_bytes_identical':True,'scope':base['scope'],'input_sha256':base['input_sha256'],
 'initial_sha256':base['initial_sha256'],'output_sha256':{k:v['sha256'] for k,v in base['outputs'].items()},
 'library_sha256':base['library_sha256'],'NPU':'not_run','graph':'not_run','whole_decode':'not_run','whole_service':'not_run','promotion_allowed':False}
(root/(a.case+'-g'+str(a.grid)+'-comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
