"""Use the fixed independent writer goldens; only add new source provenance."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.striped_store_simd_pack.fixture import export as previous_export
from csrc.experiments.hadamard_batch8.generate import check,source_binding


def export(path):
    check();report=previous_export(path);binding=source_binding()
    report['original_writer_oracle_binding']=report['writer_source_binding'];report['writer_source_binding']=binding
    for row in report['cases']:
        row['original_writer_oracle_binding']=row['writer_source_binding'];row['writer_source_binding']=binding
        (path/row['name']/'fixture.json').write_text(json.dumps(row,indent=2)+'\n')
    (path/'cases.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps({'cases':[x['name'] for x in export(a.output)['cases']],'execution':'host_oracle_only'}))
