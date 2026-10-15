"""Archive #148/#154: isolate last member QR NaN without changing shared KV."""
import argparse
import json
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from csrc.experiments.c16_member_pipeline.fixture import export
from csrc.experiments.c16_l1_reuse_20261015.generate import sha


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=export(a.output,337,-1,True);data=bytearray((a.output/'qr.bin').read_bytes())
    struct.pack_into('<I',data,(315*6*256)*4,0x7fc10001);(a.output/'qr.bin').write_bytes(data)
    meta['input_sha256']['qr']=sha(data)
    meta['injection']={'tensor':'qr','query':315,'head':0,'dimension':0,'bits':'0x7fc10001',
        'expected_error':'source0 leader315 lane0 code2 only',
        'oracle':'affected64 history rows NaN; remaining rows original dense oracle; all bytes paired'}
    (a.output/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta['injection']))
