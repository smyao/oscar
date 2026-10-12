"""D.4/#126/#129/#145/#148-155: deterministic source0 oracle for local CANN only."""
from pathlib import Path
import argparse
from tools.export_striped_prefill_cpu_cases import _write_case, _bytes
from tools.probe_history_reuse import Shape, make_fixture, SINK

def export(directory: Path):
    import torch
    from oscar_ascend.ops.reference import decode_kv
    torch.set_num_threads(4)
    for q,context in ((21,511),(21,8513),(385,511)):
        spec=Shape(f"q{q}_c{context}",(q,),(context,),256,1,1,False,recent_tokens=256)
        _write_case(torch,directory,spec,"shared_fp32")
        fixture=make_fixture(torch,spec)
        data=fixture["cpu"]
        raw=data["raw"]
        from tools.probe_history_reuse import PREFIX,BLOCK_TOKENS
        rows=[]
        for p in range(context):
            virtual=int(data["table"][0,p//128]);ratio=BLOCK_TOKENS//128
            physical=virtual//ratio;inpage=(virtual%ratio)*128+p%128
            off=PREFIX+physical*fixture["stride"]+inpage*136
            rows.append(raw[off:off+136])
        packed=torch.stack(rows)[:,None,:].contiguous()
        k,v=decode_kv(packed,256)
        out=torch.empty((q,6,256));lse=torch.empty((q,6))
        for token in range(q):
            end=min(context,max(SINK,context+token+1-256))
            scores=data["qr"][token]@k[SINK:end,0].T/16.0
            out[token]=(scores.softmax(-1)@v[SINK:end,0])@data["rv"].T
            lse[token]=scores.logsumexp(-1)
        case=directory/spec.name
        (case/"source0_output.bin").write_bytes(_bytes(out))
        (case/"source0_lse.bin").write_bytes(_bytes(lse))
        print(spec.name,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path)
    export(p.parse_args().directory)
