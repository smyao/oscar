"""Archive #13/#126/#145/#148, D.4: offline final-Rv BF16 operand screen.
No change to device precision gates, no target execution or whole-history
serving materialization. FP32-Rv deltas are diagnostics under new authorization.
"""
from pathlib import Path
import json,hashlib,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import torch
from tools.bf16_reference import attention_tiles,round_bf16
from oscar_ascend.ops.reference import encode_kv,decode_kv,rotate_clip
from tools.probe_history_reuse import _hadamard
from tools.experiments.affine_bf16_residual_screen import cases
from tools.experiments.prefill_tile_stress import _signal

torch.set_num_threads(4)
def compare(name,acc,rv,source):
 old=acc@rv.T; rounded=round_bf16(acc); cast_rv=round_bf16(rv); new=rounded@cast_rv.T
 old_model=old.bfloat16().float();new_model=new.bfloat16().float();delta=new-old;delta_model=new_model-old_model
 denom=torch.linalg.vector_norm(old.double()).clamp_min(1e-30)
 return {'case':name,'source':source,'shape':list(acc.shape),'rv_exact_BF16':torch.equal(rv,cast_rv),'rv_round_max_abs':float((cast_rv-rv).abs().max()),'acc_round_max_abs':float((rounded-acc).abs().max()),'partial_max_abs_difference':float(delta.abs().max()),'partial_relative_l2':float(torch.linalg.vector_norm(delta.double())/denom),'final_BF16_max_abs_difference':float(delta_model.abs().max()),'final_BF16_relative_l2':float(torch.linalg.vector_norm(delta_model.double())/torch.linalg.vector_norm(old_model.double()).clamp_min(1e-30)),'final_BF16_changed_elements':int((new_model!=old_model).sum()),'diagnostic_old_FP32_tolerance_exceedances':int((delta.abs()>(.005+.005*old.abs())).sum()),'all_finite':bool(torch.isfinite(new).all())}
rows=[]
# Existing store-reachable stress family, including cancellation/heavy range Q
# and V amplitude128. Only finalRv differs; prior BF16 attention is identical.
for name,q,kr,vr,rv in cases():
 slots=encode_kv(kr,vr);k,v=decode_kv(slots,256)
 acc=attention_tiles(q.float()@rv,k,v).output
 rows.append(compare(name,acc,rv,'existing affine_bf16_residual_screen.cases; original INT2 store/decode'))
for d in (64,256):
 for mode in ('identity','hadamard'):
  q,k,v,rv=_signal(d,mode);slots=encode_kv(rotate_clip(k.float(),rv),rotate_clip(v.float(),rv));dk,dv=decode_kv(slots,d)
  acc=attention_tiles(q.float()@rv,dk,dv).output
  rows.append(compare(f'old_prefill_signal_D{d}_{mode}',acc,rv,'existing prefill_tile_stress._signal; original store/decode'))
g=torch.Generator().manual_seed(20261016)
for mode in ('identity','hadamard','general_orthogonal'):
 rv=torch.eye(256) if mode=='identity' else _hadamard(torch,256)
 if mode=='general_orthogonal':rv=torch.linalg.qr(torch.randn(256,256,generator=g))[0].contiguous()
 for kind in ('random_normal','random_heavy_tail'):
  acc=torch.randn(128,6,256,generator=g)
  if kind=='random_heavy_tail':acc=torch.ldexp(acc,torch.randint(-10,11,acc.shape,generator=g))
  rows.append(compare(f'{kind}_{mode}',acc,rv,'synthetic normalized-partial stress, not asserted model distribution'))
report={'status':'completed_host_screen_not_device_qualification','scope':'only normalized-acc/Rv operand rounding changes; both matmuls accumulateFP32','cases':rows,'predeclared_comparison_role':'FP32-Rv delta diagnostics only, not a rejection under the new user-authorized BF16 contract','native_FIA_internal_arithmetic':'not_observed','model_quality':'not_established','NPU':'not_run','cost_hypothesis':'single BF16 Matmul template can avoid separate FP32 Rv launch/gather/scatter; not measured','source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['tools/bf16_reference.py','tools/experiments/affine_bf16_residual_screen.py','tools/experiments/prefill_tile_stress.py']}}
out=root/'reports/bf16_attention_20261016';out.mkdir(parents=True,exist_ok=True);(out/'final_rv_host_screen.json').write_text(json.dumps(report,indent=2)+'\n')
for x in rows:print(x['case'],'max',x['partial_max_abs_difference'],'relL2',x['partial_relative_l2'],'finalRelL2',x['final_BF16_relative_l2'],'finite',x['all_finite'],'rvExact',x['rv_exact_BF16'])
