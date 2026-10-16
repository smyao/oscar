import hashlib,json,os,resource,signal,sys,threading,time
from pathlib import Path
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from csrc.experiments.bf16_decode_20261016.replay import adapted
out=root/'reports/functional-cast-fixed-20261016'
binding=json.loads((root/'fixed_source_binding.json').read_text());library=root/'build/functional-fix/lib/liboscar_bf16_decode_fixed.so'
assert hashlib.sha256(library.read_bytes()).hexdigest()==binding['library_sha256']
path='csrc/kernels/attention_cv_bf16_decode.cpp'
assert hashlib.sha256((root/path).read_bytes()).hexdigest()==binding['sources'][path]
source=adapted();old="if model_top!={'sim_type':0,'num_aic':24,'num_aiv':48}:"
assert source.count(old)==1
source=source.replace(old,"if model_top!={'sim_type':0,'num_aic':1,'num_aiv':2}:")
assert resource.getrlimit(resource.RLIMIT_AS)[0]==resource.RLIM_INFINITY
stop=threading.Event();peak=0
limits={'RSS_limit_bytes':15*1024**3//2,'address_space_limit_bytes':None,'trace_total_limit_bytes':192*1024**2,'wall_limit_seconds':180}
(root/'reports/functional-cast-fixed-plan.json').write_text(json.dumps({'purpose':'single-core functional instruction validation; not timing or a baseline comparison','topology':{'AIC':1,'AIV':2},'blockDim':1,'case':'q4_finite_cpu','all_tasks':24,'library_sha256':binding['library_sha256'],'kernel_sha256':binding['sources'][path],'limits':limits,'hardware_NPU_access':False},indent=2)+'\n')
def watch():
 global peak
 while not stop.wait(.2):
  lines=Path('/proc/self/status').read_text().splitlines();rss=int(next(s for s in lines if s.startswith('VmRSS:')).split()[1])*1024;peak=max(peak,rss)
  size=sum(p.stat().st_size for p in (out/'log_ca').glob('*.dump') if p.is_file()) if (out/'log_ca').exists() else 0
  if rss>limits['RSS_limit_bytes'] or size>limits['trace_total_limit_bytes']:
   (root/'reports/functional-cast-fixed-resource-limit.json').write_text(json.dumps({'RSS_bytes':rss,'trace_bytes':size,'status':'resource_limit_reached_not_numeric_result'})+'\n');os.kill(os.getpid(),signal.SIGKILL);return
thread=threading.Thread(target=watch,daemon=True);thread.start()
sys.argv=[str(__file__),'--root',str(root),'--variant','candidate','--case','q4_finite_cpu','--output',str(out),'--fixture-root','fixtures','--export-root','oracle','--library',str(library),'--candidate-q4','oscar_attention_cv_bf16_decode_kernel','--candidate-q1','oscar_attention_cv_bf16_decode_q1_kernel','--candidate-workspace-bytes-per-core','1146880','--baseline-workspace-bytes-per-core','1146880','--cpu-reference','required','--simulator-config-dir',str(root/'functional-onecore-config'),'--instruction-budget-bytes','0']
try:
 ns={'__name__':'functional_onecore','__file__':str(__file__)};exec(compile(source,'functional_safe_replay','exec'),ns);rc=ns['main']()
 if (out/'report.json').exists():
  p=out/'report.json';r=json.loads(p.read_text());r['scope']='functional instruction validation only; no speed comparison';r['performance']='not_measured';r['peak_RSS_bytes']=peak;p.write_text(json.dumps(r,indent=2)+'\n')
 raise SystemExit(rc)
finally:stop.set();thread.join(timeout=1)
