# Archive #126/#148/#154 and startup D.4: local Lima CAModel experiment only.
# Existing compiled binaries and deterministic synthetic q4 fixture; not target NPU acceptance.
import ctypes,math,struct,hashlib
import os,threading,time
from pathlib import Path
fd_root=Path("/home/sunao2000.linux/oscar-simulator-check/ctx8191_triplet/log_ca")
fd_stop=threading.Event();fd_redirected=set()
def discard_own_dumps():
 sink=os.open("/dev/null",os.O_WRONLY)
 try:
  while not fd_stop.is_set():
   for item in os.listdir("/proc/self/fd"):
    if not item.isdigit():continue
    fd=int(item)
    if fd==sink:continue
    try:
     target=os.readlink(f"/proc/self/fd/{fd}")
     if target.startswith(str(fd_root)+"/") and target.endswith(".dump"):
      os.dup2(sink,fd);fd_redirected.add(target)
    except OSError:pass
   time.sleep(0.001)
 finally:os.close(sink)
fd_thread=threading.Thread(target=discard_own_dumps,daemon=True)
fd_thread.start()
from ctypes import c_void_p,c_size_t,c_int,c_int64,c_uint32,c_float,byref
acl=ctypes.CDLL("libascendcl.so",mode=ctypes.RTLD_GLOBAL)
def api(name,args):
 fn=getattr(acl,name);fn.argtypes=args;fn.restype=c_int;return fn
init=api("aclInit",[ctypes.c_char_p]);setdev=api("aclrtSetDevice",[c_int]);reset=api("aclrtResetDevice",[c_int]);finalize=api("aclFinalize",[])
alloc=api("aclrtMalloc",[ctypes.POINTER(c_void_p),c_size_t,c_int]);free=api("aclrtFree",[c_void_p]);memcpy=api("aclrtMemcpy",[c_void_p,c_size_t,c_void_p,c_size_t,c_int]);create=api("aclrtCreateStream",[ctypes.POINTER(c_void_p)]);destroy=api("aclrtDestroyStream",[c_void_p]);sync=api("aclrtSynchronizeStream",[c_void_p])
def check(name,rc):
 if rc:raise RuntimeError(f"{name} rc={rc}")
check("init",init(None));check("setdev",setdev(0))
lib=ctypes.CDLL("/home/sunao2000.linux/batched4-experiment-20260929/build-striped-decode-simd-device/lib/liboscar_striped_decode_experimental.so",mode=ctypes.RTLD_GLOBAL)
argtypes=[c_uint32,c_void_p]+[c_void_p]*15+[c_int64]*17+[c_float]
old=lib.aclrtlaunch_oscar_attention_cv_fast_kernel;old.argtypes=argtypes;old.restype=c_uint32
new=lib.aclrtlaunch_oscar_attention_cv_striped_decode_kernel;new.argtypes=argtypes;new.restype=c_uint32
simd=lib.aclrtlaunch_oscar_attention_cv_striped_decode_simd_kernel;simd.argtypes=argtypes;simd.restype=c_uint32
fixture="/home/sunao2000.linux/oscar-simulator-check/oscar-camodel-q4-cases-20260930/synthetic_q4_ctx8191/"
evidence="/home/sunao2000.linux/oscar-simulator-check/oscar-camodel-q4-cases-20260930/synthetic_q4_ctx8191/"
values=open(evidence+"attrs.txt").read().split();assert len(values)==19
attrs=[int(v) for v in values[:-2]];scale=float(values[-2]);fixture_cores=int(values[-1]);assert fixture_cores==20
n,hq,hk,d,requests,columns,count,blockTokens,blocks,prefix,stride,windowStride,tagStride,sink,recent,spec,splits=attrs
assert (n,hq,hk,d,requests,count)==(4,6,1,256,1,12)
cores=20;segments=3*splits;windowRows=sink+recent+spec
sizes={"q":n*hq*d*2,"qr":n*hq*d*4,"ck":n*hk*d*2,"cv":n*hk*d*2,"rv":d*d*4,"raw_old":prefix+blocks*stride,"raw_new":prefix+blocks*stride,"table":requests*columns*4,"wk":blocks*windowStride*2,"wv":blocks*windowStride*2,"tags":blocks*tagStride*8,"tasks":count*16*8}
paths={name:fixture+("raw.bin" if name=="raw_old" else name+".bin") for name in sizes}
paths["raw_new"]=evidence+"raw_striped.bin";paths["tasks"]=evidence+"tasks.bin"
ptrs={};hosts={}
for name,size in sizes.items():
 raw=open(paths[name],"rb").read();assert len(raw)==size,(name,len(raw),size)
 ptr=c_void_p();check("malloc_"+name,alloc(byref(ptr),size,0));ptrs[name]=ptr
 host=ctypes.create_string_buffer(raw,size);hosts[name]=host
 check("h2d_"+name,memcpy(ptr,size,host,size,1))
output_sizes={"partial":n*hq*segments*d*4,"lse":n*hq*segments*4,"status":count*2*4,"workspace":cores*917504}
for tag in ("old","new","simd"):
 for name,size in output_sizes.items():
  ptr=c_void_p();check("malloc_"+tag+name,alloc(byref(ptr),size,0));ptrs[tag+name]=ptr
stream=c_void_p();check("createStream",create(byref(stream)))
for tag,fn,raw in (("old",old,"raw_old"),("new",new,"raw_new"),("simd",simd,"raw_new")):
 pointers=[ptrs[x] for x in ("q","qr","ck","cv","rv",raw,"table","wk","wv","tags","tasks",tag+"partial",tag+"lse",tag+"status",tag+"workspace")]
 check(tag+"launch",fn(cores,stream,*pointers,*attrs,scale))
 check(tag+"sync",sync(stream))
 print("LAUNCH",tag,"grid",cores,"status=complete",flush=True)
outputs={}
for tag in ("old","new","simd"):
 for name,size in output_sizes.items():
  if name=="workspace":continue
  buf=ctypes.create_string_buffer(size);check("d2h_"+tag+name,memcpy(buf,size,ptrs[tag+name],size,2))
  outputs[tag+name]=buf.raw
  open(tag+"_"+name+".bin","wb").write(buf.raw)
failures=[]
for name in ("partial","lse","status"):
 a,b=outputs["old"+name],outputs["new"+name]
 equal=a==b
 print("BITWISE_CHECK",name,equal,"first",next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None),flush=True)
 if not equal:failures.append("basic_"+name+"_bitwise")
for name in ("partial","lse","status"):
 a,b=outputs["old"+name],outputs["simd"+name]
 equal=a==b
 print("SIMD_BITWISE_CHECK",name,equal,"first",next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None),flush=True)
 if not equal:failures.append("simd_"+name+"_bitwise")
for tag in ("old","new","simd"):
 status=outputs[tag+"status"]
 if any(x!=0 for x in struct.unpack("<"+"i"*(len(status)//4),status)):
  failures.append(tag+"_nonzero_status")
expected=open(fixture+"expected_output.bin","rb").read();expected_lse=open(fixture+"expected_lse.bin","rb").read()
partial=outputs["newpartial"];lse=outputs["newlse"];maxout=maxlse=0.0;badout=badlse=0
for row in range(n*hq):
 ls=[struct.unpack_from("<f",lse,4*(row*segments+s))[0] for s in range(segments)]
 mx=max(ls);ws=[math.exp(x-mx) if math.isfinite(x) else 0.0 for x in ls];den=sum(ws);assert den>0
 merged=mx+math.log(den);gold=struct.unpack_from("<f",expected_lse,4*row)[0]
 diff=abs(merged-gold);maxlse=max(maxlse,diff)
 if diff>0.005+0.005*abs(gold):badlse+=1
 for j in range(d):
  val=sum(ws[s]*struct.unpack_from("<f",partial,4*((row*segments+s)*d+j))[0] for s in range(segments))/den
  gold=struct.unpack_from("<f",expected,4*(row*d+j))[0]
  diff=abs(val-gold);maxout=max(maxout,diff)
  if diff>0.005+0.005*abs(gold):badout+=1
print("ORACLE max_output_abs",maxout,"bad_output",badout,"max_lse_abs",maxlse,"bad_lse",badlse,flush=True)
if badout or badlse:failures.append("frozen_oracle")
print("ORACLE_CHECK",not (badout or badlse),flush=True)
for p in ptrs.values():check("free",free(p))
check("destroyStream",destroy(stream));check("reset",reset(0));check("finalize",finalize())
fd_stop.set();fd_thread.join(timeout=2);print("FD_QUIET_REDIRECTED",len(fd_redirected),flush=True)
print("SIMULATOR_RESULT",{"status":"failed" if failures else "passed",
                          "failures":failures},flush=True)
if failures:raise SystemExit(2)
