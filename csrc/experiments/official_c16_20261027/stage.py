"""Archive #126/#145/#148/#164: source-bound staging, no native source writes."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[3]
BASE_COMMIT="232b563b69e01495046a9b0025ac9150bc11b9d7"
BASELINE_SHA256="df843b991b0112e97b524f9b2d10f53f8805a650c2187ae47345a545b0f0582f"
BASELINE_PATH="csrc/kernels/attention_cv_official_cluster16.cpp"


def baseline_source(reference):
    # #148: a later delivery commit must never silently become its own
    # control. Resolve the requested commit, then verify the original bytes
    # recorded in this experiment's source_manifest.json before any output.
    commit=subprocess.check_output(
        ["git","rev-parse","--verify","--end-of-options",reference+"^{commit}"],
        cwd=ROOT,text=True).strip()
    data=subprocess.check_output(["git","show",commit+":"+BASELINE_PATH],cwd=ROOT)
    actual=hashlib.sha256(data).hexdigest()
    if actual!=BASELINE_SHA256:
        raise ValueError(f"C16 baseline source SHA mismatch: commit={commit} expected={BASELINE_SHA256} actual={actual}")
    return commit,data.decode()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",default=BASE_COMMIT,
                        help="control Git commit/ref; its C16 bytes must match the recorded original SHA")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/long_context_20261027/c16",
                        help="directory for source.tar/source_manifest.json; use a fresh path to preserve evidence")
    args=parser.parse_args(argv)
    base,old=baseline_source(args.base)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    probe=(ROOT/"csrc/experiments/official_math_20261021/cluster_probe.cpp").read_text()
    declaration='extern "C" void oscar_attention_cv_official_cluster16_kernel(CV_ARGS);'
    assert probe.count(declaration)==1
    probe=probe.replace(declaration,declaration+'\nextern "C" void oscar_attention_cv_official_cluster16_control_20261027_kernel(CV_ARGS);')
    probe=probe.replace('#include "tikicpulib.h"','#include "tikicpulib.h"\n#include <cstdlib>')
    oldcall='else{CV_CALL(oscar_attention_cv_official_cluster16_kernel);}'
    assert probe.count(oldcall)==1
    probe=probe.replace(oldcall,'else if(std::getenv("OSCAR_C16_CONTROL") && std::string(std::getenv("OSCAR_C16_CONTROL"))=="1"){CV_CALL(oscar_attention_cv_official_cluster16_control_20261027_kernel);}'+oldcall)
    count='if(independent!=hk*splits)throw std::runtime_error("tail C1 leader missing");'
    assert probe.count(count)==1
    probe=probe.replace(count,'int64_t expectedIndependent=-1; std::ifstream expectedCounts(dir+"/expected_counts.txt");\n    if(!(expectedCounts>>expectedIndependent) || independent!=expectedIndependent)throw std::runtime_error("independent leader count differs");')
    control=old.replace('attention_cv_official_cluster16','attention_cv_official_cluster16_control_20261027')
    manifest={"base_commit":base,"baseline_sha256":hashlib.sha256(old.encode()).hexdigest(),"files":{}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        extras={"csrc/kernels/attention_cv_official_cluster16_control_20261027.cpp":control,
                "csrc/experiments/official_c16_20261027/paired_probe.cpp":probe}
        with tarfile.open(out/"source.tar","w") as tar:
            paths=set()
            for root in ("oscar_ascend","tools","csrc/kernels","csrc/include","csrc/experiments/official_math_20261021","csrc/experiments/official_c16_20261027","configs"):
                paths.update(p for p in (ROOT/root).rglob('*') if p.is_file() and p.suffix in ('.py','.h','.cpp','.txt','.json') and '__pycache__' not in p.parts)
            for p in sorted(paths):
                rel=str(p.relative_to(ROOT));tar.add(p,arcname=rel);manifest['files'][rel]=hashlib.sha256(p.read_bytes()).hexdigest()
            for rel,text in extras.items():
                p=tmp/Path(rel).name;p.write_text(text);tar.add(p,arcname=rel);manifest['files'][rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(out/"source.tar")


if __name__=="__main__":main()
