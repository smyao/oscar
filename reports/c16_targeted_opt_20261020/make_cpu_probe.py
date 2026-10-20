"""Archive #126/#145/#148/#164: independent control-vs-candidate qualification.

D.4: this checks bounded fused FIA (old full-history restore was 6.5s/card);
no history tensor, changed precision, performance run or NPU claim is added.
The input/oracle generator is separate from either C16 implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SYMBOL = "oscar_attention_cv_aligned_cluster16_kernel"
CONTROL_SYMBOL = "oscar_attention_cv_aligned_cluster16_control_kernel"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"CPU probe source anchor changed: {old[:70]}")
    return text.replace(old, new, 1)


def generate_probe(*, symbol=CANDIDATE_SYMBOL):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", symbol):
        raise ValueError("kernel symbol must be a C identifier")
    if symbol == CONTROL_SYMBOL:
        raise ValueError("candidate symbol must differ from the preserved control")
    source = (ROOT / "csrc/experiments/aligned_math_20261018/cluster_probe.cpp").read_text()
    source = replace_once(
        source, 'extern "C" void oscar_attention_cv_aligned_cluster16_kernel(CV_ARGS);',
        f'extern "C" void {CONTROL_SYMBOL}(CV_ARGS);')
    source = replace_once(
        source, 'extern "C" void oscar_attention_cv_aligned_cluster4_kernel(CV_ARGS);',
        f'extern "C" void {symbol}(CV_ARGS);')
    source = replace_once(
        source,
        '    if(mode=="c4"){CV_CALL(oscar_attention_cv_aligned_cluster4_kernel);}else{CV_CALL(oscar_attention_cv_aligned_cluster16_kernel);}',
        '''    // #145: CPU is functional evidence; it cannot prove asynchronous DMA safety.
    std::memset(work.ptr,0xFF,work.size);
    CV_CALL(oscar_attention_cv_aligned_cluster16_control_kernel);
    guards();immutable();
    std::vector<std::vector<uint8_t>> original;
    for(const auto* gm:{&partial,&lse,&status,&stats})
      original.emplace_back(gm->ptr,gm->ptr+gm->size);
    // Different output poison rejects accidental reliance on old results.
    for(auto* gm:{&partial,&lse,&status,&stats})std::memset(gm->ptr,0xC6,gm->size);
    std::memset(work.ptr,0xFF,work.size);
    CV_CALL(@TUNED@);
    guards();immutable();
    {size_t i=0;for(const auto* gm:{&partial,&lse,&status,&stats}) {
      if(std::memcmp(gm->ptr,original[i].data(),gm->size))
        throw std::runtime_error("tuned output differs bitwise at output="+std::to_string(i));
      ++i;
    }}'''.replace("@TUNED@", symbol))
    source = replace_once(
        source, '    if(independent!=1)throw std::runtime_error("tail C1 leader missing");',
        '''    int64_t expectedIndependent;
    std::ifstream expectedCounts(dir+"/expected_counts.txt");
    if(!(expectedCounts>>expectedIndependent) || independent!=expectedIndependent)
      throw std::runtime_error("independent history leader count differs");''')
    start = source.index("    for(int64_t id=0;id<taskCount;++id)for(int lane=0;lane<2;++lane) {")
    end = source.index('    Write(output+"/cluster_stats.bin"', start)
    source = source[:start] + '''    // Independent fixture faults have explicit status checks. Every remaining
    // status and NaN payload is also compared bytewise with the unchanged kernel.
''' + source[end:]
    source = replace_once(
        source, '        <<",\\"includes_bf16_rv\\":true,',
        '        <<",\\"paired_bitwise\\":true,\\"timing_acceptance\\":false,\\"includes_bf16_rv\\":true,')
    source = replace_once(
        source, '(std::string(argv[1])!="c4" && std::string(argv[1])!="c16")',
        '(std::string(argv[1])!="c16")')
    source = replace_once(
        source, 'if(dequantMode<0 || dequantMode>1)',
        'if(dequantMode!=0)')
    return ("// Archive #145/#164; D.4 read in full. Independent generated paired probe.\n"
            "// Both arms use the same prepare ABI, FP32 softmax, BF16 operands and frozen v3 oracle.\n"
            "// No speed result is produced by this CPU functional executable.\n" + source)


def generate_cmake(tuned_source):
    return '''# Archive G7-G13/#145/#164: isolated official CPU functional build.
cmake_minimum_required(VERSION 3.18)
project(c16_tuned_paired_cpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(tikicpulib REQUIRED PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
add_executable(c16_tuned_paired_cpu probe.cpp "@TUNED@"
 "@ROOT@/csrc/kernels/attention_cv_aligned_cluster16_control.cpp"
 "@ROOT@/csrc/kernels/attention_tasks.cpp" "@ROOT@/csrc/kernels/merge_lse.cpp")
target_link_libraries(c16_tuned_paired_cpu PRIVATE tikicpulib::Ascend910B4)
target_include_directories(c16_tuned_paired_cpu PRIVATE "@ROOT@/csrc/kernels")
target_include_directories(c16_tuned_paired_cpu SYSTEM AFTER PRIVATE "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
target_compile_options(c16_tuned_paired_cpu PRIVATE -O0 -g)
'''.replace("@ROOT@", str(ROOT)).replace("@TUNED@", str(tuned_source.resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "csrc/kernels/attention_cv_aligned_cluster16.cpp")
    parser.add_argument("--symbol", default=CANDIDATE_SYMBOL)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error("tuned source is missing")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "probe.cpp").write_text(generate_probe(symbol=args.symbol))
    (args.output / "CMakeLists.txt").write_text(generate_cmake(args.source))


if __name__ == "__main__":
    main()
