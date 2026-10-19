"""Archive #145/#164: paired CPU-debug validation, never NPU timing evidence."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    generated = Path(receipt['generation_manifest']).parent / 'c16_timer_kernel.cpp'
    source = (ROOT / 'csrc/experiments/aligned_math_20261018/cluster_probe.cpp').read_text()
    source = source.replace('extern "C" void oscar_attention_cv_aligned_cluster4_kernel(CV_ARGS);',
                            'extern "C" void oscar_c16_timer_kernel(CV_ARGS,uint8_t*);')
    source = source.replace('Gm stats(cores*64);', 'Gm stats(cores*64);Gm ticks(cores*3*512);')
    source = source.replace('&work,&stats,&merged', '&work,&stats,&ticks,&merged')
    start = source.index('#define CV_CALL(kernel)')
    end = source.index('#undef CV_CALL', start) + len('#undef CV_CALL')
    macro = source[start:source.index('    if(mode=="c4"){CV_CALL', start)]
    diag_macro = macro.rstrip()[:-1] + ',ticks.ptr)\n'
    paired = (macro + '    CV_CALL(oscar_attention_cv_aligned_cluster16_kernel);\n#undef CV_CALL\n'
              '    std::vector<std::vector<uint8_t>> original;\n'
              '    for(const auto* gm:{&partial,&lse,&status,&stats})original.emplace_back(gm->ptr,gm->ptr+gm->size);\n'
              + diag_macro + '    CV_CALL(oscar_c16_timer_kernel);\n#undef CV_CALL\n'
              '    {size_t index=0;for(const auto* gm:{&partial,&lse,&status,&stats}){\n'
              '      if(std::memcmp(gm->ptr,original[index].data(),gm->size))throw std::runtime_error("timer output differs bitwise at "+std::to_string(index));++index;}}\n'
              '    ticks.Guard();Write(output+"/ticks.bin",ticks.ptr,ticks.size);\n')
    source = source[:start] + paired + source[end:]
    source = source.replace('<<",\\"includes_bf16_rv\\":true,',
                            '<<",\\"paired_bitwise\\":true,\\"timer_clock_acceptance\\":\\"not_npu\\",\\"includes_bf16_rv\\":true,')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'probe.cpp').write_text(source)
    cmake = '''# Archive #145/#164: CPU-only paired numerical validation.
cmake_minimum_required(VERSION 3.18)
project(c16_timer_paired_cpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(tikicpulib REQUIRED PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
add_executable(c16_timer_paired_cpu probe.cpp "@GENERATED@"
 "@ROOT@/csrc/kernels/attention_cv_aligned_cluster16.cpp"
 "@ROOT@/csrc/kernels/attention_tasks.cpp" "@ROOT@/csrc/kernels/merge_lse.cpp")
target_link_libraries(c16_timer_paired_cpu PRIVATE tikicpulib::Ascend910B4)
target_include_directories(c16_timer_paired_cpu SYSTEM AFTER PRIVATE "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
target_compile_options(c16_timer_paired_cpu PRIVATE -O0 -g)
'''
    (args.output / 'CMakeLists.txt').write_text(cmake.replace('@ROOT@', str(ROOT)).replace('@GENERATED@', str(generated)))


if __name__ == '__main__':
    main()
