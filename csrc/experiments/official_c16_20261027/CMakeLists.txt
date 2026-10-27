# Archive G7-G13/#126/#145/#148/#164: actual SDK paired functional build.
cmake_minimum_required(VERSION 3.18)
project(oscar_long_context_c16 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(tikicpulib REQUIRED PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
add_executable(oscar_long_context_c16_cpu paired_probe.cpp
  ../../kernels/attention_cv_official_cluster4.cpp
  ../../kernels/attention_cv_official_cluster16.cpp
  ../../kernels/attention_cv_official_cluster16_control_20261027.cpp
  ../../kernels/attention_official_tasks.cpp ../../kernels/merge_lse.cpp)
target_link_libraries(oscar_long_context_c16_cpu PRIVATE tikicpulib::Ascend910B4)
target_include_directories(oscar_long_context_c16_cpu SYSTEM AFTER PRIVATE "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
target_compile_options(oscar_long_context_c16_cpu PRIVATE -O0 -g)
