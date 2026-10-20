# Archive #164/#126/#145: actual CANN CPU-debug helper boundary probe.
# It neither launches an NPU nor measures performance.
cmake_minimum_required(VERSION 3.18)
project(oscar_c16_cast_cpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT OSCAR_SOURCE_ROOT OR NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "OSCAR_SOURCE_ROOT and ASCEND_HOME_PATH are required")
endif()
find_package(tikicpulib REQUIRED
  PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
add_executable(oscar_c16_cast_cpu main.cpp kernel.cpp)
target_link_libraries(oscar_c16_cast_cpu PRIVATE tikicpulib::Ascend910B4)
target_include_directories(oscar_c16_cast_cpu PRIVATE
  "${OSCAR_SOURCE_ROOT}/csrc/kernels")
target_include_directories(oscar_c16_cast_cpu SYSTEM AFTER PRIVATE
  "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
target_compile_options(oscar_c16_cast_cpu PRIVATE -O0 -g)
