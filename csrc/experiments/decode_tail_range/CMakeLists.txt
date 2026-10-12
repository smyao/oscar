# Archive #126/#129/#140/#145/#148/#154/#155/D.4: isolated static FP32 decode.
cmake_minimum_required(VERSION 3.18)
project(oscar_tail_range LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "CANN build type" FORCE)
if(NOT SOC_VERSION STREQUAL "ascend910b4" OR NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "requires ASCEND_HOME_PATH and SOC_VERSION=ascend910b4")
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "target compute" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")
if(OSCAR_TAIL_RANGE_CPU_DEBUG)
  find_package(tikicpulib REQUIRED PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
  add_executable(oscar_tail_range_cpu ../window_batch_decode/probe.cpp attention_tail_range.cpp
    ../../kernels/attention_cv_striped_decode_simd.cpp
    ../../kernels/attention_tasks.cpp ../../kernels/merge_lse.cpp)
  target_compile_definitions(oscar_tail_range_cpu PRIVATE
    OSCAR_CANDIDATE_Q4_KERNEL=oscar_tail_range_q4_kernel
    OSCAR_CANDIDATE_Q1_KERNEL=oscar_tail_range_q1_kernel
    OSCAR_CANDIDATE_LABEL="tail_range")
  target_link_libraries(oscar_tail_range_cpu PRIVATE tikicpulib::Ascend910B4)
  target_include_directories(oscar_tail_range_cpu SYSTEM AFTER PRIVATE
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_tail_range_cpu PRIVATE -O0 -g)
else()
  foreach(candidate "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake"
                    "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
    if(EXISTS "${candidate}/ascendc.cmake")
      set(ASCENDC_CMAKE_DIR "${candidate}")
      break()
    endif()
  endforeach()
  if(NOT ASCENDC_CMAKE_DIR)
    message(FATAL_ERROR "CANN ascendc.cmake missing")
  endif()
  include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
  ascendc_library(oscar_tail_range SHARED attention_tail_range.cpp
    ../../kernels/attention_cv_striped_decode_simd.cpp)
endif()
