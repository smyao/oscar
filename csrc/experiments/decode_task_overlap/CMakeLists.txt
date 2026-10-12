# Archive G3-G13/#21/#81-85/#126/#140/#145/#148/#154-156; full startup D.4.
# Independent bounded two-task protocol; no production registration or native edits.
# CPU/CANN/complete CAModel/NPU/graph/performance qualifications remain separate.
cmake_minimum_required(VERSION 3.18)
project(oscar_decode_task_overlap LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "research build" FORCE)
if(NOT SOC_VERSION STREQUAL "ascend910b4" OR NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "Requires ascend910b4 and ASCEND_HOME_PATH")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate.py" --check
  --build-binding "${CMAKE_BINARY_DIR}/source_binding.json" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "Current-bundle source restoration or binding failed")
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "target" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")
if(OSCAR_DECODE_TASK_OVERLAP_CPU_DEBUG)
  find_package(tikicpulib REQUIRED PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
  add_executable(oscar_decode_task_overlap_cpu probe.cpp attention_baseline.cpp attention_candidate.cpp
    ../../kernels/attention_tasks.cpp ../../kernels/merge_lse.cpp)
  target_compile_definitions(oscar_decode_task_overlap_cpu PRIVATE
    OSCAR_CANDIDATE_Q4_KERNEL=oscar_task_overlap_candidate_q4_kernel
    OSCAR_CANDIDATE_Q1_KERNEL=oscar_task_overlap_candidate_q1_kernel
    OSCAR_CANDIDATE_WORKSPACE_BYTES_PER_CORE=2293760
    OSCAR_CANDIDATE_LABEL="task_overlap")
  target_link_libraries(oscar_decode_task_overlap_cpu PRIVATE tikicpulib::Ascend910B4)
  target_include_directories(oscar_decode_task_overlap_cpu SYSTEM AFTER PRIVATE "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_decode_task_overlap_cpu PRIVATE -O0 -g)
else()
  foreach(path "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake" "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
    if(EXISTS "${path}/ascendc.cmake")
      set(ASCENDC_CMAKE_DIR "${path}")
      break()
    endif()
  endforeach()
  if(NOT ASCENDC_CMAKE_DIR)
    message(FATAL_ERROR "CANN ascendc.cmake missing")
  endif()
  include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
  ascendc_library(oscar_decode_task_overlap SHARED attention_baseline.cpp attention_candidate.cpp)
endif()
