# Archive G2-G13/#79-85/#98-110: native direct-library build, no copied OPP layer.
# Native references/vllm-ascend/CMakeLists.txt:41-81,96-105,159-197.
cmake_minimum_required(VERSION 3.18)
project(oscar_ascend_ops LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import torch, pybind11; print(torch.utils.cmake_prefix_path); print(pybind11.get_cmake_dir())"
  RESULT_VARIABLE _oscar_prefix_rc OUTPUT_VARIABLE _oscar_prefixes
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _oscar_prefix_rc EQUAL 0)
  message(FATAL_ERROR "Selected Python must provide torch and pybind11 for this target build")
endif()
string(REPLACE "\n" ";" _oscar_prefixes "${_oscar_prefixes}")
list(PREPEND CMAKE_PREFIX_PATH ${_oscar_prefixes})
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(NOT SOC_VERSION STREQUAL "ascend910b4")
  message(FATAL_ERROR "This build is scoped to SOC_VERSION=ascend910b4 (A2/AICORE220); do not use ascend910b")
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "Ascend target compute unit" FORCE)
if(NOT EXISTS "${TORCH_NPU_PATH}/include/torch_npu/csrc/core/npu/NPUStream.h")
  message(FATAL_ERROR "TORCH_NPU_PATH must be the package root containing include/torch_npu/csrc/core/npu/NPUStream.h")
endif()
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")
foreach(candidate
    "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake")
  if(EXISTS "${candidate}/ascendc.cmake")
    set(ASCENDC_CMAKE_DIR "${candidate}")
    break()
  endif()
endforeach()
if(NOT ASCENDC_CMAKE_DIR)
  message(FATAL_ERROR "No ascendc.cmake found under ASCEND_HOME_PATH; target CANN toolkit required")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
ascendc_library(oscar_ascend_kernels SHARED kernels/store_int2.cpp kernels/merge_lse.cpp
  kernels/rotate_clip_store.cpp kernels/attention_cv.cpp kernels/attention_tasks.cpp kernels/status_guard.cpp)
# Archive G18/#85/#98: a versioned ELF SONAME cannot bind to an older OSCAR
# library with the same generic filename through inherited LD_LIBRARY_PATH.
if(NOT OSCAR_BUILD_SIGNATURE MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "Build with tools.build_ops so the kernel SONAME is bound to its source signature")
endif()
string(SUBSTRING "${OSCAR_BUILD_SIGNATURE}" 0 16 OSCAR_BINARY_ID)
set_target_properties(oscar_ascend_kernels PROPERTIES OUTPUT_NAME "oscar_ascend_kernels_${OSCAR_BINARY_ID}")
pybind11_add_module(_oscar_ascend_ops torch_bindings.cpp rotation_bindings.cpp attention_bindings.cpp status_bindings.cpp)
target_include_directories(_oscar_ascend_ops PRIVATE ${TORCH_INCLUDE_DIRS}
  "${TORCH_NPU_PATH}/include" "${ASCEND_HOME_PATH}/include")
target_link_directories(_oscar_ascend_ops PRIVATE "${TORCH_NPU_PATH}/lib"
  "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(_oscar_ascend_ops PRIVATE oscar_ascend_kernels
  ${TORCH_LIBRARIES} torch_npu ascendcl)
set_target_properties(_oscar_ascend_ops PROPERTIES BUILD_RPATH
  "$ORIGIN;${TORCH_NPU_PATH}/lib;${ASCEND_HOME_PATH}/lib64")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/build_manifest.json" CONTENT
"{\n  \"abi_version\": 1,\n  \"execution_interface\": \"direct_launch\",\n  \"soc\": \"${SOC_VERSION}\",\n  \"extension\": \"$<TARGET_FILE:_oscar_ascend_ops>\",\n  \"kernel_library\": \"$<TARGET_FILE:oscar_ascend_kernels>\",\n  \"source_capabilities\": [\"store_int2_out\", \"merge_lse_out\", \"rotate_out\", \"rotate_clip_store_out\", \"prepare_attention_tasks_out\", \"attention_cv_out\", \"status_guard\"],\n  \"missing_capabilities\": [],\n  \"target_numerics_verified\": false,\n  \"graph_capture_verified\": false,\n  \"graph_replay_verified\": false,\n  \"performance_verified\": false\n}\n")
install(TARGETS _oscar_ascend_ops oscar_ascend_kernels LIBRARY DESTINATION oscar_ascend)
