# Archive G7/G8/#79-85/#126/#145/#148-155 and startup D.4: isolated SDK
# static-tiling feasibility test. No production CMake or dispatch mutation.
cmake_minimum_required(VERSION 3.18)
project(oscar_wide_fp32_experiment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "isolated experiment build type" FORCE)
endif()
if(NOT SOC_VERSION STREQUAL "ascend910b4")
  message(FATAL_ERROR "wide-FP32 experiment requires SOC_VERSION=ascend910b4")
endif()
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "set ASCEND_HOME_PATH to the CANN 9.1 root")
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "target compute unit" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")

if(OSCAR_WIDE_FP32_CPU_DEBUG)
  find_package(tikicpulib REQUIRED
    PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
  add_executable(oscar_wide_fp32_cpu wide_fp32_cpu.cpp wide_fp32.cpp)
  target_link_libraries(oscar_wide_fp32_cpu PRIVATE tikicpulib::Ascend910B4)
  target_include_directories(oscar_wide_fp32_cpu SYSTEM AFTER PRIVATE
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_wide_fp32_cpu PRIVATE -O0 -g)
else()
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
    message(FATAL_ERROR "CANN ascendc.cmake missing")
  endif()
  include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
  ascendc_library(oscar_wide_fp32_experimental SHARED wide_fp32.cpp)
endif()
