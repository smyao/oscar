# Archive G7/G8/#79-85/#91/#126/#145; startup D.4: independent current-only
# experiment. No production CMake, binding or runtime route is modified.
cmake_minimum_required(VERSION 3.18)
project(oscar_current_only_experiment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "isolated experiment build type" FORCE)
endif()
if(NOT SOC_VERSION STREQUAL "ascend910b4")
  message(FATAL_ERROR "current-only experiment requires SOC_VERSION=ascend910b4")
endif()
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "set ASCEND_HOME_PATH to the CANN 9.1 root")
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "target compute unit" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")

if(OSCAR_CURRENT_ONLY_CPU_DEBUG)
  find_package(tikicpulib REQUIRED
    PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
  add_executable(oscar_current_only_cpu current_only_cpu.cpp current_only.cpp
    baseline_cast.cpp ../../kernels/merge_lse.cpp)
  target_link_libraries(oscar_current_only_cpu PRIVATE tikicpulib::Ascend910B4)
  target_compile_options(oscar_current_only_cpu PRIVATE -O0 -g)
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
  ascendc_library(oscar_current_only_experimental SHARED current_only.cpp)
endif()
