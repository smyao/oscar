# Archive G7/G8/#79-85/#126/#129/#140-153 and startup D.4: isolated
# finite-guard experiment. Production CMake, binding and variant are untouched.
cmake_minimum_required(VERSION 3.18)
project(oscar_finite_guard_experiment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Isolated experiment build type" FORCE)
endif()
if(NOT SOC_VERSION STREQUAL "ascend910b4")
  message(FATAL_ERROR "Finite-guard experiment requires SOC_VERSION=ascend910b4")
endif()
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "Set ASCEND_HOME_PATH to the CANN 9.1 root")
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "Ascend target compute unit" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")

if(OSCAR_FINITE_GUARD_CPU_DEBUG)
  find_package(tikicpulib REQUIRED
    PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
  add_executable(oscar_finite_guard_cpu finite_guard_probe.cpp
    ../../kernels/attention_tasks.cpp ../../kernels/attention_cv_fast.cpp
    attention_cv_checked_fast.cpp ../../kernels/merge_lse.cpp)
  target_link_libraries(oscar_finite_guard_cpu PRIVATE tikicpulib::Ascend910B4)
  set(_softmax_v2 "${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/activation/softmaxflashv2.h")
  if(NOT EXISTS "${_softmax_v2}")
    message(FATAL_ERROR "Official CANN SoftmaxFlashV2 header missing")
  endif()
  target_include_directories(oscar_finite_guard_cpu SYSTEM AFTER PRIVATE
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_finite_guard_cpu PRIVATE -O0 -g)
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
  ascendc_library(oscar_finite_guard_experimental SHARED
    attention_cv_checked_fast.cpp)
endif()
