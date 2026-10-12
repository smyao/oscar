# Isolated CANN/CPU verification of D256 striped prefill wrappers. The normal
# production CMake and fe0 operator are configured elsewhere. D.4/#126/#154.
cmake_minimum_required(VERSION 3.18)
project(oscar_prefill_window_range_experiment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT SOC_VERSION STREQUAL "ascend910b4")
  message(FATAL_ERROR "striped prefill wrappers require ascend910b4")
endif()
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH required")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CANN merge requires nonempty type" FORCE)
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "Ascend compute unit" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")
set(_kernels ../../kernels)
if(OSCAR_PREFILL_WINDOW_RANGE_CPU_DEBUG)
  find_package(tikicpulib REQUIRED PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
  add_executable(oscar_prefill_window_range_cpu probe.cpp
    ${_kernels}/attention_tasks.cpp base.cpp
    balanced.cpp
    cluster4.cpp
    cluster16.cpp
    ${_kernels}/attention_cv_striped.cpp
    ${_kernels}/attention_cv_striped_balanced.cpp
    ${_kernels}/attention_cv_striped_cluster4.cpp
    ${_kernels}/attention_cv_striped_cluster16.cpp
    ${_kernels}/merge_lse.cpp)
  target_link_libraries(oscar_prefill_window_range_cpu PRIVATE tikicpulib::Ascend910B4)
  target_include_directories(oscar_prefill_window_range_cpu SYSTEM AFTER PRIVATE
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_prefill_window_range_cpu PRIVATE -O0 -g)
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
  ascendc_library(oscar_prefill_window_range_experimental SHARED
    base.cpp balanced.cpp cluster4.cpp cluster16.cpp
    ${_kernels}/attention_cv_striped.cpp
    ${_kernels}/attention_cv_striped_balanced.cpp
    ${_kernels}/attention_cv_striped_cluster4.cpp
    ${_kernels}/attention_cv_striped_cluster16.cpp)
endif()
