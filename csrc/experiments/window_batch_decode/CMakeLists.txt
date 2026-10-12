# Archive #126/#129/#140/#145/#148/#155, startup D.4: isolated current-SIMD
# denominator and window-DMA candidate. No production target or registration.
cmake_minimum_required(VERSION 3.18)
project(oscar_window_batch_decode LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "CANN merge build type" FORCE)
if(NOT SOC_VERSION STREQUAL "ascend910b4" OR NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "requires ASCEND_HOME_PATH and SOC_VERSION=ascend910b4")
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "target compute" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")
if(OSCAR_WINDOW_BATCH_CPU_DEBUG)
  find_package(tikicpulib REQUIRED PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
  add_executable(oscar_window_batch_decode_cpu probe.cpp window_batch_decode.cpp
    ../../kernels/attention_cv_striped_decode_simd.cpp
    ../../kernels/attention_tasks.cpp ../../kernels/merge_lse.cpp)
  target_link_libraries(oscar_window_batch_decode_cpu PRIVATE tikicpulib::Ascend910B4)
  target_include_directories(oscar_window_batch_decode_cpu SYSTEM AFTER PRIVATE
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_window_batch_decode_cpu PRIVATE -O0 -g)
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
  ascendc_library(oscar_window_batch_decode SHARED window_batch_decode.cpp
    ../../kernels/attention_cv_striped_decode_simd.cpp)
endif()
