# Archive G7/G8/#79-85/#126/#145/#148-153 and startup D.4: isolated CANN
# build for the M512 C4 research kernel. It never links into the production
# extension or alters its capability manifest, loader, or serving routes.
cmake_minimum_required(VERSION 3.18)
project(oscar_batched4_experiment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  # CANN 9.1 merge_mix_obj.sh requires a nonempty --build-type value.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Isolated experiment build type" FORCE)
endif()
if(NOT SOC_VERSION STREQUAL "ascend910b4")
  message(FATAL_ERROR "Batched4 experiment requires SOC_VERSION=ascend910b4")
endif()
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "Set ASCEND_HOME_PATH to the CANN 9.1 root")
endif()
set(ASCEND_COMPUTE_UNIT "${SOC_VERSION}" CACHE STRING "Ascend target compute unit" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(ENV{ASCEND_CANN_PACKAGE_PATH} "${ASCEND_HOME_PATH}")

if(OSCAR_BATCHED4_CPU_DEBUG)
  find_package(tikicpulib REQUIRED
    PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
  add_executable(oscar_batched4_cpu ../../cpu/batched4_probe.cpp
    ../../kernels/attention_tasks.cpp ../../kernels/attention_cv_fast_cluster4.cpp
    ../../kernels/attention_cv_batched4.cpp ../../kernels/merge_lse.cpp)
  target_link_libraries(oscar_batched4_cpu PRIVATE tikicpulib::Ascend910B4)
  set(_softmax_v2 "${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/activation/softmaxflashv2.h")
  if(NOT EXISTS "${_softmax_v2}")
    message(FATAL_ERROR "Official CANN SoftmaxFlashV2 header missing")
  endif()
  target_include_directories(oscar_batched4_cpu SYSTEM AFTER PRIVATE
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include")
  target_compile_options(oscar_batched4_cpu PRIVATE -O0 -g)
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
  ascendc_library(oscar_batched4_experimental SHARED
    ../../kernels/attention_cv_batched4.cpp)
endif()
