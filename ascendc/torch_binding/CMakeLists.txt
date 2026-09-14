cmake_minimum_required(VERSION 3.16)
project(oscar_ascend_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Torch REQUIRED)

if(NOT DEFINED VLLM_ASCEND_SOURCE OR NOT DEFINED TORCH_NPU_PATH OR
   NOT DEFINED ASCEND_HOME_PATH)
  message(FATAL_ERROR "VLLM_ASCEND_SOURCE, TORCH_NPU_PATH and ASCEND_HOME_PATH are required")
endif()

add_library(oscar_ascend_torch SHARED
  oscar_ascend_torch.cpp
  ${VLLM_ASCEND_SOURCE}/csrc/aclnn_torch_adapter/NPUBridge.cpp
  ${VLLM_ASCEND_SOURCE}/csrc/aclnn_torch_adapter/NPUStorageImpl.cpp)

target_include_directories(oscar_ascend_torch PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${VLLM_ASCEND_SOURCE}/csrc
  ${VLLM_ASCEND_SOURCE}/csrc/aclnn_torch_adapter)

target_link_directories(oscar_ascend_torch PRIVATE
  ${TORCH_NPU_PATH}/lib
  ${ASCEND_HOME_PATH}/lib64)

target_link_libraries(oscar_ascend_torch PRIVATE
  ${TORCH_LIBRARIES} torch_npu ascendcl dl)

set_target_properties(oscar_ascend_torch PROPERTIES
  PREFIX "lib"
  BUILD_RPATH "${TORCH_NPU_PATH}/lib;${ASCEND_HOME_PATH}/lib64")
