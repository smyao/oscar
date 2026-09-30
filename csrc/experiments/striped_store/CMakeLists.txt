# Archive G26-G34/#13-22/#37-49/#126/#154: official CPU debugger runs
# accepted store and isolated striped store against the same PR goldens.
cmake_minimum_required(VERSION 3.18)
project(oscar_striped_store_cpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(tikicpulib REQUIRED PATHS "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake" NO_DEFAULT_PATH)
add_executable(oscar_striped_store_cpu striped_store_probe.cpp
  ../../kernels/rotate_clip_store.cpp ../../kernels/rotate_clip_store_striped.cpp)
target_link_libraries(oscar_striped_store_cpu PRIVATE tikicpulib::Ascend910B4)
target_compile_options(oscar_striped_store_cpu PRIVATE -O0 -g)
