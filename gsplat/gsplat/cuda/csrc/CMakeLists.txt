cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version
set(CMAKE_CUDA_ARCHITECTURES 70 75 89)  # Ti 2080 uses 75. V100 uses 70. RTX 4090 uses 89.
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/home/litt2407/.conda/envs/c3/bin/nvcc")
endif()
project(gsplat CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
find_package(Torch required)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# our library library
add_library(gsplat_test forward.cu backward.cu helpers.cuh)
target_link_libraries(gsplat PUBLIC cuda)
target_include_directories(gsplat PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(gsplat PROPERTIES CUDA_ARCHITECTURES "70;75;86")
# 如果是 Debug 模式，为 CUDA 编译器添加调试选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(gsplat PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
endif()
# # To add an executable that uses the gsplat library,
# # follow example in the comments for a script `run_forward.cpp`
# # Add the executable
# add_executable(run_forward run_forward.cpp)

# # Link against CUDA runtime library
# target_link_libraries(run_forward PUBLIC cuda gsplat)

# # Include directories for the header-only library
# target_include_directories(run_forward PRIVATE
#     ${PROJECT_SOURCE_DIR}/third_party/glm
# )
