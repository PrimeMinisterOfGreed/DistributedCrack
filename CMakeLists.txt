
cmake_minimum_required(VERSION 3.10)
project(distributed_cracker C CXX )

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
add_subdirectory(libs/md5)
if(ENABLE_CUDA)
    add_subdirectory(libs/gpu_md5)
endif()
add_subdirectory(rust_core)
add_subdirectory(core)

