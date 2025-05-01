
cmake_minimum_required(VERSION 3.10)
project(distributed_cracker C CXX )

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
add_subdirectory(libs/md5)
add_subdirectory(libs/gpu_md5)
add_subdirectory(rust_core)
add_subdirectory(core)

