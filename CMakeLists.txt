
cmake_minimum_required(VERSION 3.10)
project(distributed_cracker)

include(CheckCXXCompilerFlag)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
add_subdirectory(libs/md5)
add_subdirectory(libs/gpu_md5)
add_subdirectory(core)

