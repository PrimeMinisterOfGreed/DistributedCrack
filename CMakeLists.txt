
cmake_minimum_required(VERSION 3.24)
project(distributed_cracker)

include(CheckCXXCompilerFlag)
find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
add_subdirectory(libs/md5)
add_subdirectory(libs/gpu_md5)
add_subdirectory(distributed_cracker)


