project(DistributedCrack.GPU LANGUAGES CUDA CXX)
enable_language(CUDA)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)
add_library(DistributedCrack.GPU SHARED src/md5Cuda.cu src/md5_gpu.cpp)
target_include_directories(DistributedCrack.GPU PRIVATE  ./kernelinclude PUBLIC ${Boost_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ./include)
target_link_libraries(DistributedCrack.GPU PRIVATE cuda)
target_compile_options(DistributedCrack.GPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -g 
                       -G 
                       -pg
                       -O0
                       >)