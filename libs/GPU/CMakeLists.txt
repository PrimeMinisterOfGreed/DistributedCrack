project(DistributedCrack.GPU LANGUAGES CUDA CXX)
enable_language(CUDA)
find_package(CUDA REQUIRED)
include(../../cmake/TestProjectGenerator.cmake)
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
add_library(DistributedCrack.GPU SHARED ${SRC_FILES})
target_include_directories(DistributedCrack.GPU PRIVATE  ./kernelinclude PUBLIC ${Boost_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ./include)
target_link_libraries(DistributedCrack.GPU PRIVATE cuda)
target_compile_options(DistributedCrack.GPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -g 
                       -G 
                       -pg
                       -O0
                       -allow-unsupported-compiler
                       >)


generate_gtest_cuda(TEST_DIR "test" INCLUDE_DIR "include" SRC_DIR "src" MAIN_SRC_NAME "main.cpp" KERNEL_INCLUDE_DIR "kernelinclude" ADDITIONAL_TARGET_LIBS cuda )
