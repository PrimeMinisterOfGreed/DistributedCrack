project(Md5Gpu  CUDA CXX)


# enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# enable separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# allow unsupported compiler 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC --allow-unsupported-compiler")
# enable device relocatable device code
set(CMAKE_CUDA_RELOCATABLE_DEVICE_CODE ON)
# allow undefined symbols
set(CMAKE_CUDA_UNIFIED_MEMORY ON)
set(CMAKE_ALLOW_UNDEFINED_SYMBOLS ON)

set(TARGET_NAME md5_gpu)
set(CMAKE_CUDA_ARCHITECTURES 61)
add_library(${TARGET_NAME} SHARED
src/cuda/md5transform.cu
src/cuda/md5bruter.cu
src/md5bruter.c
src/md5transform.c
src/cuda/gpu_string_generator.cu
src/cuda/md5.cu


)
add_library(md5::gpu ALIAS ${TARGET_NAME})
target_include_directories(
    ${TARGET_NAME} PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    )
target_link_libraries(${TARGET_NAME} PRIVATE cuda c )





if(ENABLE_UNIT_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()
