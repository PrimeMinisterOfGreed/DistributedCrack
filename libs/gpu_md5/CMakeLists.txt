project(Md5Gpu LANGUAGES CUDA CXX)

set(TARGET_NAME md5_gpu)
enable_language(CUDA)
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CUDA_ARCHITECTURES 75)
add_library(${TARGET_NAME} SHARED ${SRC_FILES})
add_executable(${TARGET_NAME}_sample ${SRC_FILES} trial_main.cu)
add_library(md5::gpu ALIAS ${TARGET_NAME})
target_include_directories(
    ${TARGET_NAME} PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${TARGET_NAME} PRIVATE cuda)


target_include_directories(
    ${TARGET_NAME}_sample PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include src)
target_link_libraries(${TARGET_NAME}_sample PRIVATE cuda)


if(ENABLE_UNIT_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()
