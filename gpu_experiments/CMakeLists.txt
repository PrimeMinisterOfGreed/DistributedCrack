project(GpuExperience C CXX CUDA)
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
add_executable(exp ${SRC_FILES})
target_link_libraries(exp PUBLIC cuda md5::md5)