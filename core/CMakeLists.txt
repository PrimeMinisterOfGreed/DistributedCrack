
file(GLOB_RECURSE
    SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(mpc ${SRC_FILES})
target_include_directories(mpc
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
target_compile_options(mpc PRIVATE -fno-exceptions -fno-rtti 
)

target_link_libraries(mpc
    PUBLIC
    md5::md5
    md5::gpu
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    MPI::MPI_C
    pthread
)

add_subdirectory(test)