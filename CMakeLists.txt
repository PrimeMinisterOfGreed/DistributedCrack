project(DistributedCrack)
cmake_minimum_required(VERSION 3.14)

include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)

    if(flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)

        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()

        unset(flag_supported CACHE)
    endif()
endfunction()

function(force_flag flag)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
message("Forced flag " ${flag})
endfunction()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/)
find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost COMPONENTS mpi program_options REQUIRED)
find_package(GTest REQUIRED)

add_subdirectory("GPU")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
include(GoogleTest)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp src/*.cpp)
file(GLOB_RECURSE SRC_LIST CONFIGURE_DEPENDS src/*.cpp)

list(REMOVE_ITEM TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

add_executable(distributedcrack ${SRC_LIST})


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    force_flag("--coverage")
    force_flag("-fprofile-abs-path")
    include(./cmake/Coverage.cmake)
    setup_target_for_coverage_lcov(NAME distcrack_lcov EXECUTABLE tests DEPENDENCIES tests BASE_DIRECTORY ./coverage)

endif() 

# define unit test for non MPI functions
add_executable(tests ${TEST_SOURCES})
target_include_directories(tests PUBLIC ${MPI_INCLUDE_PATH} ./include ${Boost_INCLUDE_DIRS})
target_link_libraries(tests PUBLIC fmt::fmt ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} GTest::gtest_main PRIVATE DistributedCrack.GPU)
gtest_discover_tests(tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TEST_LIST TestList)


target_include_directories(distributedcrack PUBLIC ${MPI_INCLUDE_PATH} ./include ${Boost_INCLUDE_DIRS})
target_link_libraries(distributedcrack PUBLIC fmt::fmt ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} DistributedCrack.GPU)
