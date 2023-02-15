project(DistributedCrack)
cmake_minimum_required(VERSION 3.14)

include(CheckCXXCompilerFlag)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/)
find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost COMPONENTS mpi program_options REQUIRED)
find_package(GTest REQUIRED)


get_filename_component(_CWD ./ ABSOLUTE)
set(CMAKE_SUPPORT_DIR ${_CWD}/cmake)
add_subdirectory("./libs/GPU")
add_subdirectory("./libs/NESLib")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(./cmake/FlagManager.cmake)
include(./cmake/TestProjectGenerator.cmake)

include(GoogleTest)

enable_testing()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      force_flag("--coverage")
      force_flag("-fprofile-abs-path")
endif()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp src/*.cpp)
file(GLOB_RECURSE SRC_LIST CONFIGURE_DEPENDS src/*.cpp)

list(REMOVE_ITEM TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

add_executable(distributedcrack ${SRC_LIST})

target_include_directories(distributedcrack PUBLIC ${MPI_INCLUDE_PATH} ./include ${Boost_INCLUDE_DIRS})
target_link_libraries(distributedcrack PUBLIC fmt::fmt ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} DistributedCrack.GPU)

generate_gtest(COVERAGE TEST_DIR "test" INCLUDE_DIR "include" SRC_DIR "src" MAIN_SRC_NAME "main.cpp" ADDITIONAL_TARGET_LIBS fmt::fmt ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} DistributedCrack.GPU)