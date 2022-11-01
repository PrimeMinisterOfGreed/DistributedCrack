project(DistributedCrack)
cmake_minimum_required(VERSION 3.14)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost COMPONENTS ALL REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
include(GoogleTest)

enable_testing()

file(GLOB TEST_SOURCES  CONFIGURE_DEPENDS test/*.cpp src/*.cpp)
file(GLOB SRC_LIST CONFIGURE_DEPENDS  src/*.cpp)

list(REMOVE_ITEM TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)


add_executable(distributedcrack ${SRC_LIST})


#add_library(distributedstcLib SHARED src/Functions.cpp)

#define unit test for non MPI functions
#add_executable(tests ${TEST_SOURCES})
#target_include_directories(tests PUBLIC  ${MPI_INCLUDE_PATH} ./include ${BOOST_INCLUDE_DIRS} gtest)
#target_link_libraries(tests  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${BOOST_LIBRARIES} gtest)

#gtest_discover_tests(tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TEST_LIST TestList)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

target_include_directories(distributedcrack PUBLIC  ${MPI_INCLUDE_PATH} ./include ${Boost_INCLUDE_DIRS})
target_link_libraries(distributedcrack  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${Boost_LIBRARIES} )


