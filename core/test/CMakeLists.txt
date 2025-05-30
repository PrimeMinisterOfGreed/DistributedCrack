
find_package(GTest REQUIRED CONFIG)
set(UT_TARGET_NAME mpc_ut)

#filter main from SRC_FILES
list(FILTER SRC_FILES EXCLUDE REGEX "main.cpp")

set(TEST_SRC_FILES
    test_main.cpp
    test_promises.cpp
    test_routines.cpp
    ${SRC_FILES}
)

add_executable(${UT_TARGET_NAME} ${TEST_SRC_FILES})
target_include_directories(${UT_TARGET_NAME} PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/../src 
${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(${UT_TARGET_NAME} PRIVATE GTest::gtest GTest::gtest_main md5::md5 
    OpenMP::OpenMP_CXX  MPI::MPI_C MPI::MPI_CXX pthread
) 

if(ENABLE_CUDA)
target_link_libraries(${UT_TARGET_NAME} PRIVATE md5::gpu cuda nvidia-ml cudart)
target_compile_definitions(${UT_TARGET_NAME} PRIVATE ENABLE_CUDA)
endif()

#copy launch.toml to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/launch.toml
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILES_MATCHING PATTERN "*.toml"
)

#copy launch.toml to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/dictionary.txt
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILES_MATCHING PATTERN "*.txt"
)
