include(GoogleTest)
include(${CMAKE_SUPPORT_DIR}/FlagManager.cmake)

function(generate_gtest)
    set(oneValueArgs SRC_DIR TEST_DIR INCLUDE_DIR MAIN_SRC_NAME)
    set(multiValueArgs ADDITIONAL_TARGET_LIBS TEST_INCLUDE_FOLDERS)
    set(options OPTIONAL COVERAGE)
    cmake_parse_arguments(G "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(${G_COVERAGE})
        message("Coverage enabled")


        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            include(${CMAKE_SUPPORT_DIR}/Coverage.cmake)
            setup_target_for_coverage_lcov(NAME ${PROJECT_NAME}_lcov EXECUTABLE ${PROJECT_NAME}_test DEPENDENCIES ${PROJECT_NAME}_test
                BASE_DIRECTORY ./coverage)
        endif()

        message("CURRENT FLAGS" ${CMAKE_CXX_FLAGS})
    endif()

    file(GLOB_RECURSE _TEST_SRC ${G_TEST_DIR}/*.cpp)
    file(GLOB_RECURSE _SRC_MAIN ${G_SRC_DIR}/*.cpp)
    list(FILTER _SRC_MAIN EXCLUDE REGEX ${G_MAIN_SRC_NAME})
    add_executable(${PROJECT_NAME}_test ${_TEST_SRC} ${_SRC_MAIN})

    target_link_libraries(${PROJECT_NAME}_test PRIVATE GTest::gtest GTest::gtest_main ${G_ADDITIONAL_TARGET_LIBS} gcov)
    target_include_directories(${PROJECT_NAME}_test PRIVATE ${G_INCLUDE_DIR})
    gtest_discover_tests(${PROJECT_NAME}_test)
endfunction()

function(generate_gtest_cuda)
    set(oneValueArgs SRC_DIR TEST_DIR INCLUDE_DIR MAIN_SRC_NAME KERNEL_INCLUDE_DIR)
    set(multiValueArgs ADDITIONAL_TARGET_LIBS TEST_INCLUDE_FOLDERS)
    set(options OPTIONAL NONE)
    cmake_parse_arguments(G "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    file(GLOB_RECURSE _TEST_SRC ${G_TEST_DIR}/*.cpp ${G_TEST_DIR}/*.cu)
    file(GLOB_RECURSE _SRC_MAIN ${G_SRC_DIR}/*.cpp ${G_SRC_DIR}/*.cu)
    list(FILTER _SRC_MAIN EXCLUDE REGEX ${G_MAIN_SRC_NAME})
    add_executable(${PROJECT_NAME}_test ${_TEST_SRC} ${_SRC_MAIN})
    target_link_libraries(${PROJECT_NAME}_test PRIVATE GTest::gtest GTest::gtest_main ${G_ADDITIONAL_TARGET_LIBS})
    target_include_directories(${PROJECT_NAME}_test PRIVATE ${G_INCLUDE_DIR} ${G_KERNEL_INCLUDE_DIR} ${G_TEST_INCLUDE_FOLDERS})
    target_compile_options(${PROJECT_NAME}_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -g
        -G
        -pg
        -O0
        -allow-unsupported-compiler
        >)
    gtest_discover_tests(${PROJECT_NAME}_test)
endfunction()
