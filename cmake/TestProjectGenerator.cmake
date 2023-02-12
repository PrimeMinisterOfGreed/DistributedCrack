
include(GoogleTest)
function(generate_gtest)
      set(oneValueArgs SRC_DIR TEST_DIR INCLUDE_DIR MAIN_SRC_NAME)
      set(multiValueArgs ADDITIONAL_TARGET_LIBS)
      set(options OPTIONAL NONE)
      cmake_parse_arguments(G "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
      
      file(GLOB_RECURSE _TEST_SRC ${G_TEST_DIR}/*.cpp)
  file(GLOB_RECURSE _SRC_MAIN ${G_SRC_DIR}/*.cpp)
  message("Imported source files: " + ${_SRC_MAIN})
  list(FILTER _SRC_MAIN EXCLUDE REGEX ${G_MAIN_SRC_NAME})
  add_executable(${PROJECT_NAME}_test ${_TEST_SRC} ${_SRC_MAIN})
  target_link_libraries(${PROJECT_NAME}_test PRIVATE GTest::gtest GTest::gtest_main ${G_ADDITIONAL_TARGET_LIBS})
  target_include_directories(${PROJECT_NAME}_test PRIVATE ${G_INCLUDE_DIR})
  gtest_discover_tests(${PROJECT_NAME}_test)
endfunction()



function(generate_gtest_cuda)
    set(oneValueArgs SRC_DIR TEST_DIR INCLUDE_DIR MAIN_SRC_NAME KERNEL_INCLUDE_DIR)
    set(multiValueArgs ADDITIONAL_TARGET_LIBS)
    set(options OPTIONAL NONE)
    cmake_parse_arguments(G "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    file(GLOB_RECURSE _TEST_SRC ${G_TEST_DIR}/*.cpp ${G_TEST_DIR}/*.cu)
    file(GLOB_RECURSE _SRC_MAIN ${G_SRC_DIR}/*.cpp ${G_SRC_DIR}/*.cu)
    message("Imported source files: " + ${_SRC_MAIN})
    list(FILTER _SRC_MAIN EXCLUDE REGEX ${G_MAIN_SRC_NAME})
    add_executable(${PROJECT_NAME}_test ${_TEST_SRC} ${_SRC_MAIN})
    target_link_libraries(${PROJECT_NAME}_test PRIVATE GTest::gtest GTest::gtest_main ${G_ADDITIONAL_TARGET_LIBS})
    target_include_directories(${PROJECT_NAME}_test PRIVATE ${G_INCLUDE_DIR} ${G_KERNEL_INCLUDE_DIR})
    target_compile_options(${PROJECT_NAME}_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -g
            -G
            -pg
            -O0
            -allow-unsupported-compiler
            >)
    gtest_discover_tests(${PROJECT_NAME}_test)
endfunction()
