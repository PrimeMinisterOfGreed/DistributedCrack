find_package(GTest REQUIRED CONFIG)
set(UT_TARGET_NAME md5_ut)

file(GLOB_RECURSE UT_SRC_FILES *.c *.cpp *.cu)
add_executable(${UT_TARGET_NAME} ${UT_SRC_FILES} ${SRC_FILES})
target_include_directories(${UT_TARGET_NAME} PRIVATE ../src ../include)
target_link_libraries(${UT_TARGET_NAME} PRIVATE GTest::gtest GTest::gtest_main md5::md5)