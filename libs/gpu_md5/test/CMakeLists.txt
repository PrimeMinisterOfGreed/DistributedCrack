find_package(GTest REQUIRED CONFIG)
set(UT_TARGET_NAME md5_gpu_ut)

add_executable(${UT_TARGET_NAME} test_main.cpp ${SRC_FILES})
target_include_directories(${UT_TARGET_NAME} PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/../src 
${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(${UT_TARGET_NAME} PRIVATE GTest::gtest GTest::gtest_main nvidia-ml cuda md5::gpu md5::md5)

add_executable(${TARGET_NAME}_istrumented istrumented_test.cu ${SRC_FILES})
target_link_libraries(${TARGET_NAME}_istrumented PRIVATE cuda nvidia-ml md5::gpu)
target_include_directories(${TARGET_NAME}_istrumented PUBLIC 
../src 
../include
)