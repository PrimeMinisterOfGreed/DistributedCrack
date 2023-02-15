project(NESLib)

file(GLOB_RECURSE PRIV_SRC src/*.cpp)

include(${CMAKE_SUPPORT_DIR}/TestProjectGenerator.cmake)

add_library(NESLib SHARED ${PRIV_SRC})
target_include_directories(NESLib PUBLIC include)
target_link_libraries(NESLib PRIVATE  fmt::fmt)
generate_gtest(TEST_DIR "test" INCLUDE_DIR "include" SRC_DIR "src" MAIN_SRC_NAME "main.cpp" ADDITIONAL_TARGET_LIBS fmt::fmt "NESLib" )