Project(Md5 C )

set(TARGET_NAME md5)
file(GLOB_RECURSE SRC_FILES src/*.c)


add_library(${TARGET_NAME} STATIC ${SRC_FILES})
add_library(md5::md5 ALIAS ${TARGET_NAME})
target_include_directories(${TARGET_NAME} PUBLIC include)
#enable fpic
set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON )
if(ENABLE_UNIT_TEST)
    add_subdirectory(test)
endif()