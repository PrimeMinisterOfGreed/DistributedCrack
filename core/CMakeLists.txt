# Enable ExternalProject CMake module

set(RUST_MODULE_FLAGS
--target-dir=${CMAKE_CURRENT_BINARY_DIR} 
--manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml  
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
set(TARGET debug)
else()
set(TARGET release)
list(APPEND RUST_MODULE_FLAGS --release )
endif()




add_custom_target(mpc   
    COMMAND cargo build ${RUST_MODULE_FLAGS})






