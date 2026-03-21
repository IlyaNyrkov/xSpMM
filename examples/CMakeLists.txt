# Create the executable
add_executable(01_uspmm_checkers 01_uspmm_checkers.cpp)

# Link the core library and the utilities (for generation/comparison)
target_link_libraries(01_uspmm_checkers PRIVATE xspmm xspmm_utils)

# Pass the hardware flags to the example so it knows which Executor to instantiate
if(XSPMM_ENABLE_CUDA)
    target_compile_definitions(01_uspmm_checkers PRIVATE XSPMM_ENABLE_CUDA)
endif()

if(XSPMM_ENABLE_HIP)
    target_compile_definitions(01_uspmm_checkers PRIVATE XSPMM_ENABLE_HIP)
endif()