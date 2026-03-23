add_executable(01_uspmm_checkers 01_uspmm_checkers.cpp)
target_link_libraries(01_uspmm_checkers PRIVATE xspmm xspmm_utils)

if(XSPMM_ENABLE_CUDA)
    target_compile_definitions(01_uspmm_checkers PRIVATE XSPMM_ENABLE_CUDA)

    # Find and link cuSPARSE
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(01_uspmm_checkers PRIVATE CUDA::cusparse)
endif()

if(XSPMM_ENABLE_HIP)
    target_compile_definitions(01_uspmm_checkers PRIVATE XSPMM_ENABLE_HIP)

    # Find and link rocSPARSE
    find_package(rocsparse REQUIRED)
    target_link_libraries(01_uspmm_checkers PRIVATE roc::rocsparse)
endif()