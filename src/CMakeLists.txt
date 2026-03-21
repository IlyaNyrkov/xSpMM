# 1. Define the shared library with hardware-agnostic files
add_library(xspmm SHARED
        xspmm.cpp
        core/conversion.cpp
        core/executor.cpp
        cpu/clustering_sylos_labini.cpp
)

# 2. Add the public headers so users (and the library itself) can find them
target_include_directories(xspmm
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Enable Position Independent Code (required for shared libraries)
set_target_properties(xspmm PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Apply high-performance optimization flags to host code
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xspmm PRIVATE -O3 -ffast-math)
endif()

# ------------------------------------------------------------------------------
# Backend Specific Injections
# ------------------------------------------------------------------------------

if(XSPMM_ENABLE_CUDA)
    # 1. Trigger #ifdef XSPMM_ENABLE_CUDA in xspmm.cpp and executor.cpp
    target_compile_definitions(xspmm PRIVATE XSPMM_ENABLE_CUDA)

    # 2. Inject the CUDA files into the library
    target_sources(xspmm PRIVATE
            cuda/unstructured_mtx_mul.cu
            cuda/mtx_permutation.cu
    )

    # Note: NVIDIA Thrust is header-only and included with NVCC automatically.
endif()

if(XSPMM_ENABLE_HIP)
    # 1. Trigger #ifdef XSPMM_ENABLE_HIP in xspmm.cpp and executor.cpp
    target_compile_definitions(xspmm PRIVATE XSPMM_ENABLE_HIP)

    # 2. Inject the AMD HIP files into the library
    target_sources(xspmm PRIVATE
            hip/unstructured_mtx_mul.hip
            hip/mtx_permutation.hip
    )

    # 3. Find and link rocThrust for the prefix sum in mtx_permutation.hip
    find_package(rocthrust REQUIRED)
    target_link_libraries(xspmm PRIVATE roc::rocthrust)
endif()