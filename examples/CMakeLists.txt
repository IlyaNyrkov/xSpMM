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

add_executable(02_spmm_benchmark 02_spmm_benchmark.cpp)
target_link_libraries(02_spmm_benchmark PRIVATE xspmm xspmm_utils)

if(XSPMM_ENABLE_CUDA)
    target_compile_definitions(02_spmm_benchmark PRIVATE XSPMM_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Link cuSPARSE and cuBLAS
    target_link_libraries(02_spmm_benchmark PRIVATE CUDA::cusparse CUDA::cublas)
endif()

if(XSPMM_ENABLE_HIP)
    target_compile_definitions(02_spmm_benchmark PRIVATE XSPMM_ENABLE_HIP)
    find_package(rocsparse REQUIRED)
    find_package(rocblas REQUIRED)
    # Link rocSPARSE and rocBLAS
    target_link_libraries(02_spmm_benchmark PRIVATE roc::rocsparse roc::rocblas)
endif()

add_executable(03_spmm_scaling_benchmark 03_spmm_scaling_benchmark.cpp)
target_link_libraries(03_spmm_scaling_benchmark PRIVATE xspmm xspmm_utils)

if(XSPMM_ENABLE_CUDA)
    target_compile_definitions(03_spmm_scaling_benchmark PRIVATE XSPMM_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Link cuSPARSE and cuBLAS
    target_link_libraries(03_spmm_scaling_benchmark PRIVATE CUDA::cusparse CUDA::cublas)
endif()

if(XSPMM_ENABLE_HIP)
    target_compile_definitions(03_spmm_scaling_benchmark PRIVATE XSPMM_ENABLE_HIP)
    find_package(rocsparse REQUIRED)
    find_package(rocblas REQUIRED)
    # Link rocSPARSE and rocBLAS
    target_link_libraries(03_spmm_scaling_benchmark PRIVATE roc::rocsparse roc::rocblas)
endif()

add_executable(04_real_matrices_benchmark 04_real_matrices_benchmark.cpp)
target_link_libraries(04_real_matrices_benchmark PRIVATE xspmm xspmm_utils)

if(XSPMM_ENABLE_CUDA)
    target_compile_definitions(04_real_matrices_benchmark PRIVATE XSPMM_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Link cuSPARSE and cuBLAS
    target_link_libraries(04_real_matrices_benchmark PRIVATE CUDA::cusparse CUDA::cublas)
endif()

if(XSPMM_ENABLE_HIP)
    target_compile_definitions(04_real_matrices_benchmark PRIVATE XSPMM_ENABLE_HIP)
    find_package(rocsparse REQUIRED)
    find_package(rocblas REQUIRED)
    # Link rocSPARSE and rocBLAS
    target_link_libraries(04_real_matrices_benchmark PRIVATE roc::rocsparse roc::rocblas)
endif()

add_executable(05_core_spmm_benchmark 05_core_spmm_benchmark.cpp)
target_link_libraries(05_core_spmm_benchmark PRIVATE xspmm xspmm_utils)

if(XSPMM_ENABLE_CUDA)
    target_compile_definitions(05_core_spmm_benchmark PRIVATE XSPMM_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Link cuSPARSE and cuBLAS
    target_link_libraries(05_core_spmm_benchmark PRIVATE CUDA::cusparse CUDA::cublas)
endif()

if(XSPMM_ENABLE_HIP)
    target_compile_definitions(05_core_spmm_benchmark PRIVATE XSPMM_ENABLE_HIP)
    find_package(rocsparse REQUIRED)
    find_package(rocblas REQUIRED)
    # Link rocSPARSE and rocBLAS
    target_link_libraries(05_core_spmm_benchmark PRIVATE roc::rocsparse roc::rocblas)
endif()