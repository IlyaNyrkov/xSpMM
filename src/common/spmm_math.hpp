#pragma once

// -----------------------------------------------------------------------------
// Hardware Abstraction Layer
// -----------------------------------------------------------------------------
#ifdef __HIPCC__
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
    #include <rocwmma/rocwmma.hpp>
    namespace hw_wmma = rocwmma;           // Alias AMD's API
    constexpr int HW_WARP_SIZE = 64;       // AMD Wavefront
#else
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <mma.h>
    namespace hw_wmma = nvcuda::wmma;      // Alias NVIDIA's API
    constexpr int HW_WARP_SIZE = 32;       // NVIDIA Warp
#endif

namespace xspmm {
namespace kernels {

template <
    typename InputType,
    typename OutputType,
    typename IndexType,
    int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void bcsr_spmm_core_kernel(
    const IndexType* bcsr_row_ptr,
    const IndexType* bcsr_col_ind,
    const InputType* bcsr_values,
    const InputType* B,
    OutputType* C,
    IndexType num_block_rows_a,
    IndexType num_block_cols_b,
    IndexType ldb,
    IndexType ldc)
{
    int lane_id = threadIdx.x;
    IndexType block_row_A = blockIdx.y;
    IndexType block_col_B = blockIdx.x;

    if (block_row_A >= num_block_rows_a || block_col_B >= num_block_cols_b) return;

    __shared__ InputType shared_A[BLOCK_M * BLOCK_K];
    __shared__ InputType shared_B[BLOCK_K * BLOCK_N];

    // Use the namespace alias 'hw_wmma' instead of 'nvcuda::wmma' or 'rocwmma'
    hw_wmma::fragment<hw_wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, InputType, hw_wmma::row_major> fragA;
    hw_wmma::fragment<hw_wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, InputType, hw_wmma::row_major> fragB;
    hw_wmma::fragment<hw_wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, OutputType> fragC;

    hw_wmma::fill_fragment(fragC, static_cast<OutputType>(0));

    IndexType start_idx = bcsr_row_ptr[block_row_A];
    IndexType end_idx = bcsr_row_ptr[block_row_A + 1];

    // Calculate loops based on the compile-time HW_WARP_SIZE
    constexpr int elements_per_thread = (BLOCK_M * BLOCK_K) / HW_WARP_SIZE;

    for (IndexType i = start_idx; i < end_idx; ++i) {
        IndexType col_A = bcsr_col_ind[i];

        const InputType* ptr_A = bcsr_values + (i * BLOCK_M * BLOCK_K);
        const InputType* ptr_B = B + (col_A * BLOCK_K * ldb) + (block_col_B * BLOCK_N);

        #pragma unroll
        for(int elem = 0; elem < elements_per_thread; ++elem) {
            int flat_idx = lane_id + (elem * HW_WARP_SIZE);
            int row = flat_idx / BLOCK_K;
            int col = flat_idx % BLOCK_K;

            shared_A[flat_idx] = ptr_A[flat_idx];
            shared_B[flat_idx] = ptr_B[row * ldb + col];
        }

        __syncthreads();

        hw_wmma::load_matrix_sync(fragA, shared_A, BLOCK_K);
        hw_wmma::load_matrix_sync(fragB, shared_B, BLOCK_N);
        hw_wmma::mma_sync(fragC, fragA, fragB, fragC);

        __syncthreads();
    }

    OutputType* ptr_C = C + (block_row_A * BLOCK_M * ldc) + (block_col_B * BLOCK_N);
    hw_wmma::store_matrix_sync(ptr_C, fragC, ldc, hw_wmma::mem_row_major);
}

} // namespace kernels
} // namespace xspmm