#pragma once

#ifdef __HIPCC__
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
    #include <rocwmma/rocwmma.hpp>
    namespace hw_wmma = rocwmma;
    constexpr int HW_WARP_SIZE = 64;
#else
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <mma.h>
    namespace hw_wmma = nvcuda::wmma;
    constexpr int HW_WARP_SIZE = 32;
#endif

namespace xspmm {
namespace kernels {

template <
    typename InputType,
    typename OutputType,
    typename IndexType,
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int WARPS_PER_BLOCK,
    bool IS_CLEAN_MULTIPLE> // <-- NEW: Compile-time specialization
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
    int warp_id = threadIdx.x / HW_WARP_SIZE;
    int lane_id = threadIdx.x % HW_WARP_SIZE;

    IndexType block_row_A = blockIdx.y;
    IndexType base_block_col_B = blockIdx.x * WARPS_PER_BLOCK;
    IndexType my_block_col_B = base_block_col_B + warp_id;

    bool active_warp = (block_row_A < num_block_rows_a) && (my_block_col_B < num_block_cols_b);

    __shared__ InputType shared_A[BLOCK_M * BLOCK_K];
    __shared__ InputType shared_B[8 * BLOCK_K * BLOCK_N];
    __shared__ OutputType shared_C[8 * BLOCK_M * BLOCK_N];

    hw_wmma::fragment<hw_wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, InputType, hw_wmma::row_major> fragA;
    hw_wmma::fragment<hw_wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, InputType, hw_wmma::row_major> fragB;
    hw_wmma::fragment<hw_wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, OutputType> fragC;

    if (active_warp) {
        hw_wmma::fill_fragment(fragC, static_cast<OutputType>(0));
    }

    IndexType start_idx = bcsr_row_ptr[block_row_A];
    IndexType end_idx = bcsr_row_ptr[block_row_A + 1];

    constexpr int elements_per_thread = (BLOCK_M * BLOCK_K) / HW_WARP_SIZE;

    // Compile-time resolution for Matrix A loads across different block sizes
    constexpr int threads_in_block = WARPS_PER_BLOCK * HW_WARP_SIZE;
    constexpr int a_loads_per_thread = (BLOCK_M * BLOCK_K) / threads_in_block;

    for (IndexType i = start_idx; i < end_idx; ++i) {
        IndexType col_A = bcsr_col_ind[i];

        // 1. PERFECT MATRIX A LOAD: No branches, no dynamic loops
        #pragma unroll
        for (int e = 0; e < a_loads_per_thread; ++e) {
            int idx = threadIdx.x + (e * threads_in_block);
            shared_A[idx] = bcsr_values[(i * BLOCK_M * BLOCK_K) + idx];
        }

        // 2. DUAL-PATH MATRIX B LOAD
        if (active_warp) {
            const InputType* ptr_B = B + (col_A * BLOCK_K * ldb) + (my_block_col_B * BLOCK_N);
            InputType* my_shared_B = &shared_B[warp_id * BLOCK_K * BLOCK_N];

            #pragma unroll
            for(int elem = 0; elem < elements_per_thread; ++elem) {
                int flat_idx = lane_id + (elem * HW_WARP_SIZE);
                int row = flat_idx / BLOCK_N;
                int col = flat_idx % BLOCK_N;

                if constexpr (IS_CLEAN_MULTIPLE) {
                    // FAST PATH: Zero inner-loop branch overhead
                    my_shared_B[flat_idx] = ptr_B[row * ldb + col];
                } else {
                    // SAFE PATH: Bounds checking for jagged edges
                    if ((my_block_col_B * BLOCK_N + col) < ldb) {
                        my_shared_B[flat_idx] = ptr_B[row * ldb + col];
                    } else {
                        my_shared_B[flat_idx] = static_cast<InputType>(0);
                    }
                }
            }
        }

        __syncthreads();

        // 3. TENSOR CORE MATH
        if (active_warp) {
            InputType* my_shared_B = &shared_B[warp_id * BLOCK_K * BLOCK_N];
            hw_wmma::load_matrix_sync(fragA, shared_A, BLOCK_K);
            hw_wmma::load_matrix_sync(fragB, my_shared_B, BLOCK_N);
            hw_wmma::mma_sync(fragC, fragA, fragB, fragC);
        }

        __syncthreads();
    }

    // 4. DUAL-PATH MATRIX C STORE
    if (active_warp) {
        OutputType* ptr_C = C + (block_row_A * BLOCK_M * ldc) + (my_block_col_B * BLOCK_N);

        if constexpr (IS_CLEAN_MULTIPLE) {
            // FAST PATH
            hw_wmma::store_matrix_sync(ptr_C, fragC, ldc, hw_wmma::mem_row_major);
        } else {
            // SAFE PATH
            if ((my_block_col_B * BLOCK_N + BLOCK_N) <= ldc) {
                hw_wmma::store_matrix_sync(ptr_C, fragC, ldc, hw_wmma::mem_row_major);
            } else {
                OutputType* my_shared_C = &shared_C[warp_id * BLOCK_M * BLOCK_N];
                hw_wmma::store_matrix_sync(my_shared_C, fragC, BLOCK_N, hw_wmma::mem_row_major);

                int valid_cols = ldc - (my_block_col_B * BLOCK_N);
                constexpr int c_elements_per_thread = (BLOCK_M * BLOCK_N) / HW_WARP_SIZE;

                #pragma unroll
                for(int elem = 0; elem < c_elements_per_thread; ++elem) {
                    int flat_idx = lane_id + (elem * HW_WARP_SIZE);
                    int row = flat_idx / BLOCK_N;
                    int col = flat_idx % BLOCK_N;
                    if (col < valid_cols) {
                        ptr_C[row * ldc + col] = my_shared_C[flat_idx];
                    }
                }
            }
        }
    }
}

} // namespace kernels
} // namespace xspmm