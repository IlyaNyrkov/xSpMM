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
    IndexType ldc,
    int warps_per_block) // <-- New parameter
{
    // Identify who this thread is and which warp it belongs to
    int warp_id = threadIdx.x / HW_WARP_SIZE;
    int lane_id = threadIdx.x % HW_WARP_SIZE;

    IndexType block_row_A = blockIdx.y;

    // Calculate the base column of B for this entire Thread Block
    IndexType base_block_col_B = blockIdx.x * warps_per_block;
    // Calculate the specific column of B for THIS specific warp
    IndexType my_block_col_B = base_block_col_B + warp_id;

    // Determine if this specific warp actually has work to do (bounds check)
    bool active_warp = (block_row_A < num_block_rows_a) && (my_block_col_B < num_block_cols_b);

    // Shared Memory Allocation
    // Matrix A is shared among ALL warps in the block
    __shared__ InputType shared_A[BLOCK_M * BLOCK_K];
    // Matrix B needs space for every warp's individual 16x16 tile
    // 8 is the maximum warps we expect (256 threads / 32 NV warp size)
    __shared__ InputType shared_B[8 * BLOCK_K * BLOCK_N];

    hw_wmma::fragment<hw_wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, InputType, hw_wmma::row_major> fragA;
    hw_wmma::fragment<hw_wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, InputType, hw_wmma::row_major> fragB;
    hw_wmma::fragment<hw_wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, OutputType> fragC;

    if (active_warp) {
        hw_wmma::fill_fragment(fragC, static_cast<OutputType>(0));
    }

    IndexType start_idx = bcsr_row_ptr[block_row_A];
    IndexType end_idx = bcsr_row_ptr[block_row_A + 1];

    constexpr int elements_per_thread = (BLOCK_M * BLOCK_K) / HW_WARP_SIZE;

    for (IndexType i = start_idx; i < end_idx; ++i) {
        IndexType col_A = bcsr_col_ind[i];

        // -----------------------------------------------------------
        // 1. COLLABORATIVE LOAD OF MATRIX A
        // -----------------------------------------------------------
        // Since A is 16x16 (256 elements) and we have 256 threads,
        // each thread loads exactly 1 element! Perfect memory coalescing.
        if (threadIdx.x < (BLOCK_M * BLOCK_K)) {
            const InputType* ptr_A = bcsr_values + (i * BLOCK_M * BLOCK_K);
            shared_A[threadIdx.x] = ptr_A[threadIdx.x];
        }

        // -----------------------------------------------------------
        // 2. INDEPENDENT LOAD OF MATRIX B (Per-Warp)
        // -----------------------------------------------------------
        if (active_warp) {
            const InputType* ptr_B = B + (col_A * BLOCK_K * ldb) + (my_block_col_B * BLOCK_N);
            InputType* my_shared_B = &shared_B[warp_id * BLOCK_K * BLOCK_N];

            #pragma unroll
            for(int elem = 0; elem < elements_per_thread; ++elem) {
                int flat_idx = lane_id + (elem * HW_WARP_SIZE);
                int row = flat_idx / BLOCK_N;
                int col = flat_idx % BLOCK_N;
                my_shared_B[flat_idx] = ptr_B[row * ldb + col];
            }
        }

        // Wait for all threads to finish loading A and B
        __syncthreads();

        // -----------------------------------------------------------
        // 3. TENSOR CORE MATH
        // -----------------------------------------------------------
        if (active_warp) {
            InputType* my_shared_B = &shared_B[warp_id * BLOCK_K * BLOCK_N];

            // All active warps read from the SAME shared_A!
            hw_wmma::load_matrix_sync(fragA, shared_A, BLOCK_K);
            hw_wmma::load_matrix_sync(fragB, my_shared_B, BLOCK_N);
            hw_wmma::mma_sync(fragC, fragA, fragB, fragC);
        }

        // Prevent threads from overwriting shared memory before math is done
        __syncthreads();
    }

    // 4. Store Result
    if (active_warp) {
        OutputType* ptr_C = C + (block_row_A * BLOCK_M * ldc) + (my_block_col_B * BLOCK_N);
        hw_wmma::store_matrix_sync(ptr_C, fragC, ldc, hw_wmma::mem_row_major);
    }
}

} // namespace kernels
} // namespace xspmm
