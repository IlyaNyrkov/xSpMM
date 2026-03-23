#pragma once

#ifdef __HIPCC__
    #include <hip/hip_runtime.h>
#else
    #include <cuda_runtime.h>
#endif

namespace xspmm {
namespace kernels {

// -------------------------------------------------------------------------------------------------
// Phase 1: Determine the length (NNZ) of each new row
// -------------------------------------------------------------------------------------------------
template <typename IndexType>
__global__ void permute_row_lengths_kernel(
    IndexType num_rows,
    const IndexType* old_row_ptr,
    const IndexType* perm,
    IndexType* new_row_lengths)
{
    IndexType new_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (new_row < num_rows) {
        IndexType old_row = perm[new_row];
        IndexType start = old_row_ptr[old_row];
        IndexType end = old_row_ptr[old_row + 1];
        new_row_lengths[new_row] = end - start;
    }
}

// -------------------------------------------------------------------------------------------------
// Phase 2: Warp-Coalesced Data Movement
// Maps 1 Warp (32 threads for NV, 64 for AMD) to 1 Row to maximize memory bandwidth.
// -------------------------------------------------------------------------------------------------
template <typename ValueType, typename IndexType, int WARP_SIZE>
__global__ void permute_data_kernel(
    IndexType num_rows,
    const IndexType* old_row_ptr,
    const IndexType* old_col_ind,
    const ValueType* old_values,
    const IndexType* perm,
    const IndexType* new_row_ptr,
    IndexType* new_col_ind,
    ValueType* new_values)
{
    // Calculate which row this specific warp is responsible for
    IndexType warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_rows) return;

    IndexType new_row = warp_id;
    IndexType old_row = perm[new_row];

    IndexType old_start = old_row_ptr[old_row];
    IndexType nnz_in_row = old_row_ptr[old_row + 1] - old_start;

    IndexType new_start = new_row_ptr[new_row];

    // All threads in the warp collaboratively copy the row data
    for (IndexType j = lane_id; j < nnz_in_row; j += WARP_SIZE) {
        new_col_ind[new_start + j] = old_col_ind[old_start + j];
        new_values[new_start + j]  = old_values[old_start + j];
    }
}

template <typename ValueType, typename IndexType>
__global__ void unpermute_dense_kernel(
        IndexType M, IndexType N,
        const IndexType* perm,
        const ValueType* in_matrix,
        ValueType* out_matrix)
    {
        IndexType row = blockIdx.x * blockDim.x + threadIdx.x;
        IndexType col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < M && col < N) {
            IndexType original_row = perm[row];
            // Read from the scrambled temp matrix, write to the correct row in the final matrix
            out_matrix[original_row * N + col] = in_matrix[row * N + col];
        }
    }

    // -------------------------------------------------------------------------------------------------
    // Un-permute a dense matrix (P_inv * C_temp)
    // Maps 1 Warp to 1 Row for perfectly coalesced memory access
    // -------------------------------------------------------------------------------------------------
template <typename ValueType, typename IndexType, int WARP_SIZE>
__global__ void unpermute_dense_kernel(
        IndexType M, IndexType N,
        const IndexType* perm,
        const ValueType* in_matrix,
        ValueType* out_matrix)
{
    IndexType warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id < M) {
        IndexType row = warp_id;
        IndexType original_row = perm[row];

        // The entire warp collaboratively copies the row
        for (IndexType col = lane_id; col < N; col += WARP_SIZE) {
            out_matrix[original_row * N + col] = in_matrix[row * N + col];
        }
    }
}

} // namespace kernels
} // namespace xspmm