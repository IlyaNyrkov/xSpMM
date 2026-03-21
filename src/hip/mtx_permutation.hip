#include "hip/xspmm.hpp"
#include "../common/permutation.hpp"

// rocThrust is the AMD equivalent of NVIDIA's Thrust
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace xspmm {
namespace hip {

template <typename ValueType, typename IndexType>
CSRMatrix<ValueType, IndexType> apply_permutation(
    std::shared_ptr<const Executor> target_exec,
    const CSRMatrix<ValueType, IndexType>& A,
    const IndexType* perm)
{
    IndexType num_rows = A.get_num_rows();
    IndexType num_cols = A.get_num_cols();
    IndexType nnz = A.get_nnz();

    if (num_rows == 0) return CSRMatrix<ValueType, IndexType>(target_exec, 0, 0, 0);

    // 1. Allocate a temporary buffer for row lengths on the device
    IndexType* d_row_lengths = nullptr;
    target_exec->allocate(reinterpret_cast<void**>(&d_row_lengths), num_rows * sizeof(IndexType));

    // 2. Launch Lengths Kernel
    int threads = 256;
    int blocks = (num_rows + threads - 1) / threads;
    kernels::permute_row_lengths_kernel<<<blocks, threads, 0, 0>>>( // 0 shared mem, default stream
        num_rows, A.get_row_ptr(), perm, d_row_lengths
    );

    // 3. Create the target Matrix (this automatically allocates new_row_ptr, new_col_ind, and new_values)
    CSRMatrix<ValueType, IndexType> permuted_mat(target_exec, num_rows, num_cols, nnz);

    // 4. Compute Exclusive Scan using rocThrust to generate new_row_ptr
    thrust::device_ptr<IndexType> dev_lengths(d_row_lengths);
    thrust::device_ptr<IndexType> dev_row_ptr(permuted_mat.get_row_ptr());

    // This executes on the GPU
    thrust::exclusive_scan(thrust::device, dev_lengths, dev_lengths + num_rows, dev_row_ptr);

    // Set the very last element of row_ptr to total NNZ
    target_exec->copy_from_host(permuted_mat.get_row_ptr() + num_rows, &nnz, sizeof(IndexType));

    // 5. Launch Data Movement Kernel
    // KEY DIFFERENCE: AMD uses 64 threads per wavefront!
    const int WARP_SIZE = 64;
    int total_warps = num_rows;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int blocks_data = (total_warps + warps_per_block - 1) / warps_per_block;

    kernels::permute_data_kernel<ValueType, IndexType, WARP_SIZE><<<blocks_data, threads_per_block, 0, 0>>>(
        num_rows,
        A.get_row_ptr(), A.get_col_ind(), A.get_values(),
        perm,
        permuted_mat.get_row_ptr(), permuted_mat.get_col_ind(), permuted_mat.get_values()
    );

    // 6. Cleanup temporary memory
    target_exec->free(d_row_lengths);

    return permuted_mat;
}

// Explicit Instantiations
template CSRMatrix<float, int32_t> apply_permutation<float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, const int32_t*);
template CSRMatrix<float, int64_t> apply_permutation<float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, const int64_t*);

// (Add double or __half instantiations here as needed)

} // namespace hip
} // namespace xspmm