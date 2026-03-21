#include "xspmm/xspmm.hpp"
#include "xspmm/clustering.hpp"
#include "xspmm/core/executor.hpp"
#include <stdexcept>

// Conditionally include the bridge headers based on CMake flags
#ifdef XSPMM_ENABLE_CUDA
#include "cuda/xspmm.hpp"
#endif

#ifdef XSPMM_ENABLE_HIP
#include "hip/xspmm.hpp"
#endif

namespace xspmm {

// Forward declarations for CPU fallback implementations
// (These would be implemented in src/cpu/...)
namespace cpu {
    template <typename ValueType, typename IndexType>
    void spmm(std::shared_ptr<const Executor> exec, const BCSRMatrix<ValueType, IndexType>& A, const ValueType* B, ValueType* C, IndexType N) {
        throw std::runtime_error("CPU SpMM fallback is not yet implemented.");
    }

    template <typename ValueType, typename IndexType>
    void compute_1d_jaccard_clustering(std::shared_ptr<const Executor> exec, const CSRMatrix<ValueType, IndexType>& A, float dist_thresh, IndexType block_width, IndexType* perm_out) {
        throw std::runtime_error("CPU Clustering is not yet implemented.");
    }

    template <typename ValueType, typename IndexType>
    CSRMatrix<ValueType, IndexType> apply_permutation(std::shared_ptr<const Executor> target_exec, const CSRMatrix<ValueType, IndexType>& A, const IndexType* perm) {
        throw std::runtime_error("CPU Permutation is not yet implemented.");
    }
}

// =================================================================================================
// 1. SpMM Dispatcher
// =================================================================================================
template <typename ValueType, typename IndexType>
void spmm(const BCSRMatrix<ValueType, IndexType>& A,
          const ValueType* B,
          ValueType* C,
          IndexType N)
{
    auto exec = A.get_executor();

#ifdef XSPMM_ENABLE_CUDA
    if (auto cuda_exec = dynamic_cast<const CudaExecutor*>(exec.get())) {
        cuda::spmm(exec, A, B, C, N);
        return;
    }
#endif

#ifdef XSPMM_ENABLE_HIP
    if (auto hip_exec = dynamic_cast<const HipExecutor*>(exec.get())) {
        hip::spmm(exec, A, B, C, N);
        return;
    }
#endif

    if (auto cpu_exec = dynamic_cast<const CpuExecutor*>(exec.get())) {
        cpu::spmm(exec, A, B, C, N);
        return;
    }

    throw std::runtime_error("xSpMM: Unsupported Executor type in spmm().");
}

// =================================================================================================
// 2. Clustering Dispatcher
// =================================================================================================
template <typename ValueType, typename IndexType>
void compute_1d_jaccard_clustering(
    std::shared_ptr<const Executor> exec,
    const CSRMatrix<ValueType, IndexType>& A,
    float dist_thresh,
    IndexType block_width,
    IndexType* perm_out)
{
#ifdef XSPMM_ENABLE_CUDA
    if (auto cuda_exec = dynamic_cast<const CudaExecutor*>(exec.get())) {
        cuda::compute_1d_jaccard_clustering(exec, A, dist_thresh, block_width, perm_out);
        return;
    }
#endif

#ifdef XSPMM_ENABLE_HIP
    if (auto hip_exec = dynamic_cast<const HipExecutor*>(exec.get())) {
        hip::compute_1d_jaccard_clustering(exec, A, dist_thresh, block_width, perm_out);
        return;
    }
#endif

    if (auto cpu_exec = dynamic_cast<const CpuExecutor*>(exec.get())) {
        cpu::compute_1d_jaccard_clustering(exec, A, dist_thresh, block_width, perm_out);
        return;
    }

    throw std::runtime_error("xSpMM: Unsupported Executor type in clustering.");
}

// =================================================================================================
// 3. Permutation Dispatcher
// =================================================================================================
template <typename ValueType, typename IndexType>
CSRMatrix<ValueType, IndexType> apply_permutation(
    std::shared_ptr<const Executor> target_exec,
    const CSRMatrix<ValueType, IndexType>& A,
    const IndexType* perm)
{
#ifdef XSPMM_ENABLE_CUDA
    if (auto cuda_exec = dynamic_cast<const CudaExecutor*>(target_exec.get())) {
        return cuda::apply_permutation(target_exec, A, perm);
    }
#endif

#ifdef XSPMM_ENABLE_HIP
    if (auto hip_exec = dynamic_cast<const HipExecutor*>(target_exec.get())) {
        return hip::apply_permutation(target_exec, A, perm);
    }
#endif

    if (auto cpu_exec = dynamic_cast<const CpuExecutor*>(target_exec.get())) {
        return cpu::apply_permutation(target_exec, A, perm);
    }

    throw std::runtime_error("xSpMM: Unsupported Executor type in apply_permutation.");
}

// =================================================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// Forces the host compiler to generate these specific symbol variations into libxspmm.so
// =================================================================================================

// For 32-bit indices (High Performance)
template void spmm<float, int32_t>(const BCSRMatrix<float, int32_t>&, const float*, float*, int32_t);
template void compute_1d_jaccard_clustering<float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, float, int32_t, int32_t*);
template CSRMatrix<float, int32_t> apply_permutation<float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, const int32_t*);

// For 64-bit indices (Large Datasets)
template void spmm<float, int64_t>(const BCSRMatrix<float, int64_t>&, const float*, float*, int64_t);
template void compute_1d_jaccard_clustering<float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, float, int64_t, int64_t*);
template CSRMatrix<float, int64_t> apply_permutation<float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, const int64_t*);

// (Add double or __half instantiations here as needed)

} // namespace xspmm