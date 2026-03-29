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

// Forward declarations for CPU implementations
namespace cpu {
    template <typename ValueType, typename IndexType>
    void compute_1d_jaccard_clustering(std::shared_ptr<const Executor> exec, const CSRMatrix<ValueType, IndexType>& A, float dist_thresh, IndexType block_width, IndexType* perm_out);

    template <typename ValueType, typename IndexType>
    CSRMatrix<ValueType, IndexType> apply_permutation(std::shared_ptr<const Executor> target_exec, const CSRMatrix<ValueType, IndexType>& A, const IndexType* perm);
}

// =================================================================================================
// 1. SpMM Dispatcher (Multi-Precision)
// =================================================================================================
template <typename InputType, typename OutputType, typename IndexType>
void spmm(const BCSRMatrix<InputType, IndexType>& A,
          const InputType* B,
          OutputType* C,
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

    throw std::runtime_error("xSpMM: CPU SpMM fallback is not yet implemented. Please use a GPU Executor.");
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
    // Fallback to CPU algorithm (It safely handles GPU memory transfers internally)
    cpu::compute_1d_jaccard_clustering(exec, A, dist_thresh, block_width, perm_out);
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
// =================================================================================================


template void spmm<float, float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, const float*, float*, int32_t, int32_t, bool, SpMMTimings*);
template void spmm<float, float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, const float*, float*, int64_t, int64_t, bool, SpMMTimings*);

// --- SpMM Math ---
template void spmm<float, float, int32_t>(const BCSRMatrix<float, int32_t>&, const float*, float*, int32_t);
template void spmm<float, float, int64_t>(const BCSRMatrix<float, int64_t>&, const float*, float*, int64_t);

// --- Clustering & Permutation Utils ---
template void compute_1d_jaccard_clustering<float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, float, int32_t, int32_t*);
template CSRMatrix<float, int32_t> apply_permutation<float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, const int32_t*);

template void compute_1d_jaccard_clustering<float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, float, int64_t, int64_t*);
template CSRMatrix<float, int64_t> apply_permutation<float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, const int64_t*);

template <typename ValueType, typename IndexType>
void unpermute_dense_matrix(std::shared_ptr<const Executor> exec,
                            const ValueType* in_matrix,
                            ValueType* out_matrix,
                            const IndexType* perm,
                            IndexType M, IndexType N)
    {
#ifdef XSPMM_ENABLE_CUDA
        if (auto cuda_exec = dynamic_cast<const CudaExecutor*>(exec.get())) {
            cuda::unpermute_dense_matrix(exec, in_matrix, out_matrix, perm, M, N); return;
        }
#endif
#ifdef XSPMM_ENABLE_HIP
        if (auto hip_exec = dynamic_cast<const HipExecutor*>(exec.get())) {
            hip::unpermute_dense_matrix(exec, in_matrix, out_matrix, perm, M, N); return;
        }
#endif
        throw std::runtime_error("CPU unpermute not yet implemented in pipeline.");
}

    // TIER 2: High-Level Automated Pipeline Dispatcher
    template <typename InputType, typename OutputType, typename IndexType>
    void spmm(std::shared_ptr<const Executor> exec,
              const CSRMatrix<InputType, IndexType>& A,
              const InputType* B,
              OutputType* C,
              IndexType N,
              IndexType hw_block_size,
              bool optimize_pattern,
              SpMMTimings* timings)
{
#ifdef XSPMM_ENABLE_CUDA
    if (auto cuda_exec = dynamic_cast<const CudaExecutor*>(exec.get())) {
        cuda::spmm<InputType, OutputType, IndexType>(exec, A, B, C, N, hw_block_size, optimize_pattern, timings);
        return;
    }
#endif

#ifdef XSPMM_ENABLE_HIP
    if (auto hip_exec = dynamic_cast<const HipExecutor*>(exec.get())) {
        hip::spmm<InputType, OutputType, IndexType>(exec, A, B, C, N, hw_block_size, optimize_pattern, timings);
        return;
    }
#endif

    throw std::runtime_error("xSpMM: CPU pipeline fallback is not yet implemented.");
}

} // namespace xspmm