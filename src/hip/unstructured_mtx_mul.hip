#include <optional>
#include "xspmm.hpp"
#include "xspmm/xspmm.hpp"
#include "../common/spmm_math.hpp"
#include "xspmm/matrix/conversion.hpp"
#include "xspmm/clustering.hpp"

namespace xspmm {
namespace hip {

// =================================================================================================
// Native HIP Timing Utility
// =================================================================================================
struct HipTimer {
    hipEvent_t start, stop;
    HipTimer() { hipEventCreate(&start); hipEventCreate(&stop); }
    ~HipTimer() { hipEventDestroy(start); hipEventDestroy(stop); }
    void record_start() { hipEventRecord(start); }
    double record_stop_and_get_ms() {
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        return static_cast<double>(ms);
    }
};

// =================================================================================================
// TIER 2: High-Level Automated Pipeline
// =================================================================================================
template <typename InputType, typename OutputType, typename IndexType>
void spmm(std::shared_ptr<const Executor> exec,
          const CSRMatrix<InputType, IndexType>& A,
          const InputType* B, OutputType* C,
          IndexType N, IndexType hw_block_size,
          bool optimize_pattern, SpMMTimings* timings)
{
    HipTimer t_total, t_step;
    if (timings) t_total.record_start();

    IndexType M = A.get_num_rows();

    // Setup pointers and active references
    IndexType* d_perm = nullptr;
    OutputType* d_C_temp = nullptr;

    // Default to the original matrix unless optimization is enabled
    const CSRMatrix<InputType, IndexType>* active_A = &A;
    std::optional<CSRMatrix<InputType, IndexType>> A_opt;

    // -------------------------------------------------------------------------
    // Optional Stage: Clustering & Permutation
    // -------------------------------------------------------------------------
    if (optimize_pattern) {
        // 1. Clustering
        if (timings) t_step.record_start();
        exec->allocate(reinterpret_cast<void**>(&d_perm), M * sizeof(IndexType));
        compute_1d_jaccard_clustering(exec, A, 0.5f, hw_block_size, d_perm);
        if (timings) timings->clustering_ms = t_step.record_stop_and_get_ms();

        // 2. Permutation
        if (timings) t_step.record_start();
        A_opt = xspmm::hip::apply_permutation(exec, A, d_perm);
        active_A = &A_opt.value(); // Re-route the active pointer to the optimized matrix
        if (timings) timings->permutation_ms = t_step.record_stop_and_get_ms();

        // Allocate a temporary buffer for C because the output will be scrambled
        exec->allocate(reinterpret_cast<void**>(&d_C_temp), M * N * sizeof(OutputType));
    }

    // -------------------------------------------------------------------------
    // Core Pipeline: Conversion & Math (Executed regardless of optimization)
    // -------------------------------------------------------------------------

    // 3. Conversion
    if (timings) t_step.record_start();
    auto A_bcsr = csr_to_bcsr(exec, *active_A, hw_block_size, hw_block_size);
    if (timings) timings->conversion_ms = t_step.record_stop_and_get_ms();

    // 4. Matrix Math
    if (timings) t_step.record_start();
    // Route output to d_C_temp if permuted, otherwise directly to the user's C buffer
    OutputType* target_C = optimize_pattern ? d_C_temp : C;

    // Native hardware zeroing for clean accumulator states
    hipMemset(target_C, 0, M * N * sizeof(OutputType));

    spmm(exec, A_bcsr, B, target_C, N); // Call the Tier 1 Expert API
    if (timings) timings->spmm_ms = t_step.record_stop_and_get_ms();

    // -------------------------------------------------------------------------
    // Optional Stage: Un-Permutation & Cleanup
    // -------------------------------------------------------------------------
    if (optimize_pattern) {
        // 5. Un-Permute
        if (timings) t_step.record_start();
        xspmm::hip::unpermute_dense_matrix(exec, d_C_temp, C, d_perm, M, N);
        if (timings) timings->unpermutation_ms = t_step.record_stop_and_get_ms();

        exec->free(d_perm);
        exec->free(d_C_temp);
    }

    if (timings) timings->total_pipeline_ms = t_total.record_stop_and_get_ms();
}

// =================================================================================================
// Explicit Instantiations
// =================================================================================================
template void spmm<float, float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, const float*, float*, int32_t, int32_t, bool, SpMMTimings*);
template void spmm<float, float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, const float*, float*, int64_t, int64_t, bool, SpMMTimings*);

template <typename InputType, typename OutputType, typename IndexType>
void spmm(std::shared_ptr<const Executor> exec,
          const BCSRMatrix<InputType, IndexType>& A,
          const InputType* B,
          OutputType* C,
          IndexType N)
{
    IndexType num_block_rows_a = A.get_num_block_rows();
    const int BM = 16, BN = 16, BK = 16;
    IndexType num_block_cols_b = (N + BN - 1) / BN;

    // AMD Specific Tuning
    int threads_per_block = 64; // 1 Wavefront
    dim3 block(threads_per_block);
    dim3 grid(num_block_cols_b, num_block_rows_a);

    kernels::bcsr_spmm_core_kernel<InputType, OutputType, IndexType, BM, BN, BK><<<grid, block, 0, 0>>>(
        A.get_bcsr_row_ptr(), A.get_bcsr_col_ind(), A.get_bcsr_values(),
        B, C, num_block_rows_a, num_block_cols_b, N, N
    );
}

// Explicit Instantiations
template void spmm<__half, float, int32_t>(std::shared_ptr<const Executor>, const BCSRMatrix<__half, int32_t>&, const __half*, float*, int32_t);
template void spmm<__half, float, int64_t>(std::shared_ptr<const Executor>, const BCSRMatrix<__half, int64_t>&, const __half*, float*, int64_t);

template void spmm<float, float, int32_t>(std::shared_ptr<const Executor>, const BCSRMatrix<float, int32_t>&, const float*, float*, int32_t);
template void spmm<float, float, int64_t>(std::shared_ptr<const Executor>, const BCSRMatrix<float, int64_t>&, const float*, float*, int64_t);

} // namespace hip
} // namespace xspmm