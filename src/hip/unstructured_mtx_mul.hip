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

    if (optimize_pattern) {
        // --- PATH A: WITH CLUSTERING ---

        // 1. Clustering
        if (timings) t_step.record_start();
        IndexType* d_perm = nullptr;
        exec->allocate(reinterpret_cast<void**>(&d_perm), (size_t)M * sizeof(IndexType));
        xspmm::compute_1d_jaccard_clustering(exec, A, 0.5f, hw_block_size, d_perm);
        if (timings) timings->clustering_ms = t_step.record_stop_and_get_ms();

        // 2. Permutation (RVO guarantees safe memory transfer!)
        if (timings) t_step.record_start();
        auto A_opt = xspmm::apply_permutation(exec, A, d_perm);
        if (timings) timings->permutation_ms = t_step.record_stop_and_get_ms();

        // 3. Conversion
        if (timings) t_step.record_start();
        auto A_bcsr = xspmm::csr_to_bcsr(exec, A_opt, hw_block_size, hw_block_size);
        if (timings) timings->conversion_ms = t_step.record_stop_and_get_ms();

        // 4. Matrix Math

        OutputType* d_C_temp = nullptr;
        exec->allocate(reinterpret_cast<void**>(&d_C_temp), (size_t)M * (size_t)N * sizeof(OutputType));

        hipError_t eMem = hipMemset(d_C_temp, 0, (size_t)M * (size_t)N * sizeof(OutputType)); (void)eMem;

        if (timings) t_step.record_start();
        spmm(exec, A_bcsr, B, d_C_temp, N); // Call Tier 1
        if (timings) timings->spmm_ms = t_step.record_stop_and_get_ms();

        // 5. Un-Permute
        if (timings) t_step.record_start();
        xspmm::hip::unpermute_dense_matrix(exec, d_C_temp, C, d_perm, M, N);
        if (timings) timings->unpermutation_ms = t_step.record_stop_and_get_ms();

        // Cleanup
        exec->free(d_perm);
        exec->free(d_C_temp);

    } else {
        // --- PATH B: NO CLUSTERING (Direct Fallback) ---

        // 3. Conversion
        if (timings) t_step.record_start();
        auto A_bcsr = xspmm::csr_to_bcsr(exec, A, hw_block_size, hw_block_size);
        if (timings) timings->conversion_ms = t_step.record_stop_and_get_ms();

        // 4. Matrix Math
        if (timings) t_step.record_start();
        hipError_t eMem = hipMemset(C, 0, (size_t)M * (size_t)N * sizeof(OutputType)); (void)eMem;

        spmm(exec, A_bcsr, B, C, N); // Call Tier 1
        if (timings) timings->spmm_ms = t_step.record_stop_and_get_ms();
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

    // Helper lambda to dispatch template arguments
    auto launch_kernel = [&](auto warps_const, auto is_clean_const) {
        constexpr int WARPS = decltype(warps_const)::value;
        constexpr bool IS_CLEAN = decltype(is_clean_const)::value;
        constexpr int threads = WARPS * HW_WARP_SIZE;

        dim3 block(threads);
        dim3 grid((num_block_cols_b + WARPS - 1) / WARPS, num_block_rows_a);

        kernels::bcsr_spmm_core_kernel<InputType, OutputType, IndexType, BM, BN, BK, WARPS, IS_CLEAN><<<grid, block, 0, 0>>>(
            A.get_bcsr_row_ptr(), A.get_bcsr_col_ind(), A.get_bcsr_values(),
            B, C, num_block_rows_a, num_block_cols_b, N, N
        );
    };

    bool clean_multiple = (N % 16 == 0);

    if (clean_multiple) {
        if (N <= 16) launch_kernel(std::integral_constant<int, 1>{}, std::true_type{});
        else if (N <= 32) launch_kernel(std::integral_constant<int, 2>{}, std::true_type{});
        else launch_kernel(std::integral_constant<int, 4>{}, std::true_type{});
    } else {
        if (N <= 16) launch_kernel(std::integral_constant<int, 1>{}, std::false_type{});
        else if (N <= 32) launch_kernel(std::integral_constant<int, 2>{}, std::false_type{});
        else launch_kernel(std::integral_constant<int, 4>{}, std::false_type{});
    }

}

// Explicit Instantiations
template void spmm<__half, float, int32_t>(std::shared_ptr<const Executor>, const BCSRMatrix<__half, int32_t>&, const __half*, float*, int32_t);
template void spmm<__half, float, int64_t>(std::shared_ptr<const Executor>, const BCSRMatrix<__half, int64_t>&, const __half*, float*, int64_t);

template void spmm<float, float, int32_t>(std::shared_ptr<const Executor>, const BCSRMatrix<float, int32_t>&, const float*, float*, int32_t);
template void spmm<float, float, int64_t>(std::shared_ptr<const Executor>, const BCSRMatrix<float, int64_t>&, const float*, float*, int64_t);

} // namespace hip
} // namespace xspmm