#include "xspmm.hpp"
#include "../common/spmm_core.cuh" // Include the single source of truth

namespace xspmm {
    namespace cuda {

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

            // NVIDIA Specific Tuning
            int threads_per_block = 32; // 1 Warp
            dim3 block(threads_per_block);
            dim3 grid(num_block_cols_b, num_block_rows_a);

            kernels::bcsr_spmm_core_kernel<InputType, OutputType, IndexType, BM, BN, BK><<<grid, block>>>(
                A.get_bcsr_row_ptr(), A.get_bcsr_col_ind(), A.get_bcsr_values(),
                B, C, num_block_rows_a, num_block_cols_b, N, N
            );
        }

        // Explicit Instantiations
        template void spmm<__half, float, int32_t>(std::shared_ptr<const Executor>, const BCSRMatrix<__half, int32_t>&, const __half*, float*, int32_t);
        template void spmm<__half, float, int64_t>(std::shared_ptr<const Executor>, const BCSRMatrix<__half, int64_t>&, const __half*, float*, int64_t);

    } // namespace cuda
} // namespace xspmm