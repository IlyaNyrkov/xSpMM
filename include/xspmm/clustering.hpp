#pragma once

#include <memory>
#include "xspmm/matrix/csr.hpp"
#include "xspmm/core/executor.hpp"

namespace xspmm {

    /**
     * @brief Computes an optimized row permutation to group scattered non-zero
     * elements into dense macro-blocks.
     * * @param exec The executor where the computation should occur.
     * @param A The original unoptimized CSR matrix.
     * @param dist_thresh The threshold for clustering similarity.
     * @param block_width The target dense block width.
     * @param perm_out Pre-allocated raw pointer (sized A.get_num_rows()) where
     * the resulting permutation indices will be written. Must reside on `exec`.
     */
    template <typename ValueType, typename IndexType = int64_t>
    void compute_1d_jaccard_clustering(
        std::shared_ptr<const Executor> exec,
        const CSRMatrix<ValueType, IndexType>& A,
        float dist_thresh,
        IndexType block_width,
        IndexType* perm_out);

    /**
     * @brief Physically reorders the rows of a CSR matrix according to a permutation array.
     * * @param target_exec The hardware executor where the NEW optimized matrix should reside.
     * @param A The original CSR matrix.
     * @param perm Raw pointer to the permutation array (sized A.get_num_rows()).
     * @return A new, structurally optimized CSRMatrix allocated on the target_exec.
     */
    template <typename ValueType, typename IndexType = int64_t>
    CSRMatrix<ValueType, IndexType> apply_permutation(
        std::shared_ptr<const Executor> target_exec,
        const CSRMatrix<ValueType, IndexType>& A,
        const IndexType* perm);

} // namespace xspmm