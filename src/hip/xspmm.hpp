#pragma once

#include <memory>
#include <vector>
#include "xspmm/core/executor.hpp"
#include "xspmm/matrix/csr.hpp"
#include "xspmm/matrix/bcsr.hpp"

namespace xspmm {
    namespace hip {

        /**
         * @brief HIP backend entry point for SpMM.
         * Implemented in spmm_rocwmma.hip
         */
        template <typename ValueType, typename IndexType>
        void spmm(std::shared_ptr<const Executor> exec,
                  const BCSRMatrix<ValueType, IndexType>& A,
                  const ValueType* B,
                  ValueType* C,
                  IndexType N);

        /**
         * @brief HIP backend entry point for Jaccard Clustering.
         * Implemented in clustering_sylos_labini.hip
         */
        template <typename ValueType, typename IndexType>
        void compute_1d_jaccard_clustering(
            std::shared_ptr<const Executor> exec,
            const CSRMatrix<ValueType, IndexType>& A,
            float dist_thresh,
            IndexType block_width,
            IndexType* perm_out);

        /**
         * @brief HIP backend entry point for Matrix Permutation.
         * Implemented in clustering_sylos_labini.hip
         */
        template <typename ValueType, typename IndexType>
        CSRMatrix<ValueType, IndexType> apply_permutation(
            std::shared_ptr<const Executor> target_exec,
            const CSRMatrix<ValueType, IndexType>& A,
            const IndexType* perm);

    } // namespace hip
} // namespace xspmm