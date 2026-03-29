#pragma once

#include "xspmm/matrix/csr.hpp"
#include "xspmm/matrix/bcsr.hpp"
#include "xspmm/core/timings.hpp"
#include <memory>

namespace xspmm {

// =================================================================================================
// TIER 1: The Expert API (What we already built)
// Performs raw Matrix Core math. Best for amortizing clustering over many iterations.
// =================================================================================================
template <typename InputType, typename OutputType, typename IndexType = int64_t>
void spmm(const BCSRMatrix<InputType, IndexType>& A,
          const InputType* B,
          OutputType* C,
          IndexType N);

// =================================================================================================
// TIER 2: The End-User Pipeline API (NEW)
// A drop-in replacement for Vendor APIs. Automates clustering, conversion, math, and un-permuting.
// =================================================================================================
    template <typename InputType, typename OutputType, typename IndexType = int64_t>
    void spmm(std::shared_ptr<const Executor> exec,
              const CSRMatrix<InputType, IndexType>& A,
              const InputType* B,
              OutputType* C,
              IndexType N,
              IndexType hw_block_size = 16,
              bool optimize_pattern = true,
              SpMMTimings* timings = nullptr);

// Utility to push the scrambled dense result back to the original topological order
template <typename ValueType, typename IndexType>
void unpermute_dense_matrix(std::shared_ptr<const Executor> exec,
                            const ValueType* in_matrix,
                            ValueType* out_matrix,
                            const IndexType* perm,
                            IndexType M, IndexType N);

} // namespace xspmm