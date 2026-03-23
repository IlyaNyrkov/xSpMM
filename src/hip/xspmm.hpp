#pragma once

#include <memory>
#include "xspmm/core/executor.hpp"
#include "xspmm/matrix/csr.hpp"
#include "xspmm/matrix/bcsr.hpp"

namespace xspmm {
namespace hip {

template <typename InputType, typename OutputType, typename IndexType>
void spmm(std::shared_ptr<const Executor> exec,
          const BCSRMatrix<InputType, IndexType>& A,
          const InputType* B,
          OutputType* C,
          IndexType N);

template <typename ValueType, typename IndexType>
CSRMatrix<ValueType, IndexType> apply_permutation(std::shared_ptr<const Executor> target_exec,
                                                    const CSRMatrix<ValueType, IndexType>& A,
                                                    const IndexType* perm);

template <typename ValueType, typename IndexType>
void unpermute_dense_matrix(std::shared_ptr<const Executor> exec,
                            const ValueType* in_matrix,
                            ValueType* out_matrix,
                            const IndexType* perm,
                            IndexType M, IndexType N);

} // namespace hip
} // namespace xspmm