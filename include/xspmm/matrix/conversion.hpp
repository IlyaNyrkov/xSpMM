#pragma once

#include <memory>
#include "xspmm/matrix/csr.hpp"
#include "xspmm/matrix/bcsr.hpp"

namespace xspmm {

    /**
     * @brief Converts a CSR matrix to a BCSR matrix.
     * Note: Conversion logic currently occurs on the Host. If the input matrix
     * is on a GPU, it will be temporarily copied to RAM for the transformation.
     * * @param target_exec The executor where the resulting BCSR matrix should reside.
     * @param csr The input CSR matrix.
     * @param r_block Row block size.
     * @param c_block Col block size.
     * @return BCSRMatrix allocated on target_exec.
     */
    template <typename ValueType, typename IndexType = int64_t>
    BCSRMatrix<ValueType, IndexType> csr_to_bcsr(std::shared_ptr<const Executor> target_exec,
                                             const CSRMatrix<ValueType, IndexType>& csr,
                                             IndexType r_block, IndexType c_block);

} // namespace xspmm