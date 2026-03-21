#pragma once

#include "xspmm/matrix/bcsr.hpp"

namespace xspmm {

    /**
     * @brief Performs Unstructured Sparse Matrix-Dense Matrix Multiplication (SpMM).
     * Computes: C = A * B
     * * @tparam ValueType The numeric type of the matrix elements (e.g., float, double, __half).
     * @tparam IndexType The numeric type of the indices (defaults to int64_t).
     * * @param A The highly-optimized Block-CSR matrix. The Executor attached to this
     * matrix dictates which hardware backend (CUDA, HIP, CPU) will execute the math.
     * @param B Pointer to the dense matrix B (Device memory corresponding to A's Executor).
     * Matrix B is assumed to be size K x N, where K = A.get_num_block_cols() * A.get_c_block().
     * @param C Pointer to the dense matrix C (Device memory corresponding to A's Executor).
     * Matrix C is assumed to be size M x N, where M = A.get_num_block_rows() * A.get_r_block().
     * @param N The number of columns in dense matrices B and C.
     */
    template <typename ValueType, typename IndexType = int64_t>
    void spmm(const BCSRMatrix<ValueType, IndexType>& A,
              const ValueType* B,
              ValueType* C,
              IndexType N);

} // namespace xspmm