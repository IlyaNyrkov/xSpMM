#pragma once

#include <memory>
#include <stdexcept>
#include <cstdint>
#include "xspmm/core/executor.hpp"

namespace xspmm {

// IndexType defaults to int64_t to prevent overflow by default,
// but allows int32_t for bandwidth optimization if requested.
template <typename ValueType, typename IndexType = int64_t>
class CSRMatrix {
public:
    CSRMatrix(std::shared_ptr<const Executor> exec, IndexType num_rows, IndexType num_cols, IndexType nnz)
        : exec_(exec), num_rows_(num_rows), num_cols_(num_cols), nnz_(nnz),
          row_ptr_(nullptr), col_ind_(nullptr), values_(nullptr)
    {
        if (num_rows < 0 || num_cols < 0 || nnz < 0) {
            throw std::invalid_argument("CSR dimensions must be non-negative.");
        }

        exec_->allocate(reinterpret_cast<void**>(&row_ptr_), (num_rows + 1) * sizeof(IndexType));
        if (nnz > 0) {
            exec_->allocate(reinterpret_cast<void**>(&col_ind_), nnz * sizeof(IndexType));
            exec_->allocate(reinterpret_cast<void**>(&values_), nnz * sizeof(ValueType));
        }
    }

    ~CSRMatrix() {
        if (row_ptr_) exec_->free(row_ptr_);
        if (col_ind_) exec_->free(col_ind_);
        if (values_)  exec_->free(values_);
    }

    // [Move constructors / assignment operators remain the same, just swap 'int' for 'IndexType']...

    // Accessors
    IndexType get_num_rows() const { return num_rows_; }
    IndexType get_num_cols() const { return num_cols_; }
    IndexType get_nnz() const { return nnz_; }

    IndexType* get_row_ptr() const { return row_ptr_; }
    IndexType* get_col_ind() const { return col_ind_; }
    ValueType* get_values() const { return values_; }

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    // Updated: Now accepts raw pointers. The user is responsible for ensuring
    // the source arrays are correctly sized!
    void copy_from_host(const IndexType* host_row_ptr,
                        const IndexType* host_col_ind,
                        const ValueType* host_values)
    {
        exec_->copy_from_host(row_ptr_, host_row_ptr, (num_rows_ + 1) * sizeof(IndexType));
        if (nnz_ > 0) {
            exec_->copy_from_host(col_ind_, host_col_ind, nnz_ * sizeof(IndexType));
            exec_->copy_from_host(values_, host_values, nnz_ * sizeof(ValueType));
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
    IndexType num_rows_;
    IndexType num_cols_;
    IndexType nnz_;

    IndexType* row_ptr_;
    IndexType* col_ind_;
    ValueType* values_;
};

} // namespace xspmm