#pragma once

#include <memory>
#include <stdexcept>
#include <cstdint>
#include "xspmm/core/executor.hpp"

namespace xspmm {

// IndexType defaults to int64_t to prevent overflow by default,
// but allows int32_t for bandwidth optimization if requested.
template <typename ValueType, typename IndexType = int64_t>
class BCSRMatrix {
public:
    BCSRMatrix(std::shared_ptr<const Executor> exec,
               IndexType num_block_rows, IndexType num_block_cols, IndexType num_blocks,
               IndexType r_block, IndexType c_block)
        : exec_(exec),
          num_block_rows_(num_block_rows), num_block_cols_(num_block_cols), num_blocks_(num_blocks),
          r_block_(r_block), c_block_(c_block),
          bcsr_row_ptr_(nullptr), bcsr_col_ind_(nullptr), bcsr_values_(nullptr)
    {
        if (num_block_rows < 0 || num_block_cols < 0 || num_blocks < 0) {
            throw std::invalid_argument("BCSR dimensions must be non-negative.");
        }

        exec_->allocate(reinterpret_cast<void**>(&bcsr_row_ptr_), (num_block_rows + 1) * sizeof(IndexType));
        if (num_blocks > 0) {
            exec_->allocate(reinterpret_cast<void**>(&bcsr_col_ind_), num_blocks * sizeof(IndexType));
            IndexType total_values = num_blocks * r_block * c_block;
            exec_->allocate(reinterpret_cast<void**>(&bcsr_values_), total_values * sizeof(ValueType));
        }
    }

    ~BCSRMatrix() {
        if (bcsr_row_ptr_) exec_->free(bcsr_row_ptr_);
        if (bcsr_col_ind_) exec_->free(bcsr_col_ind_);
        if (bcsr_values_)  exec_->free(bcsr_values_);
    }

    BCSRMatrix(const BCSRMatrix&) = delete;
    BCSRMatrix& operator=(const BCSRMatrix&) = delete;

    BCSRMatrix(BCSRMatrix&& other) noexcept
        : exec_(std::move(other.exec_)),
          num_block_rows_(other.num_block_rows_), num_block_cols_(other.num_block_cols_), num_blocks_(other.num_blocks_),
          r_block_(other.r_block_), c_block_(other.c_block_),
          bcsr_row_ptr_(other.bcsr_row_ptr_), bcsr_col_ind_(other.bcsr_col_ind_), bcsr_values_(other.bcsr_values_)
    {
        other.bcsr_row_ptr_ = nullptr;
        other.bcsr_col_ind_ = nullptr;
        other.bcsr_values_ = nullptr;
    }

    BCSRMatrix& operator=(BCSRMatrix&& other) noexcept {
        if (this != &other) {
            if (bcsr_row_ptr_) exec_->free(bcsr_row_ptr_);
            if (bcsr_col_ind_) exec_->free(bcsr_col_ind_);
            if (bcsr_values_)  exec_->free(bcsr_values_);

            exec_ = std::move(other.exec_);
            num_block_rows_ = other.num_block_rows_;
            num_block_cols_ = other.num_block_cols_;
            num_blocks_ = other.num_blocks_;
            r_block_ = other.r_block_;
            c_block_ = other.c_block_;

            bcsr_row_ptr_ = other.bcsr_row_ptr_;
            bcsr_col_ind_ = other.bcsr_col_ind_;
            bcsr_values_ = other.bcsr_values_;

            other.bcsr_row_ptr_ = nullptr;
            other.bcsr_col_ind_ = nullptr;
            other.bcsr_values_ = nullptr;
        }
        return *this;
    }

    // Accessors
    IndexType get_num_block_rows() const { return num_block_rows_; }
    IndexType get_num_block_cols() const { return num_block_cols_; }
    IndexType get_num_blocks() const { return num_blocks_; }
    IndexType get_r_block() const { return r_block_; }
    IndexType get_c_block() const { return c_block_; }

    IndexType* get_bcsr_row_ptr() const { return bcsr_row_ptr_; }
    IndexType* get_bcsr_col_ind() const { return bcsr_col_ind_; }
    ValueType* get_bcsr_values() const { return bcsr_values_; }

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    // Helper for uploading CPU data using raw pointers
    void copy_from_host(const IndexType* host_row_ptr,
                        const IndexType* host_col_ind,
                        const ValueType* host_values)
    {
        exec_->copy_from_host(bcsr_row_ptr_, host_row_ptr, (num_block_rows_ + 1) * sizeof(IndexType));
        if (num_blocks_ > 0) {
            exec_->copy_from_host(bcsr_col_ind_, host_col_ind, num_blocks_ * sizeof(IndexType));
            IndexType total_values = num_blocks_ * r_block_ * c_block_;
            exec_->copy_from_host(bcsr_values_, host_values, total_values * sizeof(ValueType));
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
    IndexType num_block_rows_;
    IndexType num_block_cols_;
    IndexType num_blocks_;
    IndexType r_block_;
    IndexType c_block_;

    IndexType* bcsr_row_ptr_;
    IndexType* bcsr_col_ind_;
    ValueType* bcsr_values_;
};

} // namespace xspmm