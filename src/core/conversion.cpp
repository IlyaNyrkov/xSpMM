#include "xspmm/matrix/conversion.hpp"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace xspmm {

template <typename ValueType, typename IndexType>
BCSRMatrix<ValueType, IndexType> csr_to_bcsr(std::shared_ptr<const Executor> target_exec,
                                             const CSRMatrix<ValueType, IndexType>& csr,
                                             IndexType r_block, IndexType c_block)
{
    IndexType num_rows = csr.get_num_rows();
    IndexType num_cols = csr.get_num_cols();
    IndexType nnz = csr.get_nnz();

    // 1. Pull CSR to CPU if it is on the GPU
    std::vector<IndexType> h_row_ptr(num_rows + 1);
    std::vector<IndexType> h_col_ind(nnz);
    std::vector<ValueType> h_values(nnz);

    csr.get_executor()->copy_to_host(h_row_ptr.data(), csr.get_row_ptr(), (num_rows + 1) * sizeof(IndexType));
    if (nnz > 0) {
        csr.get_executor()->copy_to_host(h_col_ind.data(), csr.get_col_ind(), nnz * sizeof(IndexType));
        csr.get_executor()->copy_to_host(h_values.data(), csr.get_values(), nnz * sizeof(ValueType));
    }

    // 2. Compute BCSR Topology on CPU
    IndexType num_block_rows = (num_rows + r_block - 1) / r_block;
    IndexType num_block_cols = (num_cols + c_block - 1) / c_block;

    std::vector<IndexType> bcsr_row_ptr;
    bcsr_row_ptr.reserve(num_block_rows + 1);
    bcsr_row_ptr.push_back(0);

    std::vector<IndexType> bcsr_col_ind;
    std::vector<ValueType> bcsr_values;

    std::vector<IndexType> block_map(num_block_cols, -1);
    IndexType current_block_count = 0;

    for (IndexType br = 0; br < num_block_rows; ++br) {
        std::vector<IndexType> active_block_cols;

        IndexType r_start = br * r_block;
        IndexType r_end = std::min(num_rows, r_start + r_block);

        // Phase 1: Identify active blocks
        for (IndexType r = r_start; r < r_end; ++r) {
            for (IndexType idx = h_row_ptr[r]; idx < h_row_ptr[r+1]; ++idx) {
                IndexType c = h_col_ind[idx];
                IndexType bc = c / c_block;

                if (block_map[bc] != br) {
                    block_map[bc] = br;
                    active_block_cols.push_back(bc);
                }
            }
        }

        std::sort(active_block_cols.begin(), active_block_cols.end());
        for (IndexType bc : active_block_cols) {
            bcsr_col_ind.push_back(bc);
        }

        IndexType num_blocks_in_row = active_block_cols.size();
        current_block_count += num_blocks_in_row;
        bcsr_row_ptr.push_back(current_block_count);

        // Phase 2: Scatter values into dense blocks
        IndexType values_start_offset = bcsr_values.size();
        IndexType elements_to_add = num_blocks_in_row * r_block * c_block;
        bcsr_values.resize(values_start_offset + elements_to_add, static_cast<ValueType>(0));

        for (IndexType r = r_start; r < r_end; ++r) {
            for (IndexType idx = h_row_ptr[r]; idx < h_row_ptr[r+1]; ++idx) {
                IndexType c = h_col_ind[idx];
                ValueType val = h_values[idx];

                IndexType bc = c / c_block;
                IndexType intra_block_r = r % r_block;
                IndexType intra_block_c = c % c_block;

                auto it = std::lower_bound(active_block_cols.begin(), active_block_cols.end(), bc);
                IndexType block_offset_in_row = std::distance(active_block_cols.begin(), it);

                IndexType global_block_idx = bcsr_row_ptr[br] + block_offset_in_row;
                IndexType flat_idx = (global_block_idx * r_block * c_block) + (intra_block_r * c_block) + intra_block_c;

                bcsr_values[flat_idx] = val;
            }
        }
    }

    // 3. Create target Matrix and push data to its Executor
    BCSRMatrix<ValueType, IndexType> result(target_exec, num_block_rows, num_block_cols, current_block_count, r_block, c_block);
    result.copy_from_host(bcsr_row_ptr.data(), bcsr_col_ind.data(), bcsr_values.data());

    return result;
}

// =================================================================================================
// Explicit Instantiations
// =================================================================================================
// For 32-bit indices
template BCSRMatrix<float, int32_t> csr_to_bcsr<float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, int32_t, int32_t);
// For 64-bit indices
template BCSRMatrix<float, int64_t> csr_to_bcsr<float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, int64_t, int64_t);

} // namespace xspmm