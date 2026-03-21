#include "xspmm/clustering.hpp"
#include <vector>
#include <cstdint>
#include <bit>
#include <stdexcept>

namespace xspmm {
namespace cpu {

using BitVector = std::vector<uint64_t>;

// Internal helper: computes the bit-compressed quotient rows
template <typename IndexType>
std::vector<BitVector> computeQuotientRows(IndexType num_rows, IndexType num_cols, IndexType block_width,
                                           const std::vector<IndexType>& h_row_ptr,
                                           const std::vector<IndexType>& h_col_ind)
{
    IndexType num_blocks = (num_cols + block_width - 1) / block_width;
    IndexType num_words = (num_blocks + 63) / 64;

    std::vector<BitVector> quotient_rows(num_rows, BitVector(num_words, 0ULL));

    for (IndexType i = 0; i < num_rows; ++i) {
        for (IndexType j = h_row_ptr[i]; j < h_row_ptr[i+1]; ++j) {
            IndexType col = h_col_ind[j];
            IndexType block_idx = col / block_width;
            IndexType word_idx = block_idx / 64;
            IndexType bit_idx = block_idx % 64;
            quotient_rows[i][word_idx] |= (1ULL << bit_idx);
        }
    }
    return quotient_rows;
}

inline float computeJaccardDistance(const BitVector& v, const BitVector& w) {
    int intersection_pop = 0;
    int union_pop = 0;

    for (size_t i = 0; i < v.size(); ++i) {
        intersection_pop += std::popcount(v[i] & w[i]);
        union_pop += std::popcount(v[i] | w[i]);
    }

    if (union_pop == 0) return 0.0f;
    return 1.0f - (static_cast<float>(intersection_pop) / union_pop);
}

// =================================================================================================
// 1. Compute Clustering (CPU Fallback)
// =================================================================================================
template <typename ValueType, typename IndexType>
void compute_1d_jaccard_clustering(
    std::shared_ptr<const Executor> exec,
    const CSRMatrix<ValueType, IndexType>& A,
    float dist_thresh,
    IndexType block_width,
    IndexType* perm_out)
{
    IndexType num_rows = A.get_num_rows();
    IndexType num_cols = A.get_num_cols();
    IndexType nnz = A.get_nnz();

    if (num_rows == 0) return;

    // 1. Pull topology to CPU (A might be on a GPU!)
    // We only need row_ptr and col_ind for clustering, not the values.
    std::vector<IndexType> h_row_ptr(num_rows + 1);
    std::vector<IndexType> h_col_ind(nnz);

    A.get_executor()->copy_to_host(h_row_ptr.data(), A.get_row_ptr(), (num_rows + 1) * sizeof(IndexType));
    if (nnz > 0) {
        A.get_executor()->copy_to_host(h_col_ind.data(), A.get_col_ind(), nnz * sizeof(IndexType));
    }

    // 2. Compute Clustering on CPU
    std::vector<BitVector> V_quotients = computeQuotientRows(num_rows, num_cols, block_width, h_row_ptr, h_col_ind);

    std::vector<IndexType> unclustered;
    unclustered.reserve(num_rows);
    for (IndexType i = 0; i < num_rows; ++i) {
        unclustered.push_back(i);
    }

    std::vector<std::vector<IndexType>> clusters;

    while (!unclustered.empty()) {
        IndexType seed_idx = unclustered.back();
        unclustered.pop_back();

        std::vector<IndexType> c = {seed_idx};
        BitVector p_c = V_quotients[seed_idx];

        for (int64_t i = static_cast<int64_t>(unclustered.size()) - 1; i >= 0; --i) {
            IndexType candidate_idx = unclustered[i];
            float dist = computeJaccardDistance(p_c, V_quotients[candidate_idx]);

            if (dist <= dist_thresh) {
                c.push_back(candidate_idx);
                for (size_t word = 0; word < p_c.size(); ++word) {
                    p_c[word] |= V_quotients[candidate_idx][word];
                }
                unclustered[i] = unclustered.back();
                unclustered.pop_back();
            }
        }
        clusters.push_back(c);
    }

    // 3. Flatten clusters into a single host permutation array
    std::vector<IndexType> h_final_permutation;
    h_final_permutation.reserve(num_rows);
    for (const auto& cluster : clusters) {
        for (IndexType row_idx : cluster) {
            h_final_permutation.push_back(row_idx);
        }
    }

    // 4. Push the result to the target executor (copy to GPU if perm_out is on GPU)
    exec->copy_from_host(perm_out, h_final_permutation.data(), num_rows * sizeof(IndexType));
}

// =================================================================================================
// 2. Apply Permutation (CPU Fallback)
// =================================================================================================
template <typename ValueType, typename IndexType>
CSRMatrix<ValueType, IndexType> apply_permutation(
    std::shared_ptr<const Executor> target_exec,
    const CSRMatrix<ValueType, IndexType>& A,
    const IndexType* perm)
{
    IndexType num_rows = A.get_num_rows();
    IndexType num_cols = A.get_num_cols();
    IndexType nnz = A.get_nnz();

    // 1. Pull matrix A and permutation array to Host
    std::vector<IndexType> h_row_ptr(num_rows + 1);
    std::vector<IndexType> h_col_ind(nnz);
    std::vector<ValueType> h_values(nnz);
    std::vector<IndexType> h_perm(num_rows);

    A.get_executor()->copy_to_host(h_row_ptr.data(), A.get_row_ptr(), (num_rows + 1) * sizeof(IndexType));
    if (nnz > 0) {
        A.get_executor()->copy_to_host(h_col_ind.data(), A.get_col_ind(), nnz * sizeof(IndexType));
        A.get_executor()->copy_to_host(h_values.data(), A.get_values(), nnz * sizeof(ValueType));
    }

    // Copy the permutation array from wherever it lives (target_exec) to host
    target_exec->copy_to_host(h_perm.data(), perm, num_rows * sizeof(IndexType));

    // 2. Build the new permuted arrays on CPU
    std::vector<IndexType> new_row_ptr;
    std::vector<IndexType> new_col_ind;
    std::vector<ValueType> new_values;

    new_row_ptr.reserve(num_rows + 1);
    new_col_ind.reserve(nnz);
    new_values.reserve(nnz);

    new_row_ptr.push_back(0);

    for (IndexType i = 0; i < num_rows; ++i) {
        IndexType old_row = h_perm[i];
        IndexType start = h_row_ptr[old_row];
        IndexType end = h_row_ptr[old_row + 1];

        for (IndexType j = start; j < end; ++j) {
            new_col_ind.push_back(h_col_ind[j]);
            new_values.push_back(h_values[j]);
        }
        new_row_ptr.push_back(new_col_ind.size());
    }

    // 3. Create the target Matrix and upload
    CSRMatrix<ValueType, IndexType> permuted_mat(target_exec, num_rows, num_cols, nnz);
    permuted_mat.copy_from_host(new_row_ptr.data(), new_col_ind.data(), new_values.data());

    return permuted_mat;
}

// =================================================================================================
// Explicit Instantiations
// =================================================================================================
template void compute_1d_jaccard_clustering<float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, float, int32_t, int32_t*);
template CSRMatrix<float, int32_t> apply_permutation<float, int32_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int32_t>&, const int32_t*);

template void compute_1d_jaccard_clustering<float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, float, int64_t, int64_t*);
template CSRMatrix<float, int64_t> apply_permutation<float, int64_t>(std::shared_ptr<const Executor>, const CSRMatrix<float, int64_t>&, const int64_t*);

// (Add double or __half instantiations as required)

} // namespace cpu
} // namespace xspmm