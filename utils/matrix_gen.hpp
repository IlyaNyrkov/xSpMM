#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace matrix_utils {
namespace generation {

    // -----------------------------------------------------------------
    // Helper to generate a random value (keeps the core logic clean)
    // -----------------------------------------------------------------
    template <typename ValueType>
    ValueType get_val(std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(0.1f, 1.0f);
        return static_cast<ValueType>(dist(rng));
    }

    // -----------------------------------------------------------------
    // UTILITY: Row Shuffler (To create "noisy" matrices for clustering)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void shuffle_rows(
        IndexType num_rows,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        std::vector<IndexType> perm(num_rows);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<IndexType> shuf_row_ptr;
        std::vector<IndexType> shuf_col_ind;
        std::vector<ValueType> shuf_values;

        shuf_row_ptr.reserve(num_rows + 1);
        shuf_col_ind.reserve(col_ind.size());
        shuf_values.reserve(values.size());

        shuf_row_ptr.push_back(0);

        for (IndexType i = 0; i < num_rows; ++i) {
            IndexType old_row = perm[i];
            IndexType start = row_ptr[old_row];
            IndexType end = row_ptr[old_row + 1];

            for (IndexType j = start; j < end; ++j) {
                shuf_col_ind.push_back(col_ind[j]);
                shuf_values.push_back(values[j]);
            }
            shuf_row_ptr.push_back(shuf_col_ind.size());
        }

        row_ptr = std::move(shuf_row_ptr);
        col_ind = std::move(shuf_col_ind);
        values  = std::move(shuf_values);
    }

    // -----------------------------------------------------------------
    // 1. Banded & Tridiagonal (bandwidth = 1 is Tridiagonal)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void generate_banded(
        IndexType size, IndexType bandwidth,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        row_ptr.clear(); col_ind.clear(); values.clear();
        row_ptr.reserve(size + 1);
        row_ptr.push_back(0);

        for (IndexType i = 0; i < size; ++i) {
            IndexType start_col = std::max<IndexType>(0, i - bandwidth);
            IndexType end_col = std::min<IndexType>(size - 1, i + bandwidth);
            for (IndexType j = start_col; j <= end_col; ++j) {
                col_ind.push_back(j);
                values.push_back(get_val<ValueType>(rng));
            }
            row_ptr.push_back(col_ind.size());
        }
    }

    // -----------------------------------------------------------------
    // 2. Triangular (Upper or Lower)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void generate_triangular(
        IndexType size, bool is_upper, float density,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        row_ptr.clear(); col_ind.clear(); values.clear();
        row_ptr.reserve(size + 1);
        row_ptr.push_back(0);

        for (IndexType i = 0; i < size; ++i) {
            IndexType start_col = is_upper ? i : 0;
            IndexType end_col = is_upper ? size - 1 : i;
            for (IndexType j = start_col; j <= end_col; ++j) {
                if (prob(rng) < density || i == j) { // Guarantee main diagonal
                    col_ind.push_back(j);
                    values.push_back(get_val<ValueType>(rng));
                }
            }
            row_ptr.push_back(col_ind.size());
        }
    }

    // -----------------------------------------------------------------
    // 3. Hessenberg (Upper: zeros below first subdiagonal)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void generate_hessenberg(
        IndexType size, bool is_upper, float density,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        row_ptr.clear(); col_ind.clear(); values.clear();
        row_ptr.reserve(size + 1);
        row_ptr.push_back(0);

        for (IndexType i = 0; i < size; ++i) {
            IndexType start_col = is_upper ? std::max<IndexType>(0, i - 1) : 0;
            IndexType end_col = is_upper ? size - 1 : std::min<IndexType>(size - 1, i + 1);
            for (IndexType j = start_col; j <= end_col; ++j) {
                if (prob(rng) < density || std::abs(static_cast<long long>(i) - static_cast<long long>(j)) <= 1) {
                    col_ind.push_back(j);
                    values.push_back(get_val<ValueType>(rng));
                }
            }
            row_ptr.push_back(col_ind.size());
        }
    }

    // -----------------------------------------------------------------
    // 4. Arrowhead & Kite (Dense first 'width' rows/cols + diagonal)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void generate_arrowhead(
        IndexType size, IndexType width,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        row_ptr.clear(); col_ind.clear(); values.clear();
        row_ptr.reserve(size + 1);
        row_ptr.push_back(0);

        for (IndexType i = 0; i < size; ++i) {
            for (IndexType j = 0; j < size; ++j) {
                if (i < width || j < width || i == j) {
                    col_ind.push_back(j);
                    values.push_back(get_val<ValueType>(rng));
                }
            }
            row_ptr.push_back(col_ind.size());
        }
    }

    // -----------------------------------------------------------------
    // 5. Checkers Format (Alternating dense blocks of 0s and 1s)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void generate_checkers(
        IndexType rows, IndexType cols, IndexType block_size,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        row_ptr.clear(); col_ind.clear(); values.clear();
        row_ptr.reserve(rows + 1);
        row_ptr.push_back(0);

        for (IndexType i = 0; i < rows; ++i) {
            IndexType row_block = i / block_size;
            for (IndexType j = 0; j < cols; ++j) {
                IndexType col_block = j / block_size;
                // Alternate based on block grid coordinates
                if ((row_block + col_block) % 2 == 1) {
                    col_ind.push_back(j);
                    values.push_back(get_val<ValueType>(rng));
                }
            }
            row_ptr.push_back(col_ind.size());
        }
    }

    // -----------------------------------------------------------------
    // 6. Block Diagonal
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void generate_block_diagonal(
        IndexType size, IndexType block_size, float density,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        row_ptr.clear(); col_ind.clear(); values.clear();
        row_ptr.reserve(size + 1);
        row_ptr.push_back(0);

        for (IndexType i = 0; i < size; ++i) {
            IndexType block_start = (i / block_size) * block_size;
            IndexType block_end = std::min<IndexType>(size - 1, block_start + block_size - 1);

            for (IndexType j = block_start; j <= block_end; ++j) {
                if (prob(rng) < density || i == j) {
                    col_ind.push_back(j);
                    values.push_back(get_val<ValueType>(rng));
                }
            }
            row_ptr.push_back(col_ind.size());
        }
    }

    // -----------------------------------------------------------------
    // 7. Near-Symmetric (Generate symmetric, then randomly drop edges)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void generate_near_symmetric(
        IndexType size, float density, float drop_rate,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        std::vector<std::vector<IndexType>> adj(size);
        for (IndexType i = 0; i < size; ++i) {
            for (IndexType j = i; j < size; ++j) {
                if (prob(rng) < density) {
                    if (prob(rng) > drop_rate) adj[i].push_back(j);
                    if (i != j && prob(rng) > drop_rate) adj[j].push_back(i);
                }
            }
        }

        row_ptr.clear(); col_ind.clear(); values.clear();
        row_ptr.reserve(size + 1);
        row_ptr.push_back(0);

        for (IndexType i = 0; i < size; ++i) {
            std::sort(adj[i].begin(), adj[i].end());
            for (IndexType col : adj[i]) {
                col_ind.push_back(col);
                values.push_back(get_val<ValueType>(rng));
            }
            row_ptr.push_back(col_ind.size());
        }
    }

    // -----------------------------------------------------------------
    // 8. Highly Skewed / Power-Law (Scale-Free Graphs)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    void generate_power_law(
        IndexType rows, IndexType cols, float alpha,
        std::vector<IndexType>& row_ptr,
        std::vector<IndexType>& col_ind,
        std::vector<ValueType>& values,
        int seed = 42)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.001f, 1.0f);

        row_ptr.clear(); col_ind.clear(); values.clear();
        row_ptr.reserve(rows + 1);
        row_ptr.push_back(0);

        for (IndexType i = 0; i < rows; ++i) {
            float u = prob(rng);
            IndexType degree = static_cast<IndexType>(std::pow(u, -1.0f / (alpha - 1.0f)));
            degree = std::min(degree, cols);
            degree = std::max<IndexType>(1, degree);

            std::vector<IndexType> random_cols(cols);
            std::iota(random_cols.begin(), random_cols.end(), 0);

            for(IndexType k = 0; k < degree; ++k) {
                std::uniform_int_distribution<IndexType> swap_dist(k, cols - 1);
                std::swap(random_cols[k], random_cols[swap_dist(rng)]);
            }

            std::sort(random_cols.begin(), random_cols.begin() + degree);

            for (IndexType k = 0; k < degree; ++k) {
                col_ind.push_back(random_cols[k]);
                values.push_back(get_val<ValueType>(rng));
            }
            row_ptr.push_back(col_ind.size());
        }
    }

    // -----------------------------------------------------------------
    // Generate 1D Dense Matrix (Row-Major)
    // -----------------------------------------------------------------
    template <typename ValueType, typename IndexType>
    std::vector<ValueType> generate_dense(IndexType rows, IndexType cols, int seed = 42) {
        std::mt19937 rng(seed);
        std::vector<ValueType> mat(rows * cols);
        for (IndexType i = 0; i < rows * cols; ++i) {
            mat[i] = get_val<ValueType>(rng);
        }
        return mat;
    }

} // namespace generation
} // namespace matrix_utils