#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "matrix_utils/types.hpp"

namespace matrix_utils {
namespace generation {

    // -----------------------------------------------------------------
    // UTILITY: Row Shuffler (To create "noisy" matrices for clustering)
    // -----------------------------------------------------------------
    template <typename T>
    void shuffle_rows(CSRMatrix<T>& mat, int seed = 42) {
        std::mt19937 rng(seed);
        std::vector<int> perm(mat.num_rows);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        CSRMatrix<T> shuffled;
        shuffled.num_rows = mat.num_rows;
        shuffled.num_cols = mat.num_cols;
        shuffled.row_ptr.reserve(mat.num_rows + 1);
        shuffled.col_ind.reserve(mat.col_ind.size());
        shuffled.values.reserve(mat.values.size());

        shuffled.row_ptr.push_back(0);

        for (int i = 0; i < mat.num_rows; ++i) {
            int old_row = perm[i];
            int start = mat.row_ptr[old_row];
            int end = mat.row_ptr[old_row + 1];

            for (int j = start; j < end; ++j) {
                shuffled.col_ind.push_back(mat.col_ind[j]);
                shuffled.values.push_back(mat.values[j]);
            }
            shuffled.row_ptr.push_back(shuffled.col_ind.size());
        }

        mat = std::move(shuffled);
    }

    // -----------------------------------------------------------------
    // Helper to generate a random value (keeps the core logic clean)
    // -----------------------------------------------------------------
    template <typename T>
    T get_val(std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(0.1f, 1.0f);
        return static_cast<T>(dist(rng));
    }

    // -----------------------------------------------------------------
    // 1. Banded & Tridiagonal (bandwidth = 1 is Tridiagonal)
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> generate_banded(int size, int bandwidth, int seed = 42) {
        std::mt19937 rng(seed);
        CSRMatrix<T> mat{size, size, {0}, {}, {}};

        for (int i = 0; i < size; ++i) {
            int start_col = std::max(0, i - bandwidth);
            int end_col = std::min(size - 1, i + bandwidth);
            for (int j = start_col; j <= end_col; ++j) {
                mat.col_ind.push_back(j);
                mat.values.push_back(get_val<T>(rng));
            }
            mat.row_ptr.push_back(mat.col_ind.size());
        }
        return mat;
    }

    // -----------------------------------------------------------------
    // 2. Triangular (Upper or Lower)
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> generate_triangular(int size, bool is_upper, float density, int seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        CSRMatrix<T> mat{size, size, {0}, {}, {}};

        for (int i = 0; i < size; ++i) {
            int start_col = is_upper ? i : 0;
            int end_col = is_upper ? size - 1 : i;
            for (int j = start_col; j <= end_col; ++j) {
                if (prob(rng) < density || i == j) { // Guarantee main diagonal
                    mat.col_ind.push_back(j);
                    mat.values.push_back(get_val<T>(rng));
                }
            }
            mat.row_ptr.push_back(mat.col_ind.size());
        }
        return mat;
    }

    // -----------------------------------------------------------------
    // 3. Hessenberg (Upper: zeros below first subdiagonal)
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> generate_hessenberg(int size, bool is_upper, float density, int seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        CSRMatrix<T> mat{size, size, {0}, {}, {}};

        for (int i = 0; i < size; ++i) {
            int start_col = is_upper ? std::max(0, i - 1) : 0;
            int end_col = is_upper ? size - 1 : std::min(size - 1, i + 1);
            for (int j = start_col; j <= end_col; ++j) {
                if (prob(rng) < density || std::abs(i - j) <= 1) {
                    mat.col_ind.push_back(j);
                    mat.values.push_back(get_val<T>(rng));
                }
            }
            mat.row_ptr.push_back(mat.col_ind.size());
        }
        return mat;
    }

    // -----------------------------------------------------------------
    // 4. Arrowhead & Kite (Dense first 'width' rows/cols + diagonal)
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> generate_arrowhead(int size, int width, int seed = 42) {
        std::mt19937 rng(seed);
        CSRMatrix<T> mat{size, size, {0}, {}, {}};

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i < width || j < width || i == j) {
                    mat.col_ind.push_back(j);
                    mat.values.push_back(get_val<T>(rng));
                }
            }
            mat.row_ptr.push_back(mat.col_ind.size());
        }
        return mat;
    }

    // -----------------------------------------------------------------
    // 5. Checkers Format (Alternating dense blocks of 0s and 1s)
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> generate_checkers(int rows, int cols, int block_size, int seed = 42) {
        std::mt19937 rng(seed);
        CSRMatrix<T> mat{rows, cols, {0}, {}, {}};

        for (int i = 0; i < rows; ++i) {
            int row_block = i / block_size;
            for (int j = 0; j < cols; ++j) {
                int col_block = j / block_size;
                // Alternate based on block grid coordinates
                if ((row_block + col_block) % 2 == 1) {
                    mat.col_ind.push_back(j);
                    mat.values.push_back(get_val<T>(rng));
                }
            }
            mat.row_ptr.push_back(mat.col_ind.size());
        }
        return mat;
    }

    // -----------------------------------------------------------------
    // 6. Block Diagonal
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> generate_block_diagonal(int size, int block_size, float density, int seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        CSRMatrix<T> mat{size, size, {0}, {}, {}};

        for (int i = 0; i < size; ++i) {
            int block_start = (i / block_size) * block_size;
            int block_end = std::min(size - 1, block_start + block_size - 1);

            for (int j = block_start; j <= block_end; ++j) {
                if (prob(rng) < density || i == j) {
                    mat.col_ind.push_back(j);
                    mat.values.push_back(get_val<T>(rng));
                }
            }
            mat.row_ptr.push_back(mat.col_ind.size());
        }
        return mat;
    }

    // -----------------------------------------------------------------
    // 7. Near-Symmetric (Generate symmetric, then randomly drop edges)
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> generate_near_symmetric(int size, float density, float drop_rate, int seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        // Use vector of vectors temporarily to build symmetry easily
        std::vector<std::vector<int>> adj(size);
        for (int i = 0; i < size; ++i) {
            for (int j = i; j < size; ++j) {
                if (prob(rng) < density) {
                    if (prob(rng) > drop_rate) adj[i].push_back(j);
                    if (i != j && prob(rng) > drop_rate) adj[j].push_back(i);
                }
            }
        }

        CSRMatrix<T> mat{size, size, {0}, {}, {}};
        for (int i = 0; i < size; ++i) {
            std::sort(adj[i].begin(), adj[i].end());
            for (int col : adj[i]) {
                mat.col_ind.push_back(col);
                mat.values.push_back(get_val<T>(rng));
            }
            mat.row_ptr.push_back(mat.col_ind.size());
        }
        return mat;
    }

    // -----------------------------------------------------------------
    // 8. Highly Skewed / Power-Law (Scale-Free Graphs)
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> generate_power_law(int rows, int cols, float alpha = 2.5f, int seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.001f, 1.0f);

        CSRMatrix<T> mat{rows, cols, {0}, {}, {}};

        for (int i = 0; i < rows; ++i) {
            // Inverse transform sampling for a simple Pareto/Power-Law distribution
            // Degree is skewed heavily towards lower numbers, with a few massive rows
            float u = prob(rng);
            int degree = static_cast<int>(std::pow(u, -1.0f / (alpha - 1.0f)));
            degree = std::min(degree, cols);
            degree = std::max(1, degree); // Ensure at least 1 connection

            // Pick 'degree' random unique columns
            std::vector<int> random_cols(cols);
            std::iota(random_cols.begin(), random_cols.end(), 0);

            // Partial shuffle just to get 'degree' items (faster than full shuffle)
            for(int k = 0; k < degree; ++k) {
                std::uniform_int_distribution<int> swap_dist(k, cols - 1);
                std::swap(random_cols[k], random_cols[swap_dist(rng)]);
            }

            std::sort(random_cols.begin(), random_cols.begin() + degree);

            for (int k = 0; k < degree; ++k) {
                mat.col_ind.push_back(random_cols[k]);
                mat.values.push_back(get_val<T>(rng));
            }
            mat.row_ptr.push_back(mat.col_ind.size());
        }
        return mat;
    }

    // -----------------------------------------------------------------
    // Generate 1D Dense Matrix (Row-Major)
    // -----------------------------------------------------------------
    template <typename T>
    std::vector<T> generate_dense(int rows, int cols, int seed = 42) {
        std::mt19937 rng(seed);
        std::vector<T> mat(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            mat[i] = get_val<T>(rng);
        }
        return mat;
    }

} // namespace generation
} // namespace matrix_utils