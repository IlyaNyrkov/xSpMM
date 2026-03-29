#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace matrix_utils {
namespace io {

    // -----------------------------------------------------------------
    // Host-Side CSR Matrix Container (For parsing MTX files into CPU RAM)
    // -----------------------------------------------------------------
    template <typename T>
    struct HostCSRMatrix {
        int num_rows = 0;
        int num_cols = 0;
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        std::vector<T> values;
    };

    // Read MTX file
    template <typename T>
    HostCSRMatrix<T> read_mtx(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open MTX file: " + filename);
        }

        std::string line;
        int rows = 0, cols = 0, nnz = 0;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '%') continue;
            std::istringstream iss(line);
            if (!(iss >> rows >> cols >> nnz)) {
                throw std::runtime_error("Failed to parse MTX dimensions.");
            }
            break;
        }

        struct Element {
            int r, c;
            T v;
            bool operator<(const Element& other) const {
                if (r != other.r) return r < other.r;
                return c < other.c;
            }
        };

        std::vector<Element> elements;
        elements.reserve(nnz);

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '%') continue;

            std::istringstream iss(line);
            int r, c;
            float v = 1.0f;

            // SAFTEY CHECK 1: Ensure we actually read two integers
            if (!(iss >> r >> c)) continue;

            if (!(iss >> v)) { v = 1.0f; }

            // SAFTEY CHECK 2: MTX is 1-indexed. Ignore out-of-bounds garbage.
            if (r < 1 || r > rows || c < 1 || c > cols) continue;

            elements.push_back({r - 1, c - 1, static_cast<T>(v)});
        }

        std::sort(elements.begin(), elements.end());

        HostCSRMatrix<T> csr;
        csr.num_rows = rows;
        csr.num_cols = cols;
        csr.row_ptr.assign(rows + 1, 0);

        // We use elements.size() instead of 'nnz' just in case we skipped garbage lines
        csr.col_ind.reserve(elements.size());
        csr.values.reserve(elements.size());

        for (const auto& el : elements) {
            csr.col_ind.push_back(el.c);
            csr.values.push_back(el.v);
            csr.row_ptr[el.r + 1]++;
        }

        for (int i = 0; i < rows; ++i) {
            csr.row_ptr[i + 1] += csr.row_ptr[i];
        }

        return csr;
    }

    // -----------------------------------------------------------------
    // Helper to print a single value safely
    // -----------------------------------------------------------------
    template <typename T>
    inline void print_val(const T& val, int width = 12) {
        std::cout << std::setw(width) << std::setprecision(4) << std::scientific
                  << static_cast<double>(val) << " ";
    }

    // -----------------------------------------------------------------
    // 3A. Print 1D Dense Matrix (Assuming Row-Major)
    // -----------------------------------------------------------------
    template <typename T>
    void print_dense(const std::vector<T>& mat, int rows, int cols) {
        std::cout << "--- Dense Matrix (" << rows << "x" << cols << ") ---\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                print_val(mat[i * cols + j]);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // -----------------------------------------------------------------
    // 3B. Print Principal Submatrix of a 1D Dense Matrix
    // -----------------------------------------------------------------
    template <typename T>
    void print_dense_submatrix(const std::vector<T>& mat, int total_cols,
                               int start_row, int end_row, int start_col, int end_col) {
        std::cout << "--- Submatrix Rows[" << start_row << ":" << end_row - 1
                  << "] Cols[" << start_col << ":" << end_col - 1 << "] ---\n";

        for (int i = start_row; i < end_row; ++i) {
            for (int j = start_col; j < end_col; ++j) {
                print_val(mat[i * total_cols + j]);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

} // namespace io
} // namespace matrix_utils