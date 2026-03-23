#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <iomanip>

// Core xSpMM Library
#include <xspmm/core/executor.hpp>
#include <xspmm/matrix/csr.hpp>
#include <xspmm/matrix/bcsr.hpp>
#include <xspmm/matrix/conversion.hpp>
#include <xspmm/clustering.hpp>
#include <xspmm/xspmm.hpp>

// Utilities
#include "matrix_gen.hpp"
#include "matrix_compare.hpp"

// Vendor Sparse Libraries
#if defined(XSPMM_ENABLE_CUDA)
    #include <cusparse.h>
#elif defined(XSPMM_ENABLE_HIP)
    #include <rocsparse/rocsparse.h>
#endif

using ValueType = float;
using IndexType = int32_t;

int main() {
    std::cout << "==================================================\n";
    std::cout << " xSpMM: Debugging Checkers Example\n";
    std::cout << "==================================================\n";

    std::shared_ptr<xspmm::Executor> exec;
#if defined(XSPMM_ENABLE_HIP)
    exec = std::make_shared<xspmm::HipExecutor>(0);
#else
    exec = std::make_shared<xspmm::CpuExecutor>(); // Fallback
#endif

    // 2. Define Matrix Dimensions
    IndexType M = 1024;
    IndexType K = 1024;
    IndexType N = 1024;

    // The visual size of the synthetic dense blocks (can be 4, 8, 16, etc.)
    IndexType pattern_block_size = 4;

    // The required chunk size for the GPU Matrix Cores (Strictly 16 for our kernels)
    IndexType hardware_block_size = 16;

    // 3. Generate Host Data
    std::cout << "[1/7] Generating synthetic 'Checkers' sparse matrix on Host...\n";
    std::vector<IndexType> h_row_ptr;
    std::vector<IndexType> h_col_ind;
    std::vector<ValueType> h_values;

    matrix_utils::generation::generate_checkers<ValueType, IndexType>(
        M, K, pattern_block_size, h_row_ptr, h_col_ind, h_values
    );

    matrix_utils::generation::shuffle_rows<ValueType, IndexType>(
        M, h_row_ptr, h_col_ind, h_values
    );

    IndexType nnz = h_values.size();

    std::vector<ValueType> h_B(K * N, 1.0f);
    std::vector<ValueType> h_C_out_permuted(M * N, 0.0f);
    std::vector<ValueType> h_C_ref(M * N, 0.0f);

    // 4. Upload to Target Executor
    std::cout << "[2/7] Uploading unoptimized CSR to Device...\n";
    xspmm::CSRMatrix<ValueType, IndexType> d_csr_raw(exec, M, K, nnz);
    d_csr_raw.copy_from_host(h_row_ptr.data(), h_col_ind.data(), h_values.data());

    // 5. Run the Clustering Pipeline (MUST use Hardware Block Size!)
    std::cout << "[3/7] Running 1D Jaccard Clustering & Permutation...\n";
    IndexType* d_perm = nullptr;
    exec->allocate(reinterpret_cast<void**>(&d_perm), M * sizeof(IndexType));

    xspmm::compute_1d_jaccard_clustering(exec, d_csr_raw, 0.5f, hardware_block_size, d_perm);
    auto d_csr_opt = xspmm::apply_permutation(exec, d_csr_raw, d_perm);

    // 6. Convert to Hardware-Ready BCSR Format (MUST use Hardware Block Size!)
    std::cout << "[4/7] Converting Optimized CSR to BCSR Format...\n";
    auto d_bcsr = xspmm::csr_to_bcsr(exec, d_csr_opt, hardware_block_size, hardware_block_size);

    // 7. Setup Matrices and FIX MEMORY INITIALIZATION
    ValueType* d_B = nullptr;
    ValueType* d_C = nullptr;
    exec->allocate(reinterpret_cast<void**>(&d_B), K * N * sizeof(ValueType));
    exec->allocate(reinterpret_cast<void**>(&d_C), M * N * sizeof(ValueType));

    exec->copy_from_host(d_B, h_B.data(), K * N * sizeof(ValueType));
    exec->copy_from_host(d_C, h_C_out_permuted.data(), M * N * sizeof(ValueType));

    // 8. Execute xSpMM
    std::cout << "[5/7] Executing xSpMM...\n";
    xspmm::spmm(d_bcsr, d_B, d_C, N);
    exec->synchronize();
    exec->copy_to_host(h_C_out_permuted.data(), d_C, M * N * sizeof(ValueType));

    // 9. Un-permute
    std::vector<IndexType> h_perm(M);
    exec->copy_to_host(h_perm.data(), d_perm, M * sizeof(IndexType));

    std::vector<ValueType> h_C_final(M * N, 0.0f);
    for (IndexType i = 0; i < M; ++i) {
        IndexType original_row = h_perm[i];
        for (IndexType j = 0; j < N; ++j) {
            h_C_final[original_row * N + j] = h_C_out_permuted[i * N + j];
        }
    }

    // 10. Execute rocSPARSE
    std::cout << "[6/7] Computing rocSPARSE Reference...\n";
    ValueType* d_C_ref = nullptr;
    exec->allocate(reinterpret_cast<void**>(&d_C_ref), M * N * sizeof(ValueType));
    exec->copy_from_host(d_C_ref, h_C_ref.data(), M * N * sizeof(ValueType)); // Init to 0

    float alpha = 1.0f;
    float beta = 0.0f;

#if defined(XSPMM_ENABLE_HIP)
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    rocsparse_spmat_descr matA;
    rocsparse_create_csr_descr(&matA, M, K, nnz,
                               d_csr_raw.get_row_ptr(), d_csr_raw.get_col_ind(), d_csr_raw.get_values(),
                               rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r);

    rocsparse_dnmat_descr matB, matC;
    rocsparse_create_dnmat_descr(&matB, K, N, N, d_B, rocsparse_datatype_f32_r, rocsparse_order_row);
    rocsparse_create_dnmat_descr(&matC, M, N, N, d_C_ref, rocsparse_datatype_f32_r, rocsparse_order_row);

    size_t bufferSize = 0;
    rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none,
                   &alpha, matA, matB, &beta, matC, rocsparse_datatype_f32_r,
                   rocsparse_spmm_alg_default, rocsparse_spmm_stage_buffer_size, &bufferSize, nullptr);

    void* dBuffer = nullptr;
    exec->allocate(&dBuffer, bufferSize);

    rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none,
                   &alpha, matA, matB, &beta, matC, rocsparse_datatype_f32_r,
                   rocsparse_spmm_alg_default, rocsparse_spmm_stage_preprocess, &bufferSize, dBuffer);

    rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none,
                   &alpha, matA, matB, &beta, matC, rocsparse_datatype_f32_r,
                   rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute, &bufferSize, dBuffer);

    rocsparse_destroy_spmat_descr(matA);
    rocsparse_destroy_dnmat_descr(matB);
    rocsparse_destroy_dnmat_descr(matC);
    rocsparse_destroy_handle(handle);
    exec->free(dBuffer);
#endif

    exec->synchronize();
    exec->copy_to_host(h_C_ref.data(), d_C_ref, M * N * sizeof(ValueType));

    // 11. Verify & Debug Printer
    std::cout << "[7/7] Verifying Results...\n";

    int mismatch_count = 0;
    for(IndexType i = 0; i < M; ++i) {
        for(IndexType j = 0; j < N; ++j) {
            float val_out = h_C_final[i * N + j];
            float val_ref = h_C_ref[i * N + j];

            if(std::abs(val_out - val_ref) > 1e-3) {
                if (mismatch_count < 20) {
                    std::cout << "  -> Mismatch at (Row " << std::setw(2) << i
                              << ", Col " << std::setw(2) << j << ") | "
                              << "xSpMM: " << std::setw(6) << val_out
                              << " vs rocSPARSE: " << std::setw(6) << val_ref << "\n";
                }
                mismatch_count++;
            }
        }
    }

    if (mismatch_count == 0) {
        std::cout << "\n[SUCCESS] xSpMM Matrix Core Results Match Vendor Reference!\n";
    } else {
        std::cout << "\n[FAILED] Output matrices do not match. Total Mismatches: " << mismatch_count << "\n";
    }

    exec->free(d_perm);
    exec->free(d_B);
    exec->free(d_C);
    exec->free(d_C_ref);

    return 0;
}