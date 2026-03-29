#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>

// Core xSpMM Library
#include <xspmm/core/executor.hpp>
#include <xspmm/matrix/csr.hpp>
#include <xspmm/matrix/bcsr.hpp>
#include <xspmm/matrix/conversion.hpp>
#include <xspmm/xspmm.hpp>
#include "matrix_gen.hpp"

// Vendor Sparse & Dense Libraries
#if defined(XSPMM_ENABLE_CUDA)
    #include <cusparse.h>
    #include <cublas_v2.h>
#elif defined(XSPMM_ENABLE_HIP)
    #include <rocsparse/rocsparse.h>
    #include <rocblas/rocblas.h>
#endif

using ValueType = float;
using IndexType = int32_t;

// Hardware Timer Helper
struct BenchTimer {
#if defined(XSPMM_ENABLE_CUDA)
    cudaEvent_t start, stop;
    BenchTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~BenchTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void start_timer() { cudaEventRecord(start); }
    double stop_timer() {
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
    }
#elif defined(XSPMM_ENABLE_HIP)
    hipEvent_t start, stop;
    BenchTimer() { hipEventCreate(&start); hipEventCreate(&stop); }
    ~BenchTimer() { hipEventDestroy(start); hipEventDestroy(stop); }
    void start_timer() { hipEventRecord(start); }
    double stop_timer() {
        hipEventRecord(stop); hipEventSynchronize(stop);
        float ms; hipEventElapsedTime(&ms, start, stop); return ms;
    }
#endif
};

int main(int argc, char** argv) {
    // 1. Suite Configurations (Fixed A, Sweeping N)
    IndexType M = 16384;
    IndexType K = 16384;
    std::vector<IndexType> N_values = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    std::vector<float> extreme_densities = {0.7f, 0.5f, 0.2f, 0.1f, 0.01f, 0.005f, 0.001f};

    IndexType hw_block = 16;
    int warmup_iters = 3;
    int test_iters = 10;
    float alpha = 1.0f, beta = 0.0f;

    // 2. Hardware Setup
    std::shared_ptr<xspmm::Executor> exec;
#if defined(XSPMM_ENABLE_CUDA)
    exec = std::make_shared<xspmm::CudaExecutor>(0);
    cublasHandle_t blas_handle; cublasCreate(&blas_handle);
    cusparseHandle_t sparse_handle; cusparseCreate(&sparse_handle);
#elif defined(XSPMM_ENABLE_HIP)
    exec = std::make_shared<xspmm::HipExecutor>(0);
    rocblas_handle blas_handle; rocblas_create_handle(&blas_handle);
    rocsparse_handle sparse_handle; rocsparse_create_handle(&sparse_handle);
#endif

    // Print CSV Header
    std::cout << "Pattern,M,K,N,Density(%),NNZ,xSpMM_Total,xSpMM_Clust,xSpMM_Perm,xSpMM_Conv,xSpMM_Math,xSpMM_Unperm,Vendor_CSR,Vendor_BSR,Dense_BLAS\n";

    // =================================================================================
    // CORE KERNEL LAMBDA (Executes 1 Matrix A against 1 Size N)
    // =================================================================================
    auto run_benchmark = [&](const std::string& pattern_name, IndexType N,
                             const std::vector<IndexType>& h_row_ptr,
                             const std::vector<IndexType>& h_col_ind,
                             const std::vector<ValueType>& h_values,
                             const std::vector<ValueType>& h_dense_A)
    {
        IndexType nnz = h_values.size();
        float actual_density = (float)nnz / ((float)M * (float)K) * 100.0f;

        // 1. Setup Dense B and C on Host
        std::vector<ValueType> h_B(K * N, 1.0f);
        std::vector<ValueType> h_C(M * N, 0.0f);

        // 2. Upload to Device
        xspmm::CSRMatrix<ValueType, IndexType> d_csr(exec, M, K, nnz);
        d_csr.copy_from_host(h_row_ptr.data(), h_col_ind.data(), h_values.data());

        ValueType *d_B = nullptr, *d_C = nullptr, *d_dense_A = nullptr;
        exec->allocate(reinterpret_cast<void**>(&d_B), K * N * sizeof(ValueType));
        exec->allocate(reinterpret_cast<void**>(&d_C), M * N * sizeof(ValueType));
        exec->allocate(reinterpret_cast<void**>(&d_dense_A), M * K * sizeof(ValueType));

        exec->copy_from_host(d_B, h_B.data(), K * N * sizeof(ValueType));
        exec->copy_from_host(d_C, h_C.data(), M * N * sizeof(ValueType));
        exec->copy_from_host(d_dense_A, h_dense_A.data(), M * K * sizeof(ValueType));
        exec->synchronize();

        BenchTimer timer;

        // --- A. xSpMM Pipeline ---
        xspmm::SpMMTimings xspmm_times;
        for(int i=0; i<warmup_iters; i++) xspmm::spmm(exec, d_csr, d_B, d_C, N, hw_block, true, nullptr);
        exec->synchronize();
        xspmm::spmm(exec, d_csr, d_B, d_C, N, hw_block, true, &xspmm_times);
        exec->synchronize();

        // --- B. Vendor CSR ---
        double time_vendor_csr = 0;
#if defined(XSPMM_ENABLE_CUDA)
        cusparseSpMatDescr_t matA_csr; cusparseDnMatDescr_t matB, matC;
        cusparseCreateCsr(&matA_csr, M, K, nnz, d_csr.get_row_ptr(), d_csr.get_col_ind(), d_csr.get_values(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnMat(&matB, K, N, N, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&matC, M, N, N, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        size_t buf_csr = 0; void* dbuf_csr = nullptr;
        cusparseSpMM_bufferSize(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_csr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_csr);
        exec->allocate(&dbuf_csr, buf_csr);
        for(int i=0; i<warmup_iters; i++) cusparseSpMM(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_csr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dbuf_csr);
        exec->synchronize();
        timer.start_timer();
        for(int i=0; i<test_iters; i++) cusparseSpMM(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_csr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dbuf_csr);
        time_vendor_csr = timer.stop_timer() / test_iters;
        cusparseDestroySpMat(matA_csr);
#elif defined(XSPMM_ENABLE_HIP)
        rocsparse_spmat_descr matA_csr; rocsparse_dnmat_descr matB, matC;
        rocsparse_create_csr_descr(&matA_csr, M, K, nnz, d_csr.get_row_ptr(), d_csr.get_col_ind(), d_csr.get_values(), rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r);
        rocsparse_create_dnmat_descr(&matB, K, N, N, d_B, rocsparse_datatype_f32_r, rocsparse_order_row);
        rocsparse_create_dnmat_descr(&matC, M, N, N, d_C, rocsparse_datatype_f32_r, rocsparse_order_row);
        size_t buf_csr = 0; void* dbuf_csr = nullptr;
        rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_csr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_buffer_size, &buf_csr, nullptr);
        exec->allocate(&dbuf_csr, buf_csr);
        rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_csr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_preprocess, &buf_csr, dbuf_csr);
        for(int i=0; i<warmup_iters; i++) rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_csr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute, &buf_csr, dbuf_csr);
        exec->synchronize();
        timer.start_timer();
        for(int i=0; i<test_iters; i++) rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_csr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute, &buf_csr, dbuf_csr);
        time_vendor_csr = timer.stop_timer() / test_iters;
        rocsparse_destroy_spmat_descr(matA_csr);
#endif

// --- C. Vendor BSR ---
        double time_vendor_bsr = 0;
        auto d_bcsr = xspmm::csr_to_bcsr(exec, d_csr, hw_block, hw_block);
        IndexType mb = d_bcsr.get_num_block_rows();
        IndexType kb = d_bcsr.get_num_block_cols();
        IndexType nnzb = d_bcsr.get_num_blocks();

#if defined(XSPMM_ENABLE_CUDA)
        cusparseSpMatDescr_t matA_bsr;
        // FIX: cusparseCreateBsr strictly requires block rows/cols (mb, kb), not M, K!
        cusparseCreateBsr(&matA_bsr, mb, kb, nnzb, hw_block, hw_block,
                          (void*)d_bcsr.get_bcsr_row_ptr(), (void*)d_bcsr.get_bcsr_col_ind(), (void*)d_bcsr.get_bcsr_values(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        size_t buf_bsr = 0; void* dbuf_bsr = nullptr;
        cusparseSpMM_bufferSize(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_bsr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_bsr);
        exec->allocate(&dbuf_bsr, buf_bsr);

        for(int i=0; i<warmup_iters; i++) cusparseSpMM(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_bsr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dbuf_bsr);
        exec->synchronize();

        timer.start_timer();
        for(int i=0; i<test_iters; i++) cusparseSpMM(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_bsr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dbuf_bsr);
        time_vendor_bsr = timer.stop_timer() / test_iters;

        cusparseDestroySpMat(matA_bsr);
        exec->free(dbuf_bsr);
        cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC);

#elif defined(XSPMM_ENABLE_HIP)
        rocsparse_spmat_descr matA_bsr;

        // FIX: Add rocsparse_direction_row, use a single hw_block, and add (void*) casts!
        rocsparse_create_bsr_descr(&matA_bsr, mb, kb, nnzb,
                                   rocsparse_direction_row, hw_block,
                                   (void*)d_bcsr.get_bcsr_row_ptr(),
                                   (void*)d_bcsr.get_bcsr_col_ind(),
                                   (void*)d_bcsr.get_bcsr_values(),
                                   rocsparse_indextype_i32, rocsparse_indextype_i32,
                                   rocsparse_index_base_zero, rocsparse_datatype_f32_r);

        size_t buf_bsr = 0; void* dbuf_bsr = nullptr;
        rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_bsr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_buffer_size, &buf_bsr, nullptr);
        exec->allocate(&dbuf_bsr, buf_bsr);
        rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_bsr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_preprocess, &buf_bsr, dbuf_bsr);
        exec->synchronize();

        for(int i=0; i<warmup_iters; i++) {
            rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_bsr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute, &buf_bsr, dbuf_bsr);
        }
        exec->synchronize();

        timer.start_timer();
        for(int i=0; i<test_iters; i++) {
            rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_bsr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute, &buf_bsr, dbuf_bsr);
        }
        time_vendor_bsr = timer.stop_timer() / test_iters;

        rocsparse_destroy_spmat_descr(matA_bsr);
        exec->free(dbuf_bsr);
        rocsparse_destroy_dnmat_descr(matB); rocsparse_destroy_dnmat_descr(matC);
#endif

        // --- D. Dense BLAS ---
        double time_dense = 0;
#if defined(XSPMM_ENABLE_CUDA)
        for(int i=0; i<warmup_iters; i++) cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_dense_A, K, &beta, d_C, N);
        exec->synchronize();
        timer.start_timer();
        for(int i=0; i<test_iters; i++) cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_dense_A, K, &beta, d_C, N);
        time_dense = timer.stop_timer() / test_iters;
#elif defined(XSPMM_ENABLE_HIP)
        for(int i=0; i<warmup_iters; i++) rocblas_sgemm(blas_handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, d_B, N, d_dense_A, K, &beta, d_C, N);
        exec->synchronize();
        timer.start_timer();
        for(int i=0; i<test_iters; i++) rocblas_sgemm(blas_handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, d_B, N, d_dense_A, K, &beta, d_C, N);
        time_dense = timer.stop_timer() / test_iters;
#endif

        // Print Row
        std::cout << pattern_name << "," << M << "," << K << "," << N << "," << std::fixed << std::setprecision(4) << actual_density << "," << nnz << ","
                  << std::setprecision(3) << xspmm_times.total_pipeline_ms << ","
                  << xspmm_times.clustering_ms << ","
                  << xspmm_times.permutation_ms << ","
                  << xspmm_times.conversion_ms << ","
                  << xspmm_times.spmm_ms << ","
                  << xspmm_times.unpermutation_ms << ","
                  << time_vendor_csr << ","
                  << time_vendor_bsr << ","
                  << time_dense << "\n";

        // Memory Cleanup
        exec->free(d_B); exec->free(d_C); exec->free(d_dense_A);
#if defined(XSPMM_ENABLE_CUDA) || defined(XSPMM_ENABLE_HIP)
        exec->free(dbuf_csr);
#endif
    };

    // =================================================================================
    // MATRIX GENERATOR LAMBDA (Generates A once, sweeps all N)
    // =================================================================================
    auto evaluate_matrix = [&](const std::string& pattern_name,
                               const std::vector<IndexType>& rp,
                               const std::vector<IndexType>& ci,
                               const std::vector<ValueType>& v)
    {
        // Compute Dense A purely on Host to save redundant work
        std::vector<ValueType> h_dense_A(M * K, 0.0f);
        for (IndexType i = 0; i < M; ++i) {
            for (IndexType j = rp[i]; j < rp[i+1]; ++j) {
                h_dense_A[i * K + ci[j]] = v[j];
            }
        }

        // Loop over N sizes dynamically
        for (IndexType N : N_values) {
            run_benchmark(pattern_name, N, rp, ci, v, h_dense_A);
        }
    };

    // =================================================================================
    // EXECUTE BENCHMARK SUITE
    // =================================================================================
    std::vector<IndexType> rp, ci; std::vector<ValueType> v;

    // 1. Structural Matrices (No density parameter)
    matrix_utils::generation::generate_banded<ValueType, IndexType>(M, 16, rp, ci, v);
    evaluate_matrix("Banded_bw16", rp, ci, v);

    matrix_utils::generation::generate_arrowhead<ValueType, IndexType>(M, 16, rp, ci, v);
    evaluate_matrix("Arrowhead_w16", rp, ci, v);

    matrix_utils::generation::generate_checkers<ValueType, IndexType>(M, M, 16, rp, ci, v);
    evaluate_matrix("Checkers_b16", rp, ci, v);

    matrix_utils::generation::generate_power_law<ValueType, IndexType>(M, M, 2.5f, rp, ci, v);
    evaluate_matrix("PowerLaw_a2.5", rp, ci, v);

    // 2. Density-Based Matrices
    for (float density : extreme_densities) {
        matrix_utils::generation::generate_triangular<ValueType, IndexType>(M, true, density, rp, ci, v);
        evaluate_matrix("UpperTri", rp, ci, v);

        matrix_utils::generation::generate_hessenberg<ValueType, IndexType>(M, true, density, rp, ci, v);
        evaluate_matrix("Hessenberg", rp, ci, v);

        matrix_utils::generation::generate_block_diagonal<ValueType, IndexType>(M, 16, density, rp, ci, v);
        evaluate_matrix("BlockDiag", rp, ci, v);

        matrix_utils::generation::generate_near_symmetric<ValueType, IndexType>(M, density, 0.1f, rp, ci, v);
        evaluate_matrix("NearSymm", rp, ci, v);
    }

    return 0;
}