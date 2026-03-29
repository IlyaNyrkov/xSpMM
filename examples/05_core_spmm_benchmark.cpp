#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <cmath> // Added for std::abs

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

int main() {
    std::cout << "==============================================================\n";
    std::cout << "           xSpMM CORE KERNEL PERFORMANCE BENCHMARK            \n";
    std::cout << "==============================================================\n\n";

    // 1. Benchmark Configuration
    IndexType M = 16384;
    IndexType K = 16384;
    IndexType N = 128;
    IndexType hw_block = 16;

    int warmup_iters = 1;
    int test_iters = 10;
    float alpha = 1.0f, beta = 0.0f;

    // The Target Regimes
    struct TestRegime {
        std::string name;
        float density;
    };
    std::vector<TestRegime> regimes = {
        {"Peak Compute (90.0%)", 0.90f},
        {"Peak Compute 2.0 (50.0%)", 0.50f},
        {"Latency Hiding   (5.0%)", 0.05f},
        {"Memory Efficiency (0.5%)", 0.005f},
        {"Memory Efficiency 2.0 (0.1%)", 0.001f},
    };

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

    if (!exec) {
        std::cerr << "[CRITICAL ERROR] Executor is NULL!\n";
        return 1;
    }

    std::cout << "Matrix Shape : Banded\n";
    std::cout << "Dimensions   : M=" << M << ", K=" << K << ", N=" << N << "\n";
    std::cout << "Iterations   : " << warmup_iters << " Warmup / " << test_iters << " Timed\n";
    std::cout << "--------------------------------------------------------------\n\n";

    for (const auto& regime : regimes) {
        // Calculate required bandwidth to hit target density: Bandwidth ≈ (Density * K) / 2
        IndexType bandwidth = static_cast<IndexType>((regime.density * K) / 2.0f);
        if (bandwidth < 1) bandwidth = 1;

        std::vector<IndexType> h_row_ptr, h_col_ind;
        std::vector<ValueType> h_values;
        matrix_utils::generation::generate_banded<ValueType, IndexType>(M, bandwidth, h_row_ptr, h_col_ind, h_values);

        IndexType nnz = h_values.size();
        double sparse_flops = 2.0 * static_cast<double>(nnz) * N;
        double dense_flops  = 2.0 * static_cast<double>(M) * K * N;

        std::cout << ">>> " << regime.name << " | Bandwidth: " << bandwidth << " | NNZ: " << nnz << "\n";

        // 1. Setup Data
        std::vector<ValueType> h_B((size_t)K * N, 1.0f);
        std::vector<ValueType> h_C((size_t)M * N, 0.0f);
        std::vector<ValueType> h_dense_A((size_t)M * K, 0.0f);
        for (IndexType i = 0; i < M; ++i) {
            for (IndexType j = h_row_ptr[i]; j < h_row_ptr[i+1]; ++j) {
                h_dense_A[(size_t)i * K + h_col_ind[j]] = h_values[j];
            }
        }

        xspmm::CSRMatrix<ValueType, IndexType> d_csr(exec, M, K, nnz);
        d_csr.copy_from_host(h_row_ptr.data(), h_col_ind.data(), h_values.data());

        auto d_bcsr = xspmm::csr_to_bcsr(exec, d_csr, hw_block, hw_block);
        IndexType mb = d_bcsr.get_num_block_rows();
        IndexType kb = d_bcsr.get_num_block_cols();
        IndexType nnzb = d_bcsr.get_num_blocks();

        ValueType *d_B = nullptr, *d_C = nullptr, *d_dense_A = nullptr;
        exec->allocate(reinterpret_cast<void**>(&d_B), (size_t)K * N * sizeof(ValueType));
        exec->allocate(reinterpret_cast<void**>(&d_C), (size_t)M * N * sizeof(ValueType));
        exec->allocate(reinterpret_cast<void**>(&d_dense_A), (size_t)M * K * sizeof(ValueType));

        exec->copy_from_host(d_B, h_B.data(), (size_t)K * N * sizeof(ValueType));
        exec->copy_from_host(d_C, h_C.data(), (size_t)M * N * sizeof(ValueType));
        exec->copy_from_host(d_dense_A, h_dense_A.data(), (size_t)M * K * sizeof(ValueType));
        exec->synchronize();

        // =========================================================
        // 0. CORRECTNESS VALIDATION
        // =========================================================
        std::cout << "    Validating Results... ";
        ValueType* d_C_ref = nullptr;
        exec->allocate(reinterpret_cast<void**>(&d_C_ref), (size_t)M * N * sizeof(ValueType));
        exec->copy_from_host(d_C_ref, h_C.data(), (size_t)M * N * sizeof(ValueType));

        // Run our kernel
        xspmm::spmm(d_bcsr, d_B, d_C, N);

        // Run Dense BLAS as the gold standard
#if defined(XSPMM_ENABLE_CUDA)
        cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_dense_A, K, &beta, d_C_ref, N);
#elif defined(XSPMM_ENABLE_HIP)
        rocblas_sgemm(blas_handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, d_B, N, d_dense_A, K, &beta, d_C_ref, N);
#endif
        exec->synchronize();

        std::vector<ValueType> h_C_out((size_t)M * N);
        std::vector<ValueType> h_C_ref_out((size_t)M * N);
        exec->copy_to_host(h_C_out.data(), d_C, (size_t)M * N * sizeof(ValueType));
        exec->copy_to_host(h_C_ref_out.data(), d_C_ref, (size_t)M * N * sizeof(ValueType));
        exec->synchronize();

        int mismatches = 0;
        for (size_t i = 0; i < h_C_out.size(); ++i) {
            // Epsilon check: Allows for standard fp32 accumulation differences
            if (std::abs(h_C_out[i] - h_C_ref_out[i]) > 1e-2f) {
                mismatches++;
            }
        }

        if (mismatches == 0) {
            std::cout << "\033[1;32m[PASS]\033[0m\n";
        } else {
            std::cout << "\033[1;31m[FAILED] (" << mismatches << " mismatches)\033[0m\n";
        }

        // Clean up and reset C for the timing runs
        exec->free(d_C_ref);
        exec->copy_from_host(d_C, h_C.data(), (size_t)M * N * sizeof(ValueType));
        exec->synchronize();
        // =========================================================

        BenchTimer timer;

        // =========================================================
        // A. xSpMM (Core Tier-1 API)
        // =========================================================
        for(int i=0; i<warmup_iters; i++) xspmm::spmm(d_bcsr, d_B, d_C, N);
        exec->synchronize();
        timer.start_timer();
        for(int i=0; i<test_iters; i++) xspmm::spmm(d_bcsr, d_B, d_C, N);
        double time_xspmm = timer.stop_timer() / test_iters;
        double gflops_xspmm = (sparse_flops / 1e6) / time_xspmm;

        // =========================================================
        // B. Vendor CSR
        // =========================================================
        double time_vendor_csr = 0;
#if defined(XSPMM_ENABLE_CUDA)
        cusparseSpMatDescr_t matA_csr; cusparseDnMatDescr_t matB, matC;
        cusparseCreateCsr(&matA_csr, M, K, nnz, (void*)d_csr.get_row_ptr(), (void*)d_csr.get_col_ind(), (void*)d_csr.get_values(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
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
        cusparseDestroySpMat(matA_csr); cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC); exec->free(dbuf_csr);
#elif defined(XSPMM_ENABLE_HIP)
        rocsparse_spmat_descr matA_csr; rocsparse_dnmat_descr matB, matC;
        rocsparse_create_csr_descr(&matA_csr, M, K, nnz, (void*)d_csr.get_row_ptr(), (void*)d_csr.get_col_ind(), (void*)d_csr.get_values(), rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r);
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
        rocsparse_destroy_spmat_descr(matA_csr); rocsparse_destroy_dnmat_descr(matB); rocsparse_destroy_dnmat_descr(matC); exec->free(dbuf_csr);
#endif
        double gflops_vendor_csr = (sparse_flops / 1e6) / time_vendor_csr;

        // =========================================================
        // C. Vendor BSR
        // =========================================================
        double time_vendor_bsr = 0;
#if defined(XSPMM_ENABLE_CUDA)
        cusparseSpMatDescr_t matA_bsr;
        cusparseCreateBsr(&matA_bsr, mb, kb, nnzb, hw_block, hw_block, (void*)d_bcsr.get_bcsr_row_ptr(), (void*)d_bcsr.get_bcsr_col_ind(), (void*)d_bcsr.get_bcsr_values(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        size_t buf_bsr = 0; void* dbuf_bsr = nullptr;
        cusparseSpMM_bufferSize(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_bsr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_bsr);
        exec->allocate(&dbuf_bsr, buf_bsr);
        for(int i=0; i<warmup_iters; i++) cusparseSpMM(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_bsr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dbuf_bsr);
        exec->synchronize();
        timer.start_timer();
        for(int i=0; i<test_iters; i++) cusparseSpMM(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_bsr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dbuf_bsr);
        time_vendor_bsr = timer.stop_timer() / test_iters;
        cusparseDestroySpMat(matA_bsr); exec->free(dbuf_bsr);
#elif defined(XSPMM_ENABLE_HIP)
        rocsparse_spmat_descr matA_bsr;
        rocsparse_create_bsr_descr(&matA_bsr, mb, kb, nnzb, rocsparse_direction_row, hw_block, (void*)d_bcsr.get_bcsr_row_ptr(), (void*)d_bcsr.get_bcsr_col_ind(), (void*)d_bcsr.get_bcsr_values(), rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r);
        size_t buf_bsr = 0; void* dbuf_bsr = nullptr;
        rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_bsr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_buffer_size, &buf_bsr, nullptr);
        exec->allocate(&dbuf_bsr, buf_bsr);
        rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_bsr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_preprocess, &buf_bsr, dbuf_bsr);
        for(int i=0; i<warmup_iters; i++) rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_bsr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute, &buf_bsr, dbuf_bsr);
        exec->synchronize();
        timer.start_timer();
        for(int i=0; i<test_iters; i++) rocsparse_spmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none, &alpha, matA_bsr, matB, &beta, matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute, &buf_bsr, dbuf_bsr);
        time_vendor_bsr = timer.stop_timer() / test_iters;
        rocsparse_destroy_spmat_descr(matA_bsr); exec->free(dbuf_bsr);
#endif
        double gflops_vendor_bsr = (sparse_flops / 1e6) / time_vendor_bsr;

        // =========================================================
        // D. Dense BLAS
        // =========================================================
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
        double gflops_dense = (dense_flops / 1e6) / time_dense;

        // =========================================================
        // Print Metrics
        // =========================================================
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "    xSpMM Core   : " << std::setw(8) << time_xspmm << " ms | " << std::setw(8) << gflops_xspmm << " Eff. GFLOP/s\n";
        std::cout << "    Vendor CSR   : " << std::setw(8) << time_vendor_csr << " ms | " << std::setw(8) << gflops_vendor_csr << " Eff. GFLOP/s\n";
        std::cout << "    Vendor BSR   : " << std::setw(8) << time_vendor_bsr << " ms | " << std::setw(8) << gflops_vendor_bsr << " Eff. GFLOP/s\n";
        std::cout << "    Dense BLAS   : " << std::setw(8) << time_dense << " ms | " << std::setw(8) << gflops_dense << " Dense GFLOP/s\n";
        std::cout << "\n";
        std::cout << "    Speedup vs CSR   : \033[1;32m" << (time_vendor_csr / time_xspmm) << "x\033[0m\n";
        std::cout << "    Speedup vs BSR   : \033[1;32m" << (time_vendor_bsr / time_xspmm) << "x\033[0m\n";
        std::cout << "    Speedup vs Dense : \033[1;32m" << (time_dense / time_xspmm) << "x\033[0m\n";
        std::cout << "--------------------------------------------------------------\n\n";

        exec->free(d_B); exec->free(d_C); exec->free(d_dense_A);
    }

    return 0;
}