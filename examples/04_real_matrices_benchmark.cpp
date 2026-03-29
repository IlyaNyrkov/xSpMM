#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <filesystem>
#include <chrono>

// Core xSpMM Library
#include <xspmm/core/executor.hpp>
#include <xspmm/matrix/csr.hpp>
#include <xspmm/matrix/bcsr.hpp>
#include <xspmm/matrix/conversion.hpp>
#include <xspmm/clustering.hpp>
#include <xspmm/xspmm.hpp>

// Utilities
#include "matrix_io.hpp"

// Vendor Sparse & Dense Libraries
#if defined(XSPMM_ENABLE_CUDA)
    #include <cusparse.h>
    #include <cublas_v2.h>
#elif defined(XSPMM_ENABLE_HIP)
    #include <rocsparse/rocsparse.h>
    #include <rocblas/rocblas.h>
#endif

namespace fs = std::filesystem;
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
    auto get_str_arg = [&](int argc, char** argv, const std::string& name, const std::string& default_val) {
        for (int i = 1; i < argc - 1; ++i) {
            if (std::string(argv[i]) == name) return std::string(argv[i + 1]);
        }
        return default_val;
    };

    std::string base_path = get_str_arg(argc, argv, "-p", "./test_data_mtx/");

    if (!fs::exists(base_path) || !fs::is_directory(base_path)) {
        std::cerr << "Error: Directory '" << base_path << "' does not exist.\n";
        return 1;
    }

    std::cout << "========================================================\n";
    std::cout << "         xSpMM REAL-WORLD MTX BENCHMARK SUITE           \n";
    std::cout << "========================================================\n\n";

    IndexType hw_block = 16;
    int warmup_iters = 3;
    int test_iters = 10;
    float alpha = 1.0f, beta = 0.0f;
    std::vector<IndexType> target_N_sizes = {1024, 2048, 4096, 8192};

    // Hardware Setup
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

    std::cout << "File,M,K,N,Density(%),NNZ,xSpMM_Total,xSpMM_Math,Vendor_CSR,Vendor_BSR,Dense_BLAS\n";

    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mtx") {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();

            try {
                std::cerr << "\n[Debug] Processing File: " << filename << "\n";

                std::cerr << "  -> Reading MTX from disk...\n";
                auto host_mtx = matrix_utils::io::read_mtx<ValueType>(filepath);
                IndexType M = host_mtx.num_rows;
                IndexType K = host_mtx.num_cols;
                IndexType nnz = host_mtx.values.size();
                float density = (float)nnz / ((float)M * (float)K) * 100.0f;

                // ==============================================================
                // HYPER-GRANULAR TRACING
                // ==============================================================
                std::cerr << "  -> [Validation] Target Dimensions : M=" << M << ", K=" << K << ", NNZ=" << nnz << "\n";
                std::cerr << "  -> [Validation] RowPtr Vector Size: " << host_mtx.row_ptr.size() << " (Expected " << M + 1 << ")\n";
                std::cerr << "  -> [Validation] ColInd Vector Size: " << host_mtx.col_ind.size() << " (Expected " << nnz << ")\n";
                std::cerr << "  -> [Validation] Values Vector Size: " << host_mtx.values.size() << " (Expected " << nnz << ")\n";

                if (host_mtx.row_ptr.size() != (size_t)(M + 1)) {
                    throw std::runtime_error("CRITICAL PARSER ERROR: row_ptr size mismatch!");
                }

                std::cerr << "  -> Allocating Device CSR Matrix...\n";
                xspmm::CSRMatrix<ValueType, IndexType> d_csr_raw(exec, M, K, nnz);

                std::cerr << "  -> Initiating hipMemcpy (Host -> Device)...\n";
                d_csr_raw.copy_from_host(host_mtx.row_ptr.data(), host_mtx.col_ind.data(), host_mtx.values.data());
                exec->synchronize(); // Force crash right here if memory is corrupted!
                std::cerr << "  -> Upload Complete!\n";

                // ==============================================================

                std::cerr << "  -> Running Block Analysis (Before)...\n";
                auto bcsr_before = xspmm::csr_to_bcsr(exec, d_csr_raw, hw_block, hw_block);
                IndexType blocks_before = bcsr_before.get_num_blocks();

                std::cerr << "  -> Running Clustering Heuristic...\n";
                IndexType* d_perm = nullptr;
                exec->allocate(reinterpret_cast<void**>(&d_perm), (size_t)M * sizeof(IndexType));
                xspmm::compute_1d_jaccard_clustering(exec, d_csr_raw, 0.5f, hw_block, d_perm);

                std::cerr << "  -> Applying Permutation...\n";
                auto d_csr_opt = xspmm::apply_permutation(exec, d_csr_raw, d_perm);

                std::cerr << "  -> Running Block Analysis (After)...\n";
                auto bcsr_after = xspmm::csr_to_bcsr(exec, d_csr_opt, hw_block, hw_block);
                IndexType blocks_after = bcsr_after.get_num_blocks();
                exec->free(d_perm);

                int blocks_saved = blocks_before - blocks_after;
                float reduction_pct = (float)blocks_saved / blocks_before * 100.0f;

                std::cerr << ">>> Loaded: " << filename << " (" << M << "x" << K << ")\n";
                std::cerr << "    Blocks Before : " << blocks_before << "\n";
                std::cerr << "    Blocks After  : " << blocks_after << "\n";
                std::cerr << "    Reduction     : " << blocks_saved << " blocks (" << std::fixed << std::setprecision(2) << reduction_pct << "%)\n";

                std::vector<IndexType> safe_N_values;
                size_t max_vram_for_dense = 6ULL * 1024 * 1024 * 1024; // 6 GB
                for (IndexType n : target_N_sizes) {
                    if (((size_t)M + (size_t)K) * (size_t)n * sizeof(ValueType) < max_vram_for_dense) {
                        safe_N_values.push_back(n);
                    }
                }
                if (safe_N_values.empty()) safe_N_values.push_back(256);

                std::cerr << "  -> Determining Dense BLAS safety...\n";
                bool run_dense = ((size_t)M * (size_t)K * sizeof(ValueType)) < (2ULL * 1024 * 1024 * 1024);
                std::vector<ValueType> h_dense_A;
                if (run_dense) {
                    h_dense_A.resize((size_t)M * (size_t)K, 0.0f); // Fixed overflow!
                    for (IndexType i = 0; i < M; ++i) {
                        for (IndexType j = host_mtx.row_ptr[i]; j < host_mtx.row_ptr[i+1]; ++j) {
                            h_dense_A[(size_t)i * (size_t)K + (size_t)host_mtx.col_ind[j]] = host_mtx.values[j];
                        }
                    }
                }

                // 5. Execution Loop over safe Ns
                for (IndexType N : safe_N_values) {
                    std::cerr << "  -> Initializing benchmark N=" << N << "...\n";
                    std::vector<ValueType> h_B((size_t)K * (size_t)N, 1.0f);
                    std::vector<ValueType> h_C((size_t)M * (size_t)N, 0.0f);

                    ValueType *d_B = nullptr, *d_C = nullptr, *d_dense_A = nullptr;
                    exec->allocate(reinterpret_cast<void**>(&d_B), (size_t)K * (size_t)N * sizeof(ValueType));
                    exec->allocate(reinterpret_cast<void**>(&d_C), (size_t)M * (size_t)N * sizeof(ValueType));
                    exec->copy_from_host(d_B, h_B.data(), (size_t)K * (size_t)N * sizeof(ValueType));
                    exec->copy_from_host(d_C, h_C.data(), (size_t)M * (size_t)N * sizeof(ValueType));

                    if (run_dense) {
                        exec->allocate(reinterpret_cast<void**>(&d_dense_A), (size_t)M * (size_t)K * sizeof(ValueType));
                        exec->copy_from_host(d_dense_A, h_dense_A.data(), (size_t)M * (size_t)K * sizeof(ValueType));
                    }
                    exec->synchronize();
                    BenchTimer timer;

                    std::cerr << "  -> Running xSpMM Pipeline...\n";
                    xspmm::SpMMTimings xspmm_times;
                    for(int i=0; i<warmup_iters; i++) xspmm::spmm(exec, d_csr_raw, d_B, d_C, N, hw_block, true, nullptr);
                    exec->synchronize();
                    xspmm::spmm(exec, d_csr_raw, d_B, d_C, N, hw_block, true, &xspmm_times);
                    exec->synchronize();

                    std::cerr << "  -> Running Vendor CSR...\n";
                    double time_vendor_csr = 0;
#if defined(XSPMM_ENABLE_CUDA)
                    cusparseSpMatDescr_t matA_csr; cusparseDnMatDescr_t matB, matC;
                    cusparseCreateCsr(&matA_csr, M, K, nnz, (void*)d_csr_raw.get_row_ptr(), (void*)d_csr_raw.get_col_ind(), (void*)d_csr_raw.get_values(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
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
                    cusparseDestroySpMat(matA_csr); cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC);
#elif defined(XSPMM_ENABLE_HIP)
                    rocsparse_spmat_descr matA_csr; rocsparse_dnmat_descr matB, matC;
                    rocsparse_create_csr_descr(&matA_csr, M, K, nnz, (void*)d_csr_raw.get_row_ptr(), (void*)d_csr_raw.get_col_ind(), (void*)d_csr_raw.get_values(), rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r);
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
                    rocsparse_destroy_spmat_descr(matA_csr); rocsparse_destroy_dnmat_descr(matB); rocsparse_destroy_dnmat_descr(matC);
#endif

                    std::cerr << "  -> Running Vendor BSR...\n";
                    double time_vendor_bsr = 0;
                    IndexType mb = bcsr_after.get_num_block_rows();
                    IndexType kb = bcsr_after.get_num_block_cols();
                    IndexType nnzb = bcsr_after.get_num_blocks();

#if defined(XSPMM_ENABLE_CUDA)
                    cusparseSpMatDescr_t matA_bsr;
                    cusparseCreateBsr(&matA_bsr, mb, kb, nnzb, hw_block, hw_block, (void*)bcsr_after.get_row_ptr(), (void*)bcsr_after.get_col_ind(), (void*)bcsr_after.get_values(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
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
                    rocsparse_create_bsr_descr(&matA_bsr, mb, kb, nnzb, rocsparse_direction_row, hw_block, (void*)bcsr_after.get_bcsr_row_ptr(), (void*)bcsr_after.get_bcsr_col_ind(), (void*)bcsr_after.get_bcsr_values(), rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r);
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

                    std::cerr << "  -> Running Dense BLAS (if safe)...\n";
                    double time_dense = 0;
                    if (run_dense) {
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
                    }

                    // Print CSV Row (stdout)
                    std::cout << filename << "," << M << "," << K << "," << N << ","
                              << std::fixed << std::setprecision(4) << density << "," << nnz << ","
                              << std::setprecision(3) << xspmm_times.total_pipeline_ms << ","
                              << xspmm_times.spmm_ms << ","
                              << time_vendor_csr << ","
                              << time_vendor_bsr << ","
                              << (run_dense ? std::to_string(time_dense) : "OOM_SKIPPED") << "\n";

                    std::cerr << "  -> Cleaning up Memory for next N...\n";
                    exec->free(d_B); exec->free(d_C);
                    if (run_dense) exec->free(d_dense_A);
#if defined(XSPMM_ENABLE_CUDA) || defined(XSPMM_ENABLE_HIP)
                    exec->free(dbuf_csr);
#endif
                }
            } catch (const std::exception& e) {
                std::cerr << ">>> Failed to process " << filename << ": " << e.what() << "\n\n";
            }
        }
    }
    return 0;
}