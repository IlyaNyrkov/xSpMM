// src/core/executor.cpp
#include "xspmm/core/executor.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef XSPMM_ENABLE_CUDA
#include <cuda_runtime.h>

#define XSPMM_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + \
                                     cudaGetErrorString(err) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

#endif

#ifdef XSPMM_ENABLE_HIP
#include <hip/hip_runtime.h>

#define XSPMM_HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("HIP Error: ") + \
                                     hipGetErrorString(err) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

#endif

namespace xspmm {

// =================================================================================================
// CPU EXECUTOR IMPLEMENTATION
// =================================================================================================

constexpr size_t ALIGNMENT = 64;

void CpuExecutor::allocate(void** ptr, size_t bytes) const {
    size_t aligned_bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    *ptr = std::aligned_alloc(ALIGNMENT, aligned_bytes);
    if (*ptr == nullptr) {
        throw std::bad_alloc();
    }
}

void CpuExecutor::free(void* ptr) const {
    std::free(ptr);
}

void CpuExecutor::copy_from_host(void* device_dst, const void* host_src, size_t bytes) const {
    std::memcpy(device_dst, host_src, bytes);
}

void CpuExecutor::copy_to_host(void* host_dst, const void* device_src, size_t bytes) const {
    std::memcpy(host_dst, device_src, bytes);
}

void CpuExecutor::copy_device_to_device(void* dst, const void* src, size_t bytes) const {
    std::memcpy(dst, src, bytes);
}

void CpuExecutor::synchronize() const {
    // No-op for CPU
}

// =================================================================================================
// CUDA EXECUTOR IMPLEMENTATION
// =================================================================================================

#ifdef XSPMM_ENABLE_CUDA

CudaExecutor::CudaExecutor(int device_id) : device_id_(device_id) {
    XSPMM_CUDA_CHECK(cudaSetDevice(device_id_));
}

CudaExecutor::~CudaExecutor() = default;

void CudaExecutor::allocate(void** ptr, size_t bytes) const {
    XSPMM_CUDA_CHECK(cudaMalloc(ptr, bytes));
}

void CudaExecutor::free(void* ptr) const {
    XSPMM_CUDA_CHECK(cudaFree(ptr));
}

void CudaExecutor::copy_from_host(void* device_dst, const void* host_src, size_t bytes) const {
    XSPMM_CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice));
}

void CudaExecutor::copy_to_host(void* host_dst, const void* device_src, size_t bytes) const {
    XSPMM_CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost));
}

void CudaExecutor::copy_device_to_device(void* dst, const void* src, size_t bytes) const {
    XSPMM_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

void CudaExecutor::synchronize() const {
    XSPMM_CUDA_CHECK(cudaDeviceSynchronize());
}

#else // Stubs if CUDA is not enabled

CudaExecutor::CudaExecutor(int) { throw std::runtime_error("xSpMM was not compiled with CUDA support."); }
CudaExecutor::~CudaExecutor() = default;
void CudaExecutor::allocate(void**, size_t) const {}
void CudaExecutor::free(void*) const {}
void CudaExecutor::copy_from_host(void*, const void*, size_t) const {}
void CudaExecutor::copy_to_host(void*, const void*, size_t) const {}
void CudaExecutor::copy_device_to_device(void*, const void*, size_t) const {}
void CudaExecutor::synchronize() const {}

#endif // XSPMM_ENABLE_CUDA


// =================================================================================================
// HIP EXECUTOR IMPLEMENTATION
// =================================================================================================

#ifdef XSPMM_ENABLE_HIP

HipExecutor::HipExecutor(int device_id) : device_id_(device_id) {
    XSPMM_HIP_CHECK(hipSetDevice(device_id_));
}

HipExecutor::~HipExecutor() = default;

void HipExecutor::allocate(void** ptr, size_t bytes) const {
    XSPMM_HIP_CHECK(hipMalloc(ptr, bytes));
}

void HipExecutor::free(void* ptr) const {
    XSPMM_HIP_CHECK(hipFree(ptr));
}

void HipExecutor::copy_from_host(void* device_dst, const void* host_src, size_t bytes) const {
    XSPMM_HIP_CHECK(hipMemcpy(device_dst, host_src, bytes, hipMemcpyHostToDevice));
}

void HipExecutor::copy_to_host(void* host_dst, const void* device_src, size_t bytes) const {
    XSPMM_HIP_CHECK(hipMemcpy(host_dst, device_src, bytes, hipMemcpyDeviceToHost));
}

void HipExecutor::copy_device_to_device(void* dst, const void* src, size_t bytes) const {
    XSPMM_HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
}

void HipExecutor::synchronize() const {
    XSPMM_HIP_CHECK(hipDeviceSynchronize());
}

#else // Stubs if HIP is not enabled
HipExecutor::HipExecutor(int) { throw std::runtime_error("xSpMM was not compiled with HIP support."); }
HipExecutor::~HipExecutor() = default;
void HipExecutor::allocate(void**, size_t) const {}
void HipExecutor::free(void*) const {}
void HipExecutor::copy_from_host(void*, const void*, size_t) const {}
void HipExecutor::copy_to_host(void*, const void*, size_t) const {}
void HipExecutor::copy_device_to_device(void*, const void*, size_t) const {}
void HipExecutor::synchronize() const {}

#endif // XSPMM_ENABLE_HIP

} // namespace xspmm