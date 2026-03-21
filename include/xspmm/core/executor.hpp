// include/xspmm/core/executor.hpp
#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace xspmm {

class Executor {
public:
    virtual ~Executor() = default;

    virtual void allocate(void** ptr, size_t bytes) const = 0;
    virtual void free(void* ptr) const = 0;

    virtual void copy_from_host(void* device_dst, const void* host_src, size_t bytes) const = 0;
    virtual void copy_to_host(void* host_dst, const void* device_src, size_t bytes) const = 0;
    virtual void copy_device_to_device(void* dst, const void* src, size_t bytes) const = 0;

    virtual void synchronize() const = 0;
};

class CpuExecutor : public Executor {
public:
    CpuExecutor() = default;
    ~CpuExecutor() override = default;

    void allocate(void** ptr, size_t bytes) const override;
    void free(void* ptr) const override;
    void copy_from_host(void* device_dst, const void* host_src, size_t bytes) const override;
    void copy_to_host(void* host_dst, const void* device_src, size_t bytes) const override;
    void copy_device_to_device(void* dst, const void* src, size_t bytes) const override;
    void synchronize() const override;
};

class CudaExecutor : public Executor {
public:
    explicit CudaExecutor(int device_id = 0);
    ~CudaExecutor() override;

    void allocate(void** ptr, size_t bytes) const override;
    void free(void* ptr) const override;
    void copy_from_host(void* device_dst, const void* host_src, size_t bytes) const override;
    void copy_to_host(void* host_dst, const void* device_src, size_t bytes) const override;
    void copy_device_to_device(void* dst, const void* src, size_t bytes) const override;
    void synchronize() const override;

private:
    int device_id_;
};

class HipExecutor : public Executor {
public:
    explicit HipExecutor(int device_id = 0);
    ~HipExecutor() override;

    void allocate(void** ptr, size_t bytes) const override;
    void free(void* ptr) const override;
    void copy_from_host(void* device_dst, const void* host_src, size_t bytes) const override;
    void copy_to_host(void* host_dst, const void* device_src, size_t bytes) const override;
    void copy_device_to_device(void* dst, const void* src, size_t bytes) const override;
    void synchronize() const override;

private:
    int device_id_;
};

} // namespace xspmm