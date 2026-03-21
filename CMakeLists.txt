cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(xSpMM LANGUAGES CXX)

# Enforce C++20 for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Hardware backend toggles
option(XSPMM_ENABLE_CUDA "Build the NVIDIA CUDA backend" OFF)
option(XSPMM_ENABLE_HIP  "Build the AMD HIP backend" OFF)

if(NOT XSPMM_ENABLE_CUDA AND NOT XSPMM_ENABLE_HIP)
    message(WARNING "Neither CUDA nor HIP backend enabled. Only CPU operations will be available.")
endif()

# ------------------------------------------------------------------------------
# Backend Setup
# ------------------------------------------------------------------------------
if(XSPMM_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Target the native architecture of the machine compiling the code
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    message(STATUS "xSpMM: CUDA backend ENABLED (Arch: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

if(XSPMM_ENABLE_HIP)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    # Target the native architecture of the machine compiling the code (e.g., gfx90a)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "native")
    endif()
    message(STATUS "xSpMM: HIP backend ENABLED (Arch: ${CMAKE_HIP_ARCHITECTURES})")
endif()

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------
# We only add the source directory for now.
# Tests, examples, and benchmarks will be added later.
add_subdirectory(src)

add_subdirectory(utils)

add_subdirectory(examples)