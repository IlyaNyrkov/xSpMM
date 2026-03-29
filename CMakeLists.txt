cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(xSpMM LANGUAGES CXX)

# Enforce C++20 for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release build for peak performance if none is specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "xSpMM: Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Hardware backend toggles
option(XSPMM_ENABLE_CUDA "Build the NVIDIA CUDA backend" OFF)
option(XSPMM_ENABLE_HIP  "Build the AMD HIP backend" OFF)

if(NOT XSPMM_ENABLE_CUDA AND NOT XSPMM_ENABLE_HIP)
    message(WARNING "Neither CUDA nor HIP backend enabled. Only CPU operations will be available.")
endif()

# ------------------------------------------------------------------------------
# Global C++ Aggressive Optimizations (CPU Host Code)
# ------------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
    add_compile_options(
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CXX>:-march=native>"
            "$<$<COMPILE_LANGUAGE:CXX>:-ffast-math>"
            "$<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>"
    )
endif()

# ------------------------------------------------------------------------------
# Backend Setup & Device Optimizations
# ------------------------------------------------------------------------------
if(XSPMM_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    message(STATUS "xSpMM: CUDA backend ENABLED (Arch: ${CMAKE_CUDA_ARCHITECTURES})")

    # NVCC Aggressive Optimizations
    if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        add_compile_options(
                "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
                "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"
                "$<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>"
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>"
                # "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>" # Uncomment to see register usage during build
        )
    endif()
endif()

if(XSPMM_ENABLE_HIP)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "native")
    endif()
    message(STATUS "xSpMM: HIP backend ENABLED (Arch: ${CMAKE_HIP_ARCHITECTURES})")

    # HIPCC (Clang) Aggressive Optimizations
    if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        add_compile_options(
                "$<$<COMPILE_LANGUAGE:HIP>:-O3>"
                "$<$<COMPILE_LANGUAGE:HIP>:-ffast-math>"
                "$<$<COMPILE_LANGUAGE:HIP>:-funroll-loops>"
                "$<$<COMPILE_LANGUAGE:HIP>:-munsafe-fp-atomics>"
                "$<$<COMPILE_LANGUAGE:HIP>:-fgpu-flush-denormals-to-zero>"
        )
    endif()
endif()

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(utils)
add_subdirectory(examples)