.PHONY: clean help release-cuda debug-cuda release-hip debug-hip

# ==============================================================================
# Build directories
# ==============================================================================
BUILD_CUDA_REL = build_cuda_release
BUILD_HIP_REL  = build_hip_release
BUILD_CUDA_DBG = build_cuda_debug
BUILD_HIP_DBG  = build_hip_debug

# Default target
all: help

# ==============================================================================
# RELEASE BUILDS (Maximum Performance)
# ==============================================================================
release-cuda:
	@echo "Building xSpMM for NVIDIA CUDA (Release)..."
	cmake -S . -B $(BUILD_CUDA_REL) -DCMAKE_BUILD_TYPE=Release -DXSPMM_ENABLE_CUDA=ON -DXSPMM_ENABLE_HIP=OFF
	cmake --build $(BUILD_CUDA_REL) -j

release-hip:
	@echo "=> Building xSpMM for AMD HIP (Release)..."
	# Force CXX to hipcc so standard .cpp files don't get compiled by GCC
	CXX=hipcc cmake -S . -B $(BUILD_HIP_REL) -DCMAKE_BUILD_TYPE=Release -DXSPMM_ENABLE_HIP=ON -DXSPMM_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES="gfx90a"
	cmake --build $(BUILD_HIP_REL) -j

# ==============================================================================
# DEBUG BUILDS (For Development & GDB)
# ==============================================================================
debug-cuda:
	@echo "Building xSpMM for NVIDIA CUDA (Debug)..."
	cmake -S . -B $(BUILD_CUDA_DBG) -DCMAKE_BUILD_TYPE=Debug -DXSPMM_ENABLE_CUDA=ON -DXSPMM_ENABLE_HIP=OFF
	cmake --build $(BUILD_CUDA_DBG) -j

debug-hip:
	@echo "=> Building xSpMM for AMD HIP (Debug)..."
	CXX=hipcc cmake -S . -B $(BUILD_HIP_DBG) -DCMAKE_BUILD_TYPE=Debug -DXSPMM_ENABLE_HIP=ON -DXSPMM_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES="gfx90a"
	cmake --build $(BUILD_HIP_DBG) -j

# ==============================================================================
# UTILITIES
# ==============================================================================
clean:
	@echo "Cleaning all build directories..."
	rm -rf $(BUILD_CUDA_REL) $(BUILD_HIP_REL) $(BUILD_CUDA_DBG) $(BUILD_HIP_DBG)

help:
	@echo "================================================================="
	@echo " xSpMM Build System                                              "
	@echo "================================================================="
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  release-cuda     - Build for CUDA with -O3 optimizations"
	@echo "  debug-cuda       - Build for CUDA with debug symbols (-g)"
	@echo "  release-hip      - Build for HIP with -O3 optimizations (gfx90a)"
	@echo "  debug-hip        - Build for HIP with debug symbols (-g)"
	@echo "  clean            - Remove all build/ directories"
	@echo "================================================================="