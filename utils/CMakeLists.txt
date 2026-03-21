# utils/CMakeLists.txt

# Create a header-only interface library
add_library(xspmm_utils INTERFACE)

# Expose this directory so dependents can #include "matrix_gen.hpp" directly
target_include_directories(xspmm_utils INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# If your matrix generation uses standard library math features, you might want
# to ensure C++20 is strictly enforced for anyone using these headers
target_compile_features(xspmm_utils INTERFACE cxx_std_20)

# Optional: If your matrix generator utilizes OpenMP for faster CPU-side generation
# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(xspmm_utils INTERFACE OpenMP::OpenMP_CXX)
# endif()