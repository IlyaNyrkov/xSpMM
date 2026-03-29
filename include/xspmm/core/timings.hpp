#pragma once

namespace xspmm {
// =================================================================================================
// Struct to hold pipeline benchmarking data
// =================================================================================================
struct SpMMTimings {
    double clustering_ms = 0.0;
    double permutation_ms = 0.0;
    double conversion_ms = 0.0;
    double spmm_ms = 0.0;
    double unpermutation_ms = 0.0;
    double total_pipeline_ms = 0.0;
};
}