// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// Benchmark for the Window Switch optimization in sparse inverted index.
// The Window Switch strategy (SINDI paper, VLDB 2026) improves cache locality by
// processing posting lists in fixed-size windows instead of random writes.
//
// This benchmark compares:
// 1. Direct (non-windowed) processing - random writes across entire score array
// 2. Windowed processing - cache-friendly writes within fixed-size windows
//
// Expected results: 2-4x speedup on million-scale datasets due to reduced L3 cache misses

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include "simd/cache_info.h"
#include "simd/sparse_simd.h"

using namespace knowhere::sparse;
using knowhere::CacheInfo;

// ============================================================================
// Data Generation
// ============================================================================

// Generate synthetic sparse posting lists with realistic distributions
struct SparsePostingLists {
    size_t n_docs;
    size_t vocab_size;
    std::vector<std::vector<uint32_t>> posting_list_ids;
    std::vector<std::vector<float>> posting_list_vals;

    // Query: vector of (term_index, weight) pairs
    std::vector<std::pair<size_t, float>> query;

    SparsePostingLists(size_t n_docs, size_t vocab_size, size_t avg_query_terms, size_t avg_posting_len, int seed = 42)
        : n_docs(n_docs), vocab_size(vocab_size) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);
        std::uniform_int_distribution<size_t> term_dist(0, vocab_size - 1);

        posting_list_ids.resize(vocab_size);
        posting_list_vals.resize(vocab_size);

        // Zipf distribution for posting list lengths (realistic for IR)
        double alpha = 1.0;
        double sum = 0.0;
        for (size_t r = 1; r <= vocab_size; ++r) {
            sum += 1.0 / std::pow(r, alpha);
        }
        double scale = avg_posting_len * vocab_size / sum;

        for (size_t term_id = 0; term_id < vocab_size; ++term_id) {
            size_t rank = term_id + 1;
            double freq = scale / std::pow(rank, alpha);
            size_t target_len = std::min(n_docs, std::max(size_t(1), static_cast<size_t>(freq)));

            std::unordered_set<uint32_t> seen;
            std::uniform_int_distribution<uint32_t> doc_dist(0, n_docs - 1);

            while (posting_list_ids[term_id].size() < target_len) {
                uint32_t doc_id = doc_dist(rng);
                if (seen.insert(doc_id).second) {
                    posting_list_ids[term_id].push_back(doc_id);
                }
            }

            // Sort for cache-friendly access (required for Window Switch)
            std::sort(posting_list_ids[term_id].begin(), posting_list_ids[term_id].end());

            posting_list_vals[term_id].resize(posting_list_ids[term_id].size());
            for (size_t i = 0; i < posting_list_vals[term_id].size(); ++i) {
                posting_list_vals[term_id][i] = val_dist(rng);
            }
        }

        // Generate query with heavy terms (first terms have longest posting lists)
        size_t heavy_terms = std::min(size_t(15), vocab_size);
        for (size_t i = 0; i < heavy_terms; ++i) {
            query.push_back({i, val_dist(rng)});
        }
        // Add some random terms
        for (size_t i = heavy_terms; i < avg_query_terms; ++i) {
            query.push_back({term_dist(rng), val_dist(rng)});
        }
    }

    size_t
    total_postings() const {
        size_t total = 0;
        for (const auto& plist : posting_list_ids) {
            total += plist.size();
        }
        return total;
    }

    void
    print_stats() const {
        size_t max_len = 0, min_len = SIZE_MAX;
        size_t total = 0;
        for (const auto& plist : posting_list_ids) {
            if (!plist.empty()) {
                total += plist.size();
                max_len = std::max(max_len, plist.size());
                min_len = std::min(min_len, plist.size());
            }
        }
        printf("  Posting lists: vocab=%zu, total_postings=%zu\n", vocab_size, total);
        printf("  Length range: [%zu, %zu], query_terms=%zu\n", min_len, max_len, query.size());
    }
};

// ============================================================================
// Benchmark Implementations
// ============================================================================

class Timer {
    std::chrono::high_resolution_clock::time_point start_;

 public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {
    }

    double
    elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    void
    reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

// Direct (non-windowed) processing - baseline
void
compute_all_distances_direct(const SparsePostingLists& data, std::vector<float>& scores) {
    scores.assign(data.n_docs, 0.0f);

    for (const auto& [dim_idx, q_weight] : data.query) {
        const auto& plist_ids = data.posting_list_ids[dim_idx];
        const auto& plist_vals = data.posting_list_vals[dim_idx];

        accumulate_posting_list_contribution_ip_dispatch<float>(plist_ids.data(), plist_vals.data(), plist_ids.size(),
                                                                q_weight, scores.data());
    }
}

// Window Switch processing - cache-friendly
// Use detected cache size, with fallback to 100K
static size_t
GetWindowSize() {
    static const size_t size = []() {
        size_t recommended = CacheInfo::GetInstance().RecommendedWindowSize();
        // Clamp to reasonable bounds
        return std::max(size_t(50000), std::min(size_t(2000000), recommended));
    }();
    return size;
}

static const size_t kWindowSize = GetWindowSize();

void
compute_all_distances_windowed(const SparsePostingLists& data, std::vector<float>& scores) {
    scores.assign(data.n_docs, 0.0f);

    const size_t num_windows = (data.n_docs + kWindowSize - 1) / kWindowSize;
    std::vector<size_t> plist_positions(data.query.size(), 0);

    for (size_t w = 0; w < num_windows; ++w) {
        const uint32_t window_start = static_cast<uint32_t>(w * kWindowSize);
        const uint32_t window_end = static_cast<uint32_t>(std::min((w + 1) * kWindowSize, data.n_docs));

        for (size_t q_idx = 0; q_idx < data.query.size(); ++q_idx) {
            const auto& [dim_idx, q_weight] = data.query[q_idx];
            const auto& plist_ids = data.posting_list_ids[dim_idx];
            const auto& plist_vals = data.posting_list_vals[dim_idx];

            if (plist_ids.empty()) {
                continue;
            }

            size_t start_pos = plist_positions[q_idx];
            if (start_pos >= plist_ids.size()) {
                continue;
            }

            // Advance to window start
            while (start_pos < plist_ids.size() && plist_ids[start_pos] < window_start) {
                ++start_pos;
            }

            // Find end of entries in this window
            size_t end_pos = start_pos;
            while (end_pos < plist_ids.size() && plist_ids[end_pos] < window_end) {
                ++end_pos;
            }

            if (end_pos > start_pos) {
                accumulate_posting_list_contribution_ip_dispatch<float>(plist_ids.data() + start_pos,
                                                                        plist_vals.data() + start_pos,
                                                                        end_pos - start_pos, q_weight, scores.data());
            }

            plist_positions[q_idx] = end_pos;
        }
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

struct BenchmarkResult {
    double direct_time_ms;
    double windowed_time_ms;
    double speedup;
    bool correct;
};

BenchmarkResult
run_benchmark(const SparsePostingLists& data, int warmup_runs = 3, int bench_runs = 10) {
    std::vector<float> result_direct, result_windowed;
    BenchmarkResult result;

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        compute_all_distances_direct(data, result_direct);
        compute_all_distances_windowed(data, result_windowed);
    }

    // Benchmark direct
    Timer timer;
    for (int i = 0; i < bench_runs; ++i) {
        compute_all_distances_direct(data, result_direct);
    }
    result.direct_time_ms = timer.elapsed_ms() / bench_runs;

    // Benchmark windowed
    timer.reset();
    for (int i = 0; i < bench_runs; ++i) {
        compute_all_distances_windowed(data, result_windowed);
    }
    result.windowed_time_ms = timer.elapsed_ms() / bench_runs;

    result.speedup = result.direct_time_ms / result.windowed_time_ms;

    // Verify correctness
    result.correct = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < result_direct.size(); ++i) {
        double diff = std::abs(result_direct[i] - result_windowed[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-3) {
            result.correct = false;
        }
    }

    return result;
}

int
main() {
    printf("==============================================================================\n");
    printf("  Sparse Inverted Index Window Switch Benchmark\n");
    printf("  (SINDI Paper VLDB 2026 - Cache-Aware Processing Optimization)\n");
    printf("==============================================================================\n\n");

    // Print detected cache info
    auto& cache_info = CacheInfo::GetInstance();
    printf("Detected CPU Cache Sizes:\n");
    if (cache_info.L1DataCacheSize() > 0) {
        printf("  L1 Data Cache: %zu KB\n", cache_info.L1DataCacheSize() / 1024);
    }
    if (cache_info.L2CacheSize() > 0) {
        printf("  L2 Cache:      %zu KB\n", cache_info.L2CacheSize() / 1024);
    }
    if (cache_info.L3CacheSize() > 0) {
        printf("  L3 Cache:      %zu KB (%.1f MB)\n", cache_info.L3CacheSize() / 1024,
               cache_info.L3CacheSize() / (1024.0 * 1024.0));
    } else {
        printf("  L3 Cache:      Not detected (using default)\n");
    }
    printf("\n");

    printf("Window Configuration:\n");
    printf("  Window size:   %zu entries (%.1f MB for floats)\n", kWindowSize,
           kWindowSize * sizeof(float) / (1024.0 * 1024.0));
    printf("  Strategy:      Use ~50%% of L3 cache for scores array\n\n");

    // Test configurations at different scales
    struct BenchConfig {
        const char* name;
        size_t n_docs;
        size_t vocab_size;
        size_t avg_query_terms;
        size_t avg_posting_len;
    };

    std::vector<BenchConfig> configs = {
        // Below windowing threshold (baseline)
        {"Small (100K docs)", 100000, 5000, 20, 100},

        // At threshold boundary
        {"Medium (200K docs)", 200000, 8000, 25, 150},

        // Above threshold - Window Switch should help
        {"Large (500K docs)", 500000, 10000, 30, 200},

        // Million scale - maximum benefit expected
        {"Very Large (1M docs)", 1000000, 15000, 30, 300},

        // Multi-million - stress test
        {"Huge (2M docs)", 2000000, 20000, 35, 400},
    };

    printf("%-25s %12s %12s %10s %10s\n", "Configuration", "Direct (ms)", "Window (ms)", "Speedup", "Correct");
    printf("------------------------------------------------------------------------------\n");

    for (const auto& config : configs) {
        SparsePostingLists data(config.n_docs, config.vocab_size, config.avg_query_terms, config.avg_posting_len);

        auto result = run_benchmark(data);

        printf("%-25s %12.2f %12.2f %9.2fx %10s\n", config.name, result.direct_time_ms, result.windowed_time_ms,
               result.speedup, result.correct ? "PASS" : "FAIL");
    }

    printf("\n==============================================================================\n");
    printf("  Detailed Analysis for Million-Scale Dataset\n");
    printf("==============================================================================\n\n");

    // Detailed analysis at million scale
    SparsePostingLists million_data(1000000, 15000, 30, 300, 12345);
    million_data.print_stats();

    printf("\n  Running detailed benchmark (20 runs)...\n");
    auto detail_result = run_benchmark(million_data, 5, 20);

    printf("\n  Results:\n");
    printf("    Direct processing:   %.2f ms\n", detail_result.direct_time_ms);
    printf("    Windowed processing: %.2f ms\n", detail_result.windowed_time_ms);
    printf("    Speedup:             %.2fx\n", detail_result.speedup);
    printf("    Correctness:         %s\n", detail_result.correct ? "VERIFIED" : "FAILED");

    printf("\n  Analysis:\n");
    if (detail_result.speedup > 1.5) {
        printf("    Window Switch provides %.1fx speedup due to improved cache locality.\n", detail_result.speedup);
        printf("    The optimization is working as expected for large datasets.\n");
    } else if (detail_result.speedup > 1.0) {
        printf("    Window Switch provides modest %.1fx speedup.\n", detail_result.speedup);
        printf("    Benefits may vary based on CPU cache size and posting list distribution.\n");
    } else {
        printf("    Window Switch overhead exceeds benefit at this scale.\n");
        printf("    Consider adjusting threshold or window size for your workload.\n");
    }

    printf("\n==============================================================================\n");
    printf("  Benchmark completed\n");
    printf("==============================================================================\n");

    return 0;
}
