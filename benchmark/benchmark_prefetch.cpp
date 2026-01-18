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

/**
 * Benchmark for prefetching optimization in sparse inverted index.
 *
 * This benchmark compares the performance of compute_all_distances with and without
 * prefetching hints. Prefetching helps hide memory latency by loading the next term's
 * posting list data while processing the current term.
 *
 * Build (x86_64):
 *   g++ -O3 -std=c++17 -mavx512f -mavx512dq benchmark_prefetch.cpp -o benchmark_prefetch
 *
 * Or use the Makefile.prefetch if provided.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// Self-contained type definitions (avoid complex include dependencies)
using table_t = uint32_t;

// ============================================================================
// SIMD Accumulation (inlined from sparse_simd.h)
// ============================================================================

#ifdef __AVX512F__
void
accumulate_posting_list_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                                  float* scores) {
    constexpr size_t SIMD_WIDTH = 16;
    size_t j = 0;
    __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    // 2x unrolled SIMD loop
    for (; j + 2 * SIMD_WIDTH <= list_size; j += 2 * SIMD_WIDTH) {
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);
        __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));

        __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores, sizeof(float));
        __m512 new_scores0 = _mm512_fmadd_ps(vals0, q_weight_vec, current_scores0);
        _mm512_i32scatter_ps(scores, doc_ids0, new_scores0, sizeof(float));

        __m512 current_scores1 = _mm512_i32gather_ps(doc_ids1, scores, sizeof(float));
        __m512 new_scores1 = _mm512_fmadd_ps(vals1, q_weight_vec, current_scores1);
        _mm512_i32scatter_ps(scores, doc_ids1, new_scores1, sizeof(float));
    }

    // Handle remaining 16-31 elements
    for (; j + SIMD_WIDTH <= list_size; j += SIMD_WIDTH) {
        __m512 vals = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, sizeof(float));
    }

    // Masked tail
    if (j < list_size) {
        __mmask16 mask = (1u << (list_size - j)) - 1;
        __m512 vals = _mm512_maskz_loadu_ps(mask, &doc_vals[j]);
        __m512i doc_ids_vec = _mm512_maskz_loadu_epi32(mask, &doc_ids[j]);
        __m512 current_scores = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_mask_i32scatter_ps(scores, mask, doc_ids_vec, new_scores, sizeof(float));
    }
}
#endif

inline void
accumulate_posting_list_ip(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                           float* scores) {
#ifdef __AVX512F__
    accumulate_posting_list_ip_avx512(doc_ids, doc_vals, list_size, q_weight, scores);
#else
    for (size_t i = 0; i < list_size; ++i) {
        scores[doc_ids[i]] += q_weight * doc_vals[i];
    }
#endif
}

// ============================================================================
// Simple span-like view (avoid boost dependency)
// ============================================================================

template <typename T>
struct Span {
    const T* data_;
    size_t size_;

    Span() : data_(nullptr), size_(0) {
    }
    Span(const std::vector<T>& v) : data_(v.data()), size_(v.size()) {
    }
    const T*
    data() const {
        return data_;
    }
    size_t
    size() const {
        return size_;
    }
};

// ============================================================================
// Test Data Generation
// ============================================================================

struct SparseDataset {
    size_t n_docs;
    size_t vocab_size;
    std::vector<std::vector<table_t>> posting_list_ids;
    std::vector<std::vector<float>> posting_list_vals;
    std::vector<std::pair<size_t, float>> query;

    // Spans for efficient access (simulating inverted_index_ids_spans_ / vals_spans_)
    std::vector<Span<table_t>> ids_spans;
    std::vector<Span<float>> vals_spans;

    SparseDataset(size_t n_docs, size_t vocab_size, size_t query_terms, size_t avg_posting_len)
        : n_docs(n_docs), vocab_size(vocab_size) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);
        std::uniform_int_distribution<size_t> term_dist(0, vocab_size - 1);

        posting_list_ids.resize(vocab_size);
        posting_list_vals.resize(vocab_size);

        // Zipf distribution for realistic posting list lengths
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

            std::unordered_set<table_t> seen;
            std::uniform_int_distribution<table_t> doc_dist(0, n_docs - 1);

            while (posting_list_ids[term_id].size() < target_len) {
                table_t doc_id = doc_dist(rng);
                if (seen.insert(doc_id).second) {
                    posting_list_ids[term_id].push_back(doc_id);
                }
            }
            std::sort(posting_list_ids[term_id].begin(), posting_list_ids[term_id].end());

            posting_list_vals[term_id].resize(posting_list_ids[term_id].size());
            for (size_t i = 0; i < posting_list_vals[term_id].size(); ++i) {
                posting_list_vals[term_id][i] = val_dist(rng);
            }
        }

        // Build spans
        ids_spans.reserve(vocab_size);
        vals_spans.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            ids_spans.emplace_back(posting_list_ids[i]);
            vals_spans.emplace_back(posting_list_vals[i]);
        }

        // Generate query with heavy terms (to exercise long posting lists)
        size_t heavy_terms = std::min(query_terms, vocab_size);
        for (size_t i = 0; i < heavy_terms; ++i) {
            query.push_back({i, val_dist(rng)});
        }
    }

    void
    print_stats() const {
        size_t total = 0, max_len = 0;
        for (const auto& plist : posting_list_ids) {
            total += plist.size();
            max_len = std::max(max_len, plist.size());
        }
        printf("  Docs: %zu, Vocab: %zu, Query terms: %zu\n", n_docs, vocab_size, query.size());
        printf("  Avg posting len: %.1f, Max posting len: %zu\n", (double)total / vocab_size, max_len);

        // Show lengths of query terms' posting lists
        printf("  Query term posting lengths: ");
        for (size_t i = 0; i < std::min(size_t(10), query.size()); ++i) {
            printf("%zu ", posting_list_ids[query[i].first].size());
        }
        printf("...\n");
    }
};

// ============================================================================
// Timing Utilities
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

// ============================================================================
// Scalar accumulation (like BM25 path or non-AVX512 fallback)
// ============================================================================

inline void
accumulate_posting_list_scalar(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                               float* scores) {
    for (size_t i = 0; i < list_size; ++i) {
        scores[doc_ids[i]] += q_weight * doc_vals[i];
    }
}

// ============================================================================
// SIMD: No Prefetching
// ============================================================================

std::vector<float>
compute_simd_no_prefetch(const SparseDataset& dataset) {
    std::vector<float> scores(dataset.n_docs, 0.0f);

    for (size_t i = 0; i < dataset.query.size(); ++i) {
        auto dim_idx = dataset.query[i].first;
        auto q_weight = dataset.query[i].second;
        const auto& plist_ids = dataset.ids_spans[dim_idx];
        const auto& plist_vals = dataset.vals_spans[dim_idx];

        accumulate_posting_list_ip(plist_ids.data(), plist_vals.data(), plist_ids.size(), q_weight, scores.data());
    }

    return scores;
}

// ============================================================================
// SIMD: With Prefetching
// ============================================================================

std::vector<float>
compute_simd_with_prefetch(const SparseDataset& dataset) {
    std::vector<float> scores(dataset.n_docs, 0.0f);

    for (size_t i = 0; i < dataset.query.size(); ++i) {
        if (i + 1 < dataset.query.size()) {
            auto next_dim = dataset.query[i + 1].first;
            __builtin_prefetch(dataset.ids_spans[next_dim].data(), 0, 3);
            __builtin_prefetch(dataset.vals_spans[next_dim].data(), 0, 3);
        }

        auto dim_idx = dataset.query[i].first;
        auto q_weight = dataset.query[i].second;
        const auto& plist_ids = dataset.ids_spans[dim_idx];
        const auto& plist_vals = dataset.vals_spans[dim_idx];

        accumulate_posting_list_ip(plist_ids.data(), plist_vals.data(), plist_ids.size(), q_weight, scores.data());
    }

    return scores;
}

// ============================================================================
// Scalar: No Prefetching (like BM25 path)
// ============================================================================

std::vector<float>
compute_scalar_no_prefetch(const SparseDataset& dataset) {
    std::vector<float> scores(dataset.n_docs, 0.0f);

    for (size_t i = 0; i < dataset.query.size(); ++i) {
        auto dim_idx = dataset.query[i].first;
        auto q_weight = dataset.query[i].second;
        const auto& plist_ids = dataset.ids_spans[dim_idx];
        const auto& plist_vals = dataset.vals_spans[dim_idx];

        accumulate_posting_list_scalar(plist_ids.data(), plist_vals.data(), plist_ids.size(), q_weight, scores.data());
    }

    return scores;
}

// ============================================================================
// Scalar: With Prefetching (like BM25 path with optimization)
// ============================================================================

std::vector<float>
compute_scalar_with_prefetch(const SparseDataset& dataset) {
    std::vector<float> scores(dataset.n_docs, 0.0f);

    for (size_t i = 0; i < dataset.query.size(); ++i) {
        if (i + 1 < dataset.query.size()) {
            auto next_dim = dataset.query[i + 1].first;
            __builtin_prefetch(dataset.ids_spans[next_dim].data(), 0, 3);
            __builtin_prefetch(dataset.vals_spans[next_dim].data(), 0, 3);
        }

        auto dim_idx = dataset.query[i].first;
        auto q_weight = dataset.query[i].second;
        const auto& plist_ids = dataset.ids_spans[dim_idx];
        const auto& plist_vals = dataset.vals_spans[dim_idx];

        accumulate_posting_list_scalar(plist_ids.data(), plist_vals.data(), plist_ids.size(), q_weight, scores.data());
    }

    return scores;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

void
run_benchmark(const char* name, const SparseDataset& dataset, int warmup_runs, int bench_runs) {
    printf("\n=== %s ===\n", name);
    dataset.print_stats();

    Timer timer;
    std::vector<float> result1, result2;

    // ---- SIMD Path ----
    printf("\n[SIMD Path (IP metric)]\n");

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        result1 = compute_simd_no_prefetch(dataset);
        result2 = compute_simd_with_prefetch(dataset);
    }

    timer.reset();
    for (int i = 0; i < bench_runs; ++i) {
        result1 = compute_simd_no_prefetch(dataset);
    }
    double simd_no_prefetch = timer.elapsed_ms() / bench_runs;

    timer.reset();
    for (int i = 0; i < bench_runs; ++i) {
        result2 = compute_simd_with_prefetch(dataset);
    }
    double simd_with_prefetch = timer.elapsed_ms() / bench_runs;

    printf("  No prefetch:   %.3f ms\n", simd_no_prefetch);
    printf("  With prefetch: %.3f ms\n", simd_with_prefetch);
    printf("  Speedup:       %.2fx\n", simd_no_prefetch / simd_with_prefetch);
    printf("  Improvement:   %.1f%%\n", (1.0 - simd_with_prefetch / simd_no_prefetch) * 100.0);

    // ---- Scalar Path ----
    printf("\n[Scalar Path (BM25 / fallback)]\n");

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        result1 = compute_scalar_no_prefetch(dataset);
        result2 = compute_scalar_with_prefetch(dataset);
    }

    timer.reset();
    for (int i = 0; i < bench_runs; ++i) {
        result1 = compute_scalar_no_prefetch(dataset);
    }
    double scalar_no_prefetch = timer.elapsed_ms() / bench_runs;

    timer.reset();
    for (int i = 0; i < bench_runs; ++i) {
        result2 = compute_scalar_with_prefetch(dataset);
    }
    double scalar_with_prefetch = timer.elapsed_ms() / bench_runs;

    // Verify correctness
    double max_diff = 0.0;
    for (size_t i = 0; i < result1.size(); ++i) {
        double diff = std::abs(result1[i] - result2[i]);
        max_diff = std::max(max_diff, diff);
    }

    printf("  No prefetch:   %.3f ms\n", scalar_no_prefetch);
    printf("  With prefetch: %.3f ms\n", scalar_with_prefetch);
    printf("  Speedup:       %.2fx\n", scalar_no_prefetch / scalar_with_prefetch);
    printf("  Improvement:   %.1f%%\n", (1.0 - scalar_with_prefetch / scalar_no_prefetch) * 100.0);
    printf("  Correctness:   %s\n", max_diff < 1e-6 ? "PASS" : "FAIL");

    printf("==========================================\n");
}

// ============================================================================
// Main
// ============================================================================

int
main() {
    printf("======================================================================\n");
    printf("  Prefetching Optimization Benchmark for Sparse Inverted Index\n");
    printf("======================================================================\n");
    printf("\nThis benchmark measures the impact of prefetching the next term's\n");
    printf("posting list data while processing the current term in TAAT search.\n");

    const int warmup_runs = 5;
    const int bench_runs = 50;

    // Test configurations: (n_docs, vocab_size, query_terms, avg_posting_len)
    struct Config {
        const char* name;
        size_t n_docs;
        size_t vocab_size;
        size_t query_terms;
        size_t avg_posting_len;
    };

    std::vector<Config> configs = {
        // Small dataset - may not benefit much from prefetching (data fits in cache)
        {"Small (50K docs, 20 query terms)", 50000, 2000, 20, 64},

        // Medium dataset - should see some benefit
        {"Medium (200K docs, 25 query terms)", 200000, 5000, 25, 128},

        // Large dataset - more memory pressure, prefetching should help more
        {"Large (500K docs, 30 query terms)", 500000, 8000, 30, 256},

        // Very large dataset - maximum memory pressure
        {"Very Large (1M docs, 30 query terms)", 1000000, 10000, 30, 512},

        // Many query terms - more prefetch opportunities
        {"Many Query Terms (500K docs, 50 terms)", 500000, 10000, 50, 256},

        // Few query terms - less prefetch benefit (fewer iterations)
        {"Few Query Terms (500K docs, 10 terms)", 500000, 10000, 10, 256},

        // Long posting lists - SIMD computation dominates, less relative prefetch benefit
        {"Long Postings (1M docs, avg=2048)", 1000000, 10000, 25, 2048},

        // Short posting lists - memory latency more visible
        {"Short Postings (500K docs, avg=32)", 500000, 10000, 30, 32},
    };

    for (const auto& cfg : configs) {
        SparseDataset dataset(cfg.n_docs, cfg.vocab_size, cfg.query_terms, cfg.avg_posting_len);
        run_benchmark(cfg.name, dataset, warmup_runs, bench_runs);
    }

    printf("\n======================================================================\n");
    printf("  Benchmark completed\n");
    printf("======================================================================\n");

    return 0;
}
