// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

// Benchmark for pipelined gather optimization in sparse IP accumulation
//
// Build: g++ -O3 -std=c++17 -mavx512f -o benchmark_pipelined_gather benchmark_pipelined_gather.cpp
// Run: ./benchmark_pipelined_gather

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Original implementation (sequential gather-compute-scatter)
void
accumulate_original(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight, float* scores) {
    constexpr size_t SIMD_WIDTH = 16;
    size_t j = 0;

    __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    // 2x unrolled - sequential: gather0, compute0, scatter0, gather1, compute1, scatter1
    for (; j + 2 * SIMD_WIDTH <= list_size; j += 2 * SIMD_WIDTH) {
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);
        __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));

        // Sequential: gather0 -> compute0 -> scatter0
        __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores, sizeof(float));
        __m512 new_scores0 = _mm512_fmadd_ps(vals0, q_weight_vec, current_scores0);
        _mm512_i32scatter_ps(scores, doc_ids0, new_scores0, sizeof(float));

        // Sequential: gather1 -> compute1 -> scatter1
        __m512 current_scores1 = _mm512_i32gather_ps(doc_ids1, scores, sizeof(float));
        __m512 new_scores1 = _mm512_fmadd_ps(vals1, q_weight_vec, current_scores1);
        _mm512_i32scatter_ps(scores, doc_ids1, new_scores1, sizeof(float));
    }

    for (; j + SIMD_WIDTH <= list_size; j += SIMD_WIDTH) {
        __m512 vals = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, sizeof(float));
    }

    if (j < list_size) {
        __mmask16 mask = (1u << (list_size - j)) - 1;
        __m512 vals = _mm512_maskz_loadu_ps(mask, &doc_vals[j]);
        __m512i doc_ids_vec = _mm512_maskz_loadu_epi32(mask, &doc_ids[j]);
        __m512 current_scores = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_mask_i32scatter_ps(scores, mask, doc_ids_vec, new_scores, sizeof(float));
    }
}

// Pipelined implementation (gather0, gather1, compute0, scatter0, compute1, scatter1)
void
accumulate_pipelined(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight, float* scores) {
    constexpr size_t SIMD_WIDTH = 16;
    size_t j = 0;

    __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    // 2x unrolled - pipelined: gather0, gather1, compute0, scatter0, compute1, scatter1
    for (; j + 2 * SIMD_WIDTH <= list_size; j += 2 * SIMD_WIDTH) {
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);
        __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));

        // Pipelined: issue both gathers early
        __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores, sizeof(float));
        __m512 current_scores1 = _mm512_i32gather_ps(doc_ids1, scores, sizeof(float));

        // Process chunk 0
        __m512 new_scores0 = _mm512_fmadd_ps(vals0, q_weight_vec, current_scores0);
        _mm512_i32scatter_ps(scores, doc_ids0, new_scores0, sizeof(float));

        // Process chunk 1
        __m512 new_scores1 = _mm512_fmadd_ps(vals1, q_weight_vec, current_scores1);
        _mm512_i32scatter_ps(scores, doc_ids1, new_scores1, sizeof(float));
    }

    for (; j + SIMD_WIDTH <= list_size; j += SIMD_WIDTH) {
        __m512 vals = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, sizeof(float));
    }

    if (j < list_size) {
        __mmask16 mask = (1u << (list_size - j)) - 1;
        __m512 vals = _mm512_maskz_loadu_ps(mask, &doc_vals[j]);
        __m512i doc_ids_vec = _mm512_maskz_loadu_epi32(mask, &doc_ids[j]);
        __m512 current_scores = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_mask_i32scatter_ps(scores, mask, doc_ids_vec, new_scores, sizeof(float));
    }
}

// 4x unrolled pipelined version for even more latency hiding
void
accumulate_pipelined_4x(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                        float* scores) {
    constexpr size_t SIMD_WIDTH = 16;
    size_t j = 0;

    __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    // 4x unrolled - pipelined gathers
    for (; j + 4 * SIMD_WIDTH <= list_size; j += 4 * SIMD_WIDTH) {
        // Load all data
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);
        __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));
        __m512 vals2 = _mm512_loadu_ps(&doc_vals[j + 2 * SIMD_WIDTH]);
        __m512i doc_ids2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + 2 * SIMD_WIDTH]));
        __m512 vals3 = _mm512_loadu_ps(&doc_vals[j + 3 * SIMD_WIDTH]);
        __m512i doc_ids3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + 3 * SIMD_WIDTH]));

        // Issue all 4 gathers early
        __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores, sizeof(float));
        __m512 current_scores1 = _mm512_i32gather_ps(doc_ids1, scores, sizeof(float));
        __m512 current_scores2 = _mm512_i32gather_ps(doc_ids2, scores, sizeof(float));
        __m512 current_scores3 = _mm512_i32gather_ps(doc_ids3, scores, sizeof(float));

        // Process all chunks
        __m512 new_scores0 = _mm512_fmadd_ps(vals0, q_weight_vec, current_scores0);
        _mm512_i32scatter_ps(scores, doc_ids0, new_scores0, sizeof(float));

        __m512 new_scores1 = _mm512_fmadd_ps(vals1, q_weight_vec, current_scores1);
        _mm512_i32scatter_ps(scores, doc_ids1, new_scores1, sizeof(float));

        __m512 new_scores2 = _mm512_fmadd_ps(vals2, q_weight_vec, current_scores2);
        _mm512_i32scatter_ps(scores, doc_ids2, new_scores2, sizeof(float));

        __m512 new_scores3 = _mm512_fmadd_ps(vals3, q_weight_vec, current_scores3);
        _mm512_i32scatter_ps(scores, doc_ids3, new_scores3, sizeof(float));
    }

    // Handle remaining with 2x pipelined
    for (; j + 2 * SIMD_WIDTH <= list_size; j += 2 * SIMD_WIDTH) {
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);
        __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));

        __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores, sizeof(float));
        __m512 current_scores1 = _mm512_i32gather_ps(doc_ids1, scores, sizeof(float));

        __m512 new_scores0 = _mm512_fmadd_ps(vals0, q_weight_vec, current_scores0);
        _mm512_i32scatter_ps(scores, doc_ids0, new_scores0, sizeof(float));

        __m512 new_scores1 = _mm512_fmadd_ps(vals1, q_weight_vec, current_scores1);
        _mm512_i32scatter_ps(scores, doc_ids1, new_scores1, sizeof(float));
    }

    for (; j + SIMD_WIDTH <= list_size; j += SIMD_WIDTH) {
        __m512 vals = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, sizeof(float));
    }

    if (j < list_size) {
        __mmask16 mask = (1u << (list_size - j)) - 1;
        __m512 vals = _mm512_maskz_loadu_ps(mask, &doc_vals[j]);
        __m512i doc_ids_vec = _mm512_maskz_loadu_epi32(mask, &doc_ids[j]);
        __m512 current_scores = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_mask_i32scatter_ps(scores, mask, doc_ids_vec, new_scores, sizeof(float));
    }
}

// Scalar reference for correctness check
void
accumulate_scalar(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight, float* scores) {
    for (size_t i = 0; i < list_size; ++i) {
        scores[doc_ids[i]] += q_weight * doc_vals[i];
    }
}

struct TestData {
    std::vector<uint32_t> doc_ids;
    std::vector<float> doc_vals;
    size_t num_docs;
};

TestData
generate_posting_list(size_t list_size, size_t num_docs) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> doc_dist(0, num_docs - 1);
    std::uniform_real_distribution<float> val_dist(0.1f, 10.0f);

    // Generate unique doc_ids (simulating a real posting list)
    std::vector<uint32_t> doc_ids(list_size);
    std::iota(doc_ids.begin(), doc_ids.end(), 0);
    std::shuffle(doc_ids.begin(), doc_ids.end(), gen);
    if (list_size < num_docs) {
        doc_ids.resize(list_size);
    }
    std::sort(doc_ids.begin(), doc_ids.end());

    std::vector<float> doc_vals(list_size);
    for (size_t i = 0; i < list_size; ++i) {
        doc_vals[i] = val_dist(gen);
    }

    return {doc_ids, doc_vals, num_docs};
}

template <typename Func>
double
benchmark(Func func, const TestData& data, float* scores, int iterations) {
    double total_ns = 0;

    for (int iter = 0; iter < iterations; ++iter) {
        std::memset(scores, 0, data.num_docs * sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        func(data.doc_ids.data(), data.doc_vals.data(), data.doc_ids.size(), 1.5f, scores);
        auto end = std::chrono::high_resolution_clock::now();
        total_ns += std::chrono::duration<double, std::nano>(end - start).count();
    }

    return total_ns / iterations;
}

bool
verify_correctness(const TestData& data, size_t num_docs) {
    std::vector<float> scores_ref(num_docs, 0);
    std::vector<float> scores_orig(num_docs, 0);
    std::vector<float> scores_pipe(num_docs, 0);
    std::vector<float> scores_pipe4(num_docs, 0);

    accumulate_scalar(data.doc_ids.data(), data.doc_vals.data(), data.doc_ids.size(), 1.5f, scores_ref.data());
    accumulate_original(data.doc_ids.data(), data.doc_vals.data(), data.doc_ids.size(), 1.5f, scores_orig.data());
    accumulate_pipelined(data.doc_ids.data(), data.doc_vals.data(), data.doc_ids.size(), 1.5f, scores_pipe.data());
    accumulate_pipelined_4x(data.doc_ids.data(), data.doc_vals.data(), data.doc_ids.size(), 1.5f, scores_pipe4.data());

    for (size_t i = 0; i < num_docs; ++i) {
        if (std::abs(scores_ref[i] - scores_orig[i]) > 1e-5 || std::abs(scores_ref[i] - scores_pipe[i]) > 1e-5 ||
            std::abs(scores_ref[i] - scores_pipe4[i]) > 1e-5) {
            std::cerr << "Mismatch at " << i << ": ref=" << scores_ref[i] << " orig=" << scores_orig[i]
                      << " pipe=" << scores_pipe[i] << " pipe4=" << scores_pipe4[i] << "\n";
            return false;
        }
    }
    return true;
}

int
main() {
    std::cout << "=== Benchmark: Pipelined Gather Optimization ===\n\n";

    struct Config {
        size_t list_size;
        size_t num_docs;
        const char* name;
    };

    std::vector<Config> configs = {
        {100, 10000, "Small (100 elements, 10K docs)"},
        {1000, 100000, "Medium (1K elements, 100K docs)"},
        {10000, 1000000, "Large (10K elements, 1M docs)"},
        {100000, 1000000, "XLarge (100K elements, 1M docs)"},
    };

    const int iterations = 1000;

    for (const auto& config : configs) {
        std::cout << "Config: " << config.name << "\n";

        auto data = generate_posting_list(config.list_size, config.num_docs);
        std::vector<float> scores(config.num_docs);

        // Verify correctness
        if (!verify_correctness(data, config.num_docs)) {
            std::cerr << "  ERROR: Correctness check failed!\n";
            continue;
        }

        double orig_ns = benchmark(accumulate_original, data, scores.data(), iterations);
        double pipe_ns = benchmark(accumulate_pipelined, data, scores.data(), iterations);
        double pipe4_ns = benchmark(accumulate_pipelined_4x, data, scores.data(), iterations);

        std::cout << "  Original (2x):     " << orig_ns / 1000.0 << " us\n";
        std::cout << "  Pipelined (2x):    " << pipe_ns / 1000.0 << " us\n";
        std::cout << "  Pipelined (4x):    " << pipe4_ns / 1000.0 << " us\n";
        std::cout << "  Speedup (2x pipe): " << orig_ns / pipe_ns << "x\n";
        std::cout << "  Speedup (4x pipe): " << orig_ns / pipe4_ns << "x\n";
        std::cout << "\n";
    }

    return 0;
}
