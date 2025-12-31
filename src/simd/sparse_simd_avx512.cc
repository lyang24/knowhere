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

// This file is compiled with -mavx512f flag to enable AVX512 intrinsics
// Runtime CPU detection ensures it's only called on CPUs with AVX512 support

#include <immintrin.h>

#include "sparse_simd.h"

namespace knowhere::sparse {

// ============================================================================
// AVX512 SIMD Implementation (16-wide vectorization with hardware scatter)
// ============================================================================
// Accumulates contributions from a single posting list for IP metric
// scores[doc_ids[i]] += q_weight * doc_vals[i] for all i in [0, list_size)
void
accumulate_posting_list_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                                  float* scores) {
    constexpr size_t SIMD_WIDTH = 16;  // AVX512 processes 16 floats
    size_t j = 0;

    // 2x unrolled SIMD loop to hide gather/scatter latency
    for (; j + 2 * SIMD_WIDTH <= list_size; j += 2 * SIMD_WIDTH) {
        // No manual prefetch - random access patterns don't benefit and can pollute cache
        // Hardware prefetchers + AVX512 gather units handle this better

        // Chunk 0: elements [j, j+16)
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));

        // Chunk 1: elements [j+16, j+32)
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);
        __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));

        __m512 q_weight_vec = _mm512_set1_ps(q_weight);

        // Process chunk 0
        __m512 contribution0 = _mm512_mul_ps(vals0, q_weight_vec);
        __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores, sizeof(float));
        __m512 new_scores0 = _mm512_add_ps(current_scores0, contribution0);
        _mm512_i32scatter_ps(scores, doc_ids0, new_scores0, sizeof(float));

        // Process chunk 1
        __m512 contribution1 = _mm512_mul_ps(vals1, q_weight_vec);
        __m512 current_scores1 = _mm512_i32gather_ps(doc_ids1, scores, sizeof(float));
        __m512 new_scores1 = _mm512_add_ps(current_scores1, contribution1);
        _mm512_i32scatter_ps(scores, doc_ids1, new_scores1, sizeof(float));
    }

    // Handle remaining 16-31 elements
    for (; j + SIMD_WIDTH <= list_size; j += SIMD_WIDTH) {
        __m512 vals = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 q_weight_vec = _mm512_set1_ps(q_weight);
        __m512 contribution = _mm512_mul_ps(vals, q_weight_vec);
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_add_ps(current_scores, contribution);
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, sizeof(float));
    }

    // Scalar tail (remaining 0-15 elements)
    for (; j < list_size; ++j) {
        scores[doc_ids[j]] += q_weight * doc_vals[j];
    }
}

}  // namespace knowhere::sparse
