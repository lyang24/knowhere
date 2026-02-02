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

// This file is compiled with -mavx2 flag to enable AVX2 intrinsics
// Runtime CPU detection ensures it's only called on CPUs with AVX2 support

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace knowhere::sparse {

// AVX2 SIMD seek: find first position where id >= target
// Returns position or size if not found
// Uses 8-wide vectorization (AVX2 processes 8 x 32-bit integers)
size_t
simd_seek_avx2_impl(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
    constexpr size_t AVX2_WIDTH = 8;
    size_t pos = start_pos;

    // Scalar until aligned to 32-byte boundary
    while (pos < size && (reinterpret_cast<uintptr_t>(&ids[pos]) % 32) != 0) {
        if (ids[pos] >= target) {
            return pos;
        }
        pos++;
    }

    // Broadcast target to all 8 lanes
    __m256i target_vec = _mm256_set1_epi32(static_cast<int32_t>(target));

    // SIMD loop: process 8 elements at a time
    while (pos + AVX2_WIDTH <= size) {
        __m256i id_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ids[pos]));

        // For AVX2: id >= target is equivalent to !(target > id)
        // _mm256_cmpgt_epi32 gives mask where target > id
        __m256i gt_mask = _mm256_cmpgt_epi32(target_vec, id_vec);  // target > id
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(gt_mask));

        if (mask != 0xFF) {
            // At least one element where !(target > id), i.e., id >= target
            // Find first position where id >= target
            int not_mask = ~mask & 0xFF;
            return pos + __builtin_ctz(not_mask);
        }
        pos += AVX2_WIDTH;
    }

    // Scalar tail: handle remaining elements
    while (pos < size) {
        if (ids[pos] >= target) {
            return pos;
        }
        pos++;
    }

    return size;  // Not found
}

}  // namespace knowhere::sparse
