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

#ifndef SPARSE_INVERTED_INDEX_SIMD_H
#define SPARSE_INVERTED_INDEX_SIMD_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace knowhere::sparse {

// =============================================================================
// SIMD Constants
// =============================================================================
constexpr size_t SIMD_ALIGNMENT = 64;   // AVX512 alignment
constexpr size_t AVX512_WIDTH = 16;     // 16 floats per AVX512 register
constexpr size_t AVX2_WIDTH = 8;        // 8 floats per AVX2 register
constexpr size_t CACHE_LINE_SIZE = 64;  // Typical cache line size

// Window size: 64K docs -> distance array = 256KB, fits in L2/L3 cache
constexpr size_t DEFAULT_WINDOW_SIZE = 65536;

// =============================================================================
// Aligned Memory Allocation
// =============================================================================

// Aligned allocator for STL containers
template <typename T, size_t Alignment = SIMD_ALIGNMENT>
class AlignedAllocator {
 public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {
    }

    pointer
    allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            ptr = nullptr;
        }
#endif
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }

    void
    deallocate(pointer p, size_type) noexcept {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool
    operator==(const AlignedAllocator&) const noexcept {
        return true;
    }

    bool
    operator!=(const AlignedAllocator&) const noexcept {
        return false;
    }
};

// Aligned vector type alias
template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, SIMD_ALIGNMENT>>;

// =============================================================================
// SIMD-Friendly Posting List
// =============================================================================

// A posting list optimized for SIMD operations:
// - Aligned memory for efficient SIMD loads
// - Padded to SIMD width for safe vectorized access
// - Stores doc IDs and values in separate contiguous arrays (SoA)
template <typename IdType = uint32_t, typename ValType = float>
struct SIMDPostingList {
    AlignedVector<IdType> ids;    // Document IDs, aligned and padded
    AlignedVector<ValType> vals;  // Values, aligned and padded
    size_t actual_size = 0;       // Actual number of elements (before padding)
    ValType max_val = 0;          // Maximum value in this posting list

    SIMDPostingList() = default;

    // Reserve capacity with padding
    void
    reserve(size_t capacity, size_t simd_width = AVX512_WIDTH) {
        size_t padded_capacity = ((capacity + simd_width - 1) / simd_width) * simd_width;
        ids.reserve(padded_capacity);
        vals.reserve(padded_capacity);
    }

    // Add an element
    void
    push_back(IdType id, ValType val) {
        ids.push_back(id);
        vals.push_back(val);
        actual_size++;
        if (val > max_val) {
            max_val = val;
        }
    }

    // Pad to SIMD width with zeros (call after all elements are added)
    void
    finalize(size_t simd_width = AVX512_WIDTH) {
        size_t padded_size = ((actual_size + simd_width - 1) / simd_width) * simd_width;
        // Pad with sentinel values (max doc_id and zero value)
        while (ids.size() < padded_size) {
            ids.push_back(std::numeric_limits<IdType>::max());
            vals.push_back(static_cast<ValType>(0));
        }
    }

    // Get padded size (safe for SIMD operations)
    size_t
    padded_size() const {
        return ids.size();
    }

    // Get actual size
    size_t
    size() const {
        return actual_size;
    }

    // Check if empty
    bool
    empty() const {
        return actual_size == 0;
    }

    // Get aligned pointers for SIMD operations
    const IdType*
    ids_data() const {
        return ids.data();
    }

    const ValType*
    vals_data() const {
        return vals.data();
    }
};

// =============================================================================
// SIMD-Accelerated Operations
// =============================================================================

// Scalar seek: baseline implementation
inline size_t
scalar_seek(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
    for (size_t pos = start_pos; pos < size; pos++) {
        if (ids[pos] >= target) {
            return pos;
        }
    }
    return size;
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)

// SIMD seek: find first position where id >= target
// Returns position or size if not found
// Requires: ids array is sorted and padded to AVX512_WIDTH
inline size_t
simd_seek_avx512(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
    // Skip to aligned position
    size_t pos = start_pos;

    // Scalar until aligned or found
    while (pos < size && (reinterpret_cast<uintptr_t>(&ids[pos]) % SIMD_ALIGNMENT) != 0) {
        if (ids[pos] >= target) {
            return pos;
        }
        pos++;
    }

    // SIMD search
    __m512i target_vec = _mm512_set1_epi32(static_cast<int32_t>(target));

    while (pos + AVX512_WIDTH <= size) {
        __m512i id_vec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&ids[pos]));
        // Compare: mask where id >= target (equivalent to !(id < target))
        __mmask16 ge_mask = _mm512_cmpge_epi32_mask(id_vec, target_vec);

        if (ge_mask != 0) {
            // Found at least one match, return first position
            return pos + __builtin_ctz(ge_mask);
        }
        pos += AVX512_WIDTH;
    }

    // Scalar tail
    while (pos < size) {
        if (ids[pos] >= target) {
            return pos;
        }
        pos++;
    }

    return size;  // Not found
}

#endif  // AVX512F && AVX512DQ

#if defined(__AVX2__)

// AVX2 seek implementation
inline size_t
simd_seek_avx2(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
    size_t pos = start_pos;

    // Scalar until aligned
    while (pos < size && (reinterpret_cast<uintptr_t>(&ids[pos]) % 32) != 0) {
        if (ids[pos] >= target) {
            return pos;
        }
        pos++;
    }

    __m256i target_vec = _mm256_set1_epi32(static_cast<int32_t>(target));

    while (pos + AVX2_WIDTH <= size) {
        __m256i id_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ids[pos]));
        // For AVX2: use _mm256_cmpgt_epi32 and handle equality separately
        // id >= target is equivalent to !(target > id)
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

    // Scalar tail
    while (pos < size) {
        if (ids[pos] >= target) {
            return pos;
        }
        pos++;
    }

    return size;
}

#endif  // AVX2

// Dispatch function for SIMD seek
// Uses compile-time detection - the actual SIMD version depends on compiler flags
inline size_t
simd_seek(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    return simd_seek_avx512(ids, size, start_pos, target);
#elif defined(__AVX2__)
    return simd_seek_avx2(ids, size, start_pos, target);
#else
    return scalar_seek(ids, size, start_pos, target);
#endif
}

// =============================================================================
// Window-Based Index Structure
// =============================================================================

// A window contains posting lists for a subset of documents
// This keeps the distance array small enough to fit in cache
template <typename IdType = uint32_t, typename ValType = float>
struct IndexWindow {
    size_t doc_offset;                                            // First doc_id in this window
    size_t doc_count;                                             // Number of docs in this window
    std::vector<SIMDPostingList<IdType, ValType>> posting_lists;  // Per-dimension posting lists

    IndexWindow(size_t offset, size_t count, size_t num_dims)
        : doc_offset(offset), doc_count(count), posting_lists(num_dims) {
    }

    // Get local doc_id from global doc_id
    IdType
    to_local_id(IdType global_id) const {
        return global_id - static_cast<IdType>(doc_offset);
    }

    // Get global doc_id from local doc_id
    IdType
    to_global_id(IdType local_id) const {
        return local_id + static_cast<IdType>(doc_offset);
    }
};

// =============================================================================
// Thread-Local Distance Buffer
// =============================================================================

// Reusable aligned buffer for distance accumulation
class DistanceBuffer {
 public:
    explicit DistanceBuffer(size_t capacity = DEFAULT_WINDOW_SIZE) {
        resize(capacity);
    }

    void
    resize(size_t new_capacity) {
        size_t padded = ((new_capacity + AVX512_WIDTH - 1) / AVX512_WIDTH) * AVX512_WIDTH;
        buffer_.resize(padded);
        capacity_ = new_capacity;
    }

    void
    reset(size_t size) {
        // Fast zero using SIMD
        std::memset(buffer_.data(), 0, size * sizeof(float));
    }

    float*
    data() {
        return buffer_.data();
    }

    const float*
    data() const {
        return buffer_.data();
    }

    float&
    operator[](size_t idx) {
        return buffer_[idx];
    }

    const float&
    operator[](size_t idx) const {
        return buffer_[idx];
    }

    size_t
    capacity() const {
        return capacity_;
    }

 private:
    AlignedVector<float> buffer_;
    size_t capacity_ = 0;
};

}  // namespace knowhere::sparse

#endif  // SPARSE_INVERTED_INDEX_SIMD_H
