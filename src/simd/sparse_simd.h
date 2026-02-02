#ifndef KNOWHERE_SIMD_SPARSE_SIMD_H
#define KNOWHERE_SIMD_SPARSE_SIMD_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "knowhere/sparse_utils.h"
#include "simd/instruction_set.h"

namespace knowhere::sparse {

#if defined(__x86_64__) || defined(_M_X64)
void
accumulate_posting_list_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                                  float* scores);

// AVX2 SIMD seek: find first position where id >= target
size_t
simd_seek_avx2_impl(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target);
#endif

// Scalar seek implementation (fallback)
inline size_t
scalar_seek_impl(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
    for (size_t pos = start_pos; pos < size; pos++) {
        if (ids[pos] >= target) {
            return pos;
        }
    }
    return size;
}

// Dispatch function for SIMD seek with runtime CPU detection
inline size_t
simd_seek_dispatch(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
#if defined(__x86_64__) || defined(_M_X64)
    if (faiss::InstructionSet::GetInstance().AVX2()) {
        return simd_seek_avx2_impl(ids, size, start_pos, target);
    }
#endif
    return scalar_seek_impl(ids, size, start_pos, target);
}

template <typename QType>
inline void
accumulate_posting_list_contribution_ip_dispatch(const uint32_t* doc_ids, const QType* doc_vals, size_t list_size,
                                                 float q_weight, float* scores) {
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<QType, float>) {
        if (faiss::InstructionSet::GetInstance().AVX512F()) {
            accumulate_posting_list_ip_avx512(doc_ids, doc_vals, list_size, q_weight, scores);
            return;
        }
    }
#endif

    // Scalar fallback for IP computation
    for (size_t i = 0; i < list_size; ++i) {
        const auto doc_id = doc_ids[i];
        scores[doc_id] += q_weight * static_cast<float>(doc_vals[i]);
    }
}

}  // namespace knowhere::sparse

#endif  // KNOWHERE_SIMD_SPARSE_SIMD_H
