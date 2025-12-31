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
#endif

template <typename QType>
inline void
accumulate_posting_list_contribution_ip_dispatch(const uint32_t* doc_ids, const QType* doc_vals, size_t list_size,
                                                 float q_weight, float* scores, const DocValueComputer<QType>& computer,
                                                 const float* doc_len_ratios = nullptr) {
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<QType, float>) {
        if (faiss::InstructionSet::GetInstance().AVX512F()) {
            accumulate_posting_list_ip_avx512(doc_ids, doc_vals, list_size, q_weight, scores);
            return;
        }
    }
#endif

    for (size_t i = 0; i < list_size; ++i) {
        const auto doc_id = doc_ids[i];
        const float v = computer(doc_vals[i], doc_len_ratios ? doc_len_ratios[doc_id] : 0.f);
        scores[doc_id] += q_weight * v;
    }
}

}  // namespace knowhere::sparse

#endif  // KNOWHERE_SIMD_SPARSE_SIMD_H
