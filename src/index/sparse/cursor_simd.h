// SIMD helpers shared across sparse-index posting-list cursors.
//
// The classic `cursor.next_geq(target)` advance is a linear scan over the
// posting list's sorted uint32_t doc-id array. Sparse BM25 / IP queries call
// this once per (cursor, candidate doc) — at scale that scan is a measurable
// fraction of total query time, even though each call is short. This header
// provides a single helper, next_geq_scan, that does the scan 16 doc-ids at a
// time via AVX-512 when available and falls back to the original scalar loop
// otherwise.

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace knowhere::sparse::inverted::detail {

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
// AVX-512 doc-id scan. Returns the first position p in [start, n) such that
// ids[p] >= target, or n if no such position exists. Reads 16 doc-ids per
// loop iteration via aligned-safe `_mm512_loadu_si512`. Tail uses scalar.
//
// Built with `__attribute__((target("avx512f")))` so it lives in a TU compiled
// without AVX-512 baseline. The dispatcher gates entry on runtime CPUID, so
// hosts without AVX-512 never execute this body.
__attribute__((target("avx512f"))) inline std::size_t
next_geq_scan_avx512(const std::uint32_t* ids, std::size_t start, std::size_t n, std::uint32_t target) noexcept {
    std::size_t pos = start;
    const __m512i tgt = _mm512_set1_epi32(static_cast<int>(target));
    while (pos + 16 <= n) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ids[pos]));
        __mmask16 ge = _mm512_cmpge_epu32_mask(v, tgt);
        if (ge != 0) {
            return pos + __builtin_ctz(static_cast<unsigned int>(ge));
        }
        pos += 16;
    }
    while (pos < n && ids[pos] < target) {
        ++pos;
    }
    return pos;
}
#endif

// Linear scan dispatcher. Identical semantics to the scalar
// `while (pos < n && ids[pos] < target) ++pos;` loop used by all sparse-index
// cursors today; just faster on AVX-512 hosts.
inline std::size_t
next_geq_scan(const std::uint32_t* ids, std::size_t start, std::size_t n, std::uint32_t target) noexcept {
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    static const bool has_avx512 = __builtin_cpu_supports("avx512f");
    if (has_avx512) {
        return next_geq_scan_avx512(ids, start, n, target);
    }
#endif
    std::size_t pos = start;
    while (pos < n && ids[pos] < target) {
        ++pos;
    }
    return pos;
}

}  // namespace knowhere::sparse::inverted::detail
