// Document-at-a-Time (DAAT) MaxScore searcher.
// Derived from the PISA search engine (Performant Indexes and Search for Academia).
//   Paper: H. Turtle and J. Flood, "Query Evaluation: Strategies and Optimizations",
//          Information Processing & Management, 1995.
//   Repository: https://github.com/pisa-engine/pisa
//   License: Apache License 2.0

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "index/sparse/scorer.h"
#include "index/sparse/searcher/searcher.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

namespace detail {

// Sum of BM25 contributions for `n` matching query terms at the same document.
// Each contribution is qval_p1[i] * tf[i] / (tf[i] + doc_norm), where doc_norm
// is per-doc (precomputed once outside) and qval_p1/tf are per-(term, doc).
//
// Scalar reference; auto-vectorization typically can't kick in because the
// loop body has a division on a small dynamic trip count.
inline float
bm25_batch_contrib_scalar(const float* qval_p1s, const float* tfs, std::size_t n, float doc_norm) noexcept {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += qval_p1s[i] * tfs[i] / (tfs[i] + doc_norm);
    }
    return sum;
}

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
// AVX-512 kernel: 16 contributions per pass plus a masked tail. Compiled with
// the AVX-512 target attribute so it can live in a non-AVX-512 translation unit.
__attribute__((target("avx512f"))) inline float
bm25_batch_contrib_avx512(const float* qval_p1s, const float* tfs, std::size_t n, float doc_norm) noexcept {
    __m512 sum = _mm512_setzero_ps();
    const __m512 dn = _mm512_set1_ps(doc_norm);
    std::size_t i = 0;
    while (i + 16 <= n) {
        __m512 q = _mm512_loadu_ps(qval_p1s + i);
        __m512 tf_v = _mm512_loadu_ps(tfs + i);
        __m512 num = _mm512_mul_ps(q, tf_v);
        __m512 den = _mm512_add_ps(tf_v, dn);
        sum = _mm512_add_ps(sum, _mm512_div_ps(num, den));
        i += 16;
    }
    const std::size_t tail = n - i;
    if (tail > 0) {
        const __mmask16 mask = static_cast<__mmask16>((1U << tail) - 1U);
        __m512 q = _mm512_maskz_loadu_ps(mask, qval_p1s + i);
        __m512 tf_v = _mm512_maskz_loadu_ps(mask, tfs + i);
        __m512 num = _mm512_mul_ps(q, tf_v);
        // den unused lanes get 1.0 so the masked div doesn't see /0 even though
        // we mask the result anyway. Belt and suspenders.
        __m512 den = _mm512_mask_add_ps(_mm512_set1_ps(1.0f), mask, tf_v, dn);
        sum = _mm512_add_ps(sum, _mm512_maskz_div_ps(mask, num, den));
    }
    return _mm512_reduce_add_ps(sum);
}
#endif

// Dispatch. Uses AVX-512 when the host supports it and the trip count is large
// enough to amortize the horizontal reduce. The threshold (n >= 4) is a guess
// to tune empirically: below it scalar division latency wins; above it the
// SIMD div throughput dominates.
inline float
bm25_batch_contrib(const float* qval_p1s, const float* tfs, std::size_t n, float doc_norm) noexcept {
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    static const bool has_avx512 = __builtin_cpu_supports("avx512f");
    if (has_avx512 && n >= 4) {
        return bm25_batch_contrib_avx512(qval_p1s, tfs, n, doc_norm);
    }
#endif
    return bm25_batch_contrib_scalar(qval_p1s, tfs, n, doc_norm);
}

// Maximum essential matches collected in SoA buffers per doc. Real BM25 queries
// rarely exceed 30-50 terms; any overflow is scalar-scored as a fallback so
// correctness is preserved even on pathological inputs.
constexpr std::size_t kSimdBatchCap = 64;

}  // namespace detail

template <typename IndexType>
class DaatMaxScoreSearcher : public RankedSearcher {
 public:
    struct Cursor {
        typename IndexType::posting_list_iterator index_cursor;
        DimScorer scorer;
        float max_score;
        float qval_p1;

        [[nodiscard]] uint32_t
        vec_id() const noexcept {
            return index_cursor.vec_id();
        }

        [[nodiscard]] float
        score() noexcept {
            return scorer(index_cursor.vec_id(), index_cursor.val());
        }

        void
        next() noexcept {
            index_cursor.next();
        }

        void
        next_geq(uint32_t vec_id) noexcept {
            index_cursor.next_geq(vec_id);
        }

        [[nodiscard]] bool
        valid() const noexcept {
            return index_cursor.valid();
        }
    };

    explicit DaatMaxScoreSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                                  const std::shared_ptr<IndexScorer>& search_scorer, const uint32_t k,
                                  const uint32_t max_vec_id, const BitsetView& bitset, float dim_max_score_ratio)
        : RankedSearcher(k),
          cursors_(make_cursors(index, query, search_scorer, bitset, dim_max_score_ratio)),
          max_vec_id_(max_vec_id),
          row_sums_(index.get_row_sums()),
          scorer_type_(search_scorer->config().scorer_type) {
        if (scorer_type_ == IndexScorerType::BM25) {
            const auto* bm25_scorer = dynamic_cast<const BM25IndexScorer*>(search_scorer.get());
            assert(bm25_scorer != nullptr);
            bm25_p2_ = bm25_scorer->p2();
            bm25_p3_ = bm25_scorer->p3();
        }
    }

    [[nodiscard]] auto
    sorted(std::vector<Cursor>& cursors) -> std::vector<Cursor> {
        std::vector<size_t> term_positions(cursors.size());
        std::iota(term_positions.begin(), term_positions.end(), 0);
        std::sort(term_positions.begin(), term_positions.end(),
                  [&](auto&& lhs, auto&& rhs) { return cursors[lhs].max_score > cursors[rhs].max_score; });
        std::vector<Cursor> sorted;
        sorted.reserve(cursors.size());
        for (auto pos : term_positions) {
            sorted.push_back(std::move(cursors[pos]));
        };
        return sorted;
    }

    [[nodiscard]] auto
    calc_upper_bounds(std::vector<Cursor>& cursors) -> std::vector<float> {
        std::vector<float> upper_bounds(cursors.size());
        auto out = upper_bounds.rbegin();
        float bound = 0.0;
        for (auto pos = cursors.rbegin(); pos != cursors.rend(); ++pos) {
            bound += pos->max_score;
            *out++ = bound;
        }
        return upper_bounds;
    }

    [[nodiscard]] auto
    min_vec_id(std::vector<Cursor>& cursors) -> uint32_t {
        return std::min_element(cursors.begin(), cursors.end(),
                                [](auto&& lhs, auto&& rhs) { return lhs.vec_id() < rhs.vec_id(); })
            ->vec_id();
    }

    enum class UpdateResult : bool { Continue, ShortCircuit };
    enum class VectorStatus : bool { Insert, Skip };

    template <IndexScorerType ScorerType>
    void
    run_sorted(std::vector<Cursor>& cursors, uint64_t max_vec_id) {
        auto upper_bounds = calc_upper_bounds(cursors);
        auto above_threshold = [&](auto score) { return topk_.WouldEnter(score); };

        auto first_upper_bound = upper_bounds.end();
        auto first_lookup = cursors.end();
        auto next_vec_id = min_vec_id(cursors);

        auto update_non_essential_lists = [&] {
            while (first_lookup != cursors.begin() && !above_threshold(*std::prev(first_upper_bound))) {
                --first_lookup;
                --first_upper_bound;
                if (first_lookup == cursors.begin()) {
                    return UpdateResult::ShortCircuit;
                }
            }
            return UpdateResult::Continue;
        };

        if (update_non_essential_lists() == UpdateResult::ShortCircuit) {
            return;
        }

        float current_score = 0;
        uint32_t current_vec_id = 0;

        while (current_vec_id < max_vec_id) {
            auto status = VectorStatus::Skip;
            while (status == VectorStatus::Skip) {
                if (next_vec_id >= max_vec_id) [[unlikely]] {
                    return;
                }

                current_score = 0;
                current_vec_id = std::exchange(next_vec_id, max_vec_id);
                float doc_norm = 0.0f;

                if constexpr (ScorerType == IndexScorerType::BM25) {
                    // Prefetch row_sums_ for next iterations that will be used by the BM25 scorer
                    // Experiments show this prefetch pattern is optimal vs only prefetching next_vec_id
                    __builtin_prefetch(&row_sums_[current_vec_id], 0, 3);
                    doc_norm = bm25_p2_ + bm25_p3_ * row_sums_[current_vec_id];
                }

                auto score_term = [&](auto& cursor) -> float {
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        const float tf = static_cast<float>(cursor.index_cursor.val());
                        return cursor.qval_p1 * tf / (tf + doc_norm);
                    } else {
                        return cursor.score();
                    }
                };

                // Essential pass. For BM25, collect matching (qval_p1, tf) pairs
                // into stack SoA so the per-(term, doc) divisions can be issued
                // as a single SIMD batch instead of a chain of scalar divs. For
                // IP we keep the scalar accumulation.
                if constexpr (ScorerType == IndexScorerType::BM25) {
                    std::array<float, detail::kSimdBatchCap> tf_buf;
                    std::array<float, detail::kSimdBatchCap> qval_p1_buf;
                    std::size_t n_matches = 0;
                    std::for_each(cursors.begin(), first_lookup, [&](auto& cursor) {
                        if (cursor.vec_id() == current_vec_id) {
                            if (n_matches < detail::kSimdBatchCap) [[likely]] {
                                tf_buf[n_matches] = static_cast<float>(cursor.index_cursor.val());
                                qval_p1_buf[n_matches] = cursor.qval_p1;
                                ++n_matches;
                            } else {
                                // Overflow path: extra essential matches at one doc beyond the
                                // batch cap are scored scalar to keep correctness.
                                current_score += score_term(cursor);
                            }
                            cursor.next();
                            __builtin_prefetch(&row_sums_[cursor.vec_id()], 0, 3);
                        }
                        if (auto vec_id = cursor.vec_id(); vec_id < next_vec_id) {
                            next_vec_id = vec_id;
                        }
                    });
                    if (n_matches > 0) {
                        current_score +=
                            detail::bm25_batch_contrib(qval_p1_buf.data(), tf_buf.data(), n_matches, doc_norm);
                    }
                } else {
                    std::for_each(cursors.begin(), first_lookup, [&](auto& cursor) {
                        if (cursor.vec_id() == current_vec_id) {
                            current_score += score_term(cursor);
                            cursor.next();
                        }
                        if (auto vec_id = cursor.vec_id(); vec_id < next_vec_id) {
                            next_vec_id = vec_id;
                        }
                    });
                }

                status = VectorStatus::Insert;
                auto lookup_bound = first_upper_bound;
                for (auto pos = first_lookup; pos != cursors.end(); ++pos, ++lookup_bound) {
                    auto& cursor = *pos;
                    if (!above_threshold(current_score + *lookup_bound)) {
                        status = VectorStatus::Skip;
                        break;
                    }
                    cursor.next_geq(current_vec_id);
                    if (cursor.vec_id() == current_vec_id) {
                        current_score += score_term(cursor);
                    }
                }
            }
            if (topk_.Push(current_score, current_vec_id) &&
                update_non_essential_lists() == UpdateResult::ShortCircuit) {
                return;
            }
        }
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }
        auto cursors = sorted(cursors_);
        if (scorer_type_ == IndexScorerType::BM25) {
            run_sorted<IndexScorerType::BM25>(cursors, max_vec_id_);
        } else {
            run_sorted<IndexScorerType::IP>(cursors, max_vec_id_);
        }
        std::swap(cursors, cursors_);
    }

 private:
    static std::vector<Cursor>
    make_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                 const std::shared_ptr<IndexScorer>& index_scorer, const BitsetView& bitset,
                 float dim_max_score_ratio) {
        std::vector<Cursor> cursors;
        cursors.reserve(query.size());
        const BM25IndexScorer* bm25_scorer = nullptr;
        if (index_scorer->config().scorer_type == IndexScorerType::BM25) {
            bm25_scorer = dynamic_cast<const BM25IndexScorer*>(index_scorer.get());
            assert(bm25_scorer != nullptr);
        }
        for (const auto& [dim_id, dim_val] : query) {
            cursors.push_back(Cursor{index.get_dim_plist_cursor(dim_id, bitset), index_scorer->dim_scorer(dim_val),
                                     dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val),
                                     bm25_scorer != nullptr ? dim_val * bm25_scorer->p1() : 0.0f});
        }
        return cursors;
    }

    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
    // row_sums_ is only used for BM25 scorer
    const std::vector<float>& row_sums_;
    IndexScorerType scorer_type_;
    float bm25_p2_{0.0f};
    float bm25_p3_{0.0f};
};

}  // namespace knowhere::sparse::inverted
