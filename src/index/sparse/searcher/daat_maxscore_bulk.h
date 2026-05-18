// Bulk/windowed DAAT MaxScore searcher (experimental v1).
//
// Motivation:
//   The production DaatMaxScoreSearcher is per-document alternating: it picks a
//   candidate vec_id, visits essential cursors, probes non-essential ones, and
//   scores in place. This shape resists SIMD because (a) the inner loop is a
//   branch on cursor.vec_id() == current_vec_id, (b) BM25 row_sums[d] is read
//   in random order across cursor visits, and (c) per-(doc, term) scoring
//   alternates between iterators rather than batching one iterator's postings.
//
//   This experimental searcher rearranges execution into a windowed batched
//   loop: for each fixed-size doc window, precompute per-doc BM25
//   denominators in one sequential pass over row_sums, then walk each cursor's
//   postings inside the window and accumulate contributions into a dense
//   scratch buffer. Top-k Push happens at window end.
//
//   The windowed shape is closer in spirit to Lucene's MaxScoreBulkScorer and
//   turbopuffer's "vectorized MAXSCORE", and is the structural change needed
//   before a real SIMD BM25 batch kernel can land.
//
// v1 scope (intentionally minimal):
//   - BM25 only. IP path is rejected at construction (cursors_ left empty so
//     search() short-circuits); the caller should route IP queries to
//     DaatMaxScoreSearcher.
//   - No essential/non-essential pruning yet: every cursor is scanned over
//     every window. This will visit strictly more (doc, term) pairs than
//     production DAAT_MAXSCORE. The point of v1 is to measure whether the
//     batched control flow itself is fast enough that adding pruning back in
//     Stage B can win overall.
//   - Fixed window size of 1024 docs (~4 KB scratch buffers, fits L1).
//   - dim_max_score_ratio currently unused (no pruning to scale).
//   - Uses the same on-disk metadata as DAAT_MAXSCORE (max_score_per_dim_,
//     row_sums_); no new build-time work required.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "index/sparse/scorer.h"
#include "index/sparse/searcher/searcher.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

template <typename IndexType>
class DaatMaxScoreBulkSearcher : public RankedSearcher {
 public:
    static constexpr uint32_t kWindowSize = 1024;

    struct Cursor {
        typename IndexType::posting_list_iterator index_cursor;
        // qval * (k1 + 1) precomputed per query term; together with the
        // per-window doc_norm this is all the BM25 batch kernel needs.
        float qval_p1;
        // Per-term max BM25 contribution at the heaviest doc. Carried for
        // future pruning (Stage B); unused in v1.
        float max_score;
    };

    explicit DaatMaxScoreBulkSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                                      const std::shared_ptr<IndexScorer>& search_scorer, const uint32_t k,
                                      const uint32_t max_vec_id, const BitsetView& bitset, float dim_max_score_ratio)
        : RankedSearcher(k), max_vec_id_(max_vec_id), row_sums_(index.get_row_sums()) {
        const auto& cfg = search_scorer->config();
        // v1 supports BM25 only. IP would need a different per-window
        // contribution kernel (no doc_norm) and brings no incremental signal
        // for this experiment.
        if (cfg.scorer_type != IndexScorerType::BM25) {
            return;
        }
        const float k1 = cfg.scorer_params.bm25.k1;
        const float b = cfg.scorer_params.bm25.b;
        const float avgdl = cfg.scorer_params.bm25.avgdl;
        p1_ = k1 + 1.0f;
        p2_ = k1 * (1.0f - b);
        p3_ = k1 * b / avgdl;

        cursors_.reserve(query.size());
        for (const auto& [dim_id, dim_val] : query) {
            cursors_.push_back(Cursor{index.get_dim_plist_cursor(dim_id, bitset), dim_val * p1_,
                                      dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val)});
        }
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }

        // Stack scratch, ~12 KB total, sized to keep all three arrays in L1.
        // Aligned for potential SIMD batch kernel in a follow-up stage.
        alignas(64) std::array<float, kWindowSize> scores{};
        alignas(64) std::array<float, kWindowSize> doc_norms{};
        alignas(64) std::array<uint8_t, kWindowSize> hit{};

        for (uint32_t win_start = 0; win_start < max_vec_id_; win_start += kWindowSize) {
            const uint32_t win_end = std::min<uint32_t>(win_start + kWindowSize, max_vec_id_);
            const uint32_t win_len = win_end - win_start;
            process_window(win_start, win_end, win_len, scores, doc_norms, hit);
        }
    }

 private:
    void
    process_window(uint32_t win_start, uint32_t win_end, uint32_t win_len, std::array<float, kWindowSize>& scores,
                   std::array<float, kWindowSize>& doc_norms, std::array<uint8_t, kWindowSize>& hit) {
        // 1) Sequential bulk read of row_sums[win_start..win_end) replaces the
        //    O(matched-pair) random row_sums lookups in the alternating
        //    searcher. This is the main locality win we want to measure.
        for (uint32_t i = 0; i < win_len; ++i) {
            doc_norms[i] = p2_ + p3_ * row_sums_[win_start + i];
        }
        // 2) Reset only the live prefix.
        std::fill(scores.begin(), scores.begin() + win_len, 0.0f);
        std::fill(hit.begin(), hit.begin() + win_len, uint8_t{0});

        // 3) For each cursor (== query term), walk its posting list within the
        //    window and accumulate the BM25 contribution into the dense
        //    scratch. The inner loop is uniform per cursor: no branch on
        //    "which iterator are we on", no per-term threshold check, no
        //    deferred non-essential probe. This is the shape a SIMD batch
        //    kernel would consume.
        for (auto& cursor : cursors_) {
            cursor.index_cursor.next_geq(win_start);
            while (cursor.index_cursor.valid() && cursor.index_cursor.vec_id() < win_end) {
                const uint32_t rel = cursor.index_cursor.vec_id() - win_start;
                const float tf = static_cast<float>(cursor.index_cursor.val());
                scores[rel] += cursor.qval_p1 * tf / (tf + doc_norms[rel]);
                hit[rel] = 1;
                cursor.index_cursor.next();
            }
        }

        // 4) Push survivors to top-k. WouldEnter is the only pruning lever in
        //    v1 (no MaxScore demotion).
        for (uint32_t i = 0; i < win_len; ++i) {
            if (hit[i] && topk_.WouldEnter(scores[i])) {
                topk_.Push(scores[i], win_start + i);
            }
        }
    }

    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
    const std::vector<float>& row_sums_;
    float p1_ = 0.0f;
    float p2_ = 0.0f;
    float p3_ = 0.0f;
};

}  // namespace knowhere::sparse::inverted
