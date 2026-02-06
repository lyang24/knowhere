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

// Experiment: Test different term ordering formulas for MaxScore
// Goal: Find a formula that works well across datasets with varying query lengths

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

// CSR format loader
struct CSRDataset {
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    std::vector<float> data;
    int64_t n_rows = 0;
    int64_t n_cols = 0;
    int64_t nnz = 0;

    bool
    load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            return false;
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(int64_t));
        indptr.resize(n_rows + 1);
        file.read(reinterpret_cast<char*>(indptr.data()), (n_rows + 1) * sizeof(int64_t));
        indices.resize(nnz);
        file.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int32_t));
        data.resize(nnz);
        file.read(reinterpret_cast<char*>(data.data()), nnz * sizeof(float));
        return file.good();
    }
};

// Inverted index for testing
struct InvertedIndex {
    int64_t n_docs;
    int64_t vocab_size;
    std::vector<std::vector<int32_t>> posting_ids;  // term -> doc_ids
    std::vector<std::vector<float>> posting_vals;   // term -> values
    std::vector<float> max_scores;                  // term -> max score in posting list
    std::vector<float> doc_lens;                    // doc -> length (sum of values)
    float avgdl = 0;

    void
    build(const CSRDataset& ds) {
        n_docs = ds.n_rows;
        vocab_size = ds.n_cols;
        posting_ids.resize(vocab_size);
        posting_vals.resize(vocab_size);
        max_scores.resize(vocab_size, 0.0f);
        doc_lens.resize(n_docs, 0.0f);

        // Build inverted index
        for (int64_t doc = 0; doc < ds.n_rows; ++doc) {
            int64_t start = ds.indptr[doc];
            int64_t end = ds.indptr[doc + 1];
            for (int64_t j = start; j < end; ++j) {
                int32_t term = ds.indices[j];
                float val = ds.data[j];
                posting_ids[term].push_back(doc);
                posting_vals[term].push_back(val);
                if (val > max_scores[term])
                    max_scores[term] = val;
                doc_lens[doc] += val;
            }
        }

        // Compute avgdl
        float sum = 0;
        for (auto len : doc_lens) sum += len;
        avgdl = sum / n_docs;
    }
};

// Query representation
struct Query {
    std::vector<std::pair<int32_t, float>> terms;  // (term_id, weight)
};

std::vector<Query>
load_queries(const CSRDataset& ds) {
    std::vector<Query> queries;
    for (int64_t i = 0; i < ds.n_rows; ++i) {
        Query q;
        int64_t start = ds.indptr[i];
        int64_t end = ds.indptr[i + 1];
        for (int64_t j = start; j < end; ++j) {
            q.terms.push_back({ds.indices[j], ds.data[j]});
        }
        queries.push_back(q);
    }
    return queries;
}

// BM25 scorer
inline float
bm25_score(float tf, float doc_len, float avgdl, float k1 = 1.2f, float b = 0.75f) {
    return tf * (k1 + 1.0f) / (tf + k1 * (1.0f - b + b * doc_len / avgdl));
}

// Ordering formula type
using OrderingFormula = std::function<float(float ub, size_t df, size_t n_docs, size_t query_len)>;

// Formula definitions
std::vector<std::pair<std::string, OrderingFormula>>
get_formulas() {
    return {
        {"V1: UB", [](float ub, size_t df, size_t n_docs, size_t query_len) { return ub; }},
        {"V2: IDF * UB",
         [](float ub, size_t df, size_t n_docs, size_t query_len) {
             float idf = std::log(static_cast<float>(n_docs) / (df + 1.0f));
             return idf * ub;
         }},
        {"F3: UB / sqrt(df)",
         [](float ub, size_t df, size_t n_docs, size_t query_len) { return ub / std::sqrt(df + 1.0f); }},
        {"F4: UB / df", [](float ub, size_t df, size_t n_docs, size_t query_len) { return ub / (df + 1.0f); }},
        {"F5: IDF^2 * UB",
         [](float ub, size_t df, size_t n_docs, size_t query_len) {
             float idf = std::log(static_cast<float>(n_docs) / (df + 1.0f));
             return idf * idf * ub;
         }},
        {"F6: UB * log(N/df) / log(df+2)",
         [](float ub, size_t df, size_t n_docs, size_t query_len) {
             float idf = std::log(static_cast<float>(n_docs) / (df + 1.0f));
             float cost = std::log(df + 2.0f);
             return ub * idf / cost;
         }},
        {"F7: UB / log(df+2)",
         [](float ub, size_t df, size_t n_docs, size_t query_len) { return ub / std::log(df + 2.0f); }},
        {"F8: IDF * UB / sqrt(query_len)",
         [](float ub, size_t df, size_t n_docs, size_t query_len) {
             float idf = std::log(static_cast<float>(n_docs) / (df + 1.0f));
             return idf * ub / std::sqrt(query_len + 1.0f);
         }},
        {"F9: UB * (N/df)^0.5",
         [](float ub, size_t df, size_t n_docs, size_t query_len) {
             return ub * std::sqrt(static_cast<float>(n_docs) / (df + 1.0f));
         }},
        {"F10: UB * (N/df)^0.25",
         [](float ub, size_t df, size_t n_docs, size_t query_len) {
             return ub * std::pow(static_cast<float>(n_docs) / (df + 1.0f), 0.25f);
         }},
    };
}

// MaxScore search with configurable ordering
struct SearchResult {
    std::vector<std::pair<float, int32_t>> topk;  // (score, doc_id)
};

SearchResult
maxscore_search(const InvertedIndex& idx, const Query& q, int k, const OrderingFormula& formula) {
    // Copy and sort query terms by formula
    auto terms = q.terms;
    size_t n_docs = idx.n_docs;
    size_t query_len = terms.size();

    std::sort(terms.begin(), terms.end(), [&](auto& a, auto& b) {
        float ub_a = idx.max_scores[a.first] * a.second;
        float ub_b = idx.max_scores[b.first] * b.second;
        size_t df_a = idx.posting_ids[a.first].size();
        size_t df_b = idx.posting_ids[b.first].size();
        return formula(ub_a, df_a, n_docs, query_len) > formula(ub_b, df_b, n_docs, query_len);
    });

    // Build cursors
    struct Cursor {
        const int32_t* ids;
        const float* vals;
        size_t size;
        size_t pos = 0;
        float q_weight;
        float max_score;
        int32_t
        cur_id() const {
            return pos < size ? ids[pos] : INT32_MAX;
        }
        float
        cur_val() const {
            return vals[pos];
        }
        void
        next() {
            if (pos < size)
                pos++;
        }
    };

    std::vector<Cursor> cursors;
    for (auto& [term, weight] : terms) {
        if (idx.posting_ids[term].empty())
            continue;
        Cursor c;
        c.ids = idx.posting_ids[term].data();
        c.vals = idx.posting_vals[term].data();
        c.size = idx.posting_ids[term].size();
        c.q_weight = weight;
        c.max_score = idx.max_scores[term] * weight;
        cursors.push_back(c);
    }

    if (cursors.empty())
        return {};

    // Compute upper bounds (suffix sums)
    std::vector<float> upper_bounds(cursors.size());
    float sum = 0;
    for (size_t i = cursors.size(); i > 0; --i) {
        sum += cursors[i - 1].max_score;
        upper_bounds[i - 1] = sum;
    }

    // Min-heap for top-k (we use max-heap with negated scores)
    std::vector<std::pair<float, int32_t>> heap;
    float threshold = 0;

    auto heap_push = [&](float score, int32_t doc) {
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push_back({score, doc});
            std::push_heap(heap.begin(), heap.end(), [](auto& a, auto& b) { return a.first > b.first; });
            if (heap.size() == static_cast<size_t>(k)) {
                threshold = heap[0].first;
            }
        } else if (score > threshold) {
            std::pop_heap(heap.begin(), heap.end(), [](auto& a, auto& b) { return a.first > b.first; });
            heap.back() = {score, doc};
            std::push_heap(heap.begin(), heap.end(), [](auto& a, auto& b) { return a.first > b.first; });
            threshold = heap[0].first;
        }
    };

    // Find first non-essential index
    size_t first_ne = cursors.size();
    auto update_first_ne = [&]() {
        while (first_ne > 0 && upper_bounds[first_ne - 1] <= threshold) {
            --first_ne;
        }
    };

    // Main loop
    while (first_ne > 0) {
        // Find next candidate (min doc_id among essential cursors)
        int32_t next_doc = INT32_MAX;
        for (size_t i = 0; i < first_ne; ++i) {
            if (cursors[i].cur_id() < next_doc) {
                next_doc = cursors[i].cur_id();
            }
        }
        if (next_doc == INT32_MAX)
            break;

        // Score document
        float score = 0;
        float doc_len = idx.doc_lens[next_doc];
        for (size_t i = 0; i < first_ne; ++i) {
            if (cursors[i].cur_id() == next_doc) {
                float tf = cursors[i].cur_val();
                score += cursors[i].q_weight * bm25_score(tf, doc_len, idx.avgdl);
                cursors[i].next();
            }
        }

        // Check if we can prune early (score + remaining upper bound <= threshold)
        if (score + (first_ne < cursors.size() ? upper_bounds[first_ne] : 0) > threshold) {
            // Add non-essential contributions
            for (size_t i = first_ne; i < cursors.size(); ++i) {
                while (cursors[i].cur_id() < next_doc) cursors[i].next();
                if (cursors[i].cur_id() == next_doc) {
                    float tf = cursors[i].cur_val();
                    score += cursors[i].q_weight * bm25_score(tf, doc_len, idx.avgdl);
                    cursors[i].next();
                }
            }
            heap_push(score, next_doc);
            update_first_ne();
        }
    }

    SearchResult result;
    result.topk = heap;
    std::sort(result.topk.begin(), result.topk.end(), [](auto& a, auto& b) { return a.first > b.first; });
    return result;
}

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
};

int
main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <data-dir> [nq]\n", argv[0]);
        printf("  data-dir should contain base_small.csr and queries.dev.csr\n");
        return 1;
    }

    std::string data_dir = argv[1];
    int nq = argc > 2 ? atoi(argv[2]) : 500;

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Term Ordering Formula Experiment                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    // Load data
    printf("[Loading Data]\n");
    CSRDataset base_ds, query_ds;
    if (!base_ds.load(data_dir + "/base_small.csr")) {
        printf("Error: Cannot load base vectors\n");
        return 1;
    }
    printf("  Base: %ld docs, %ld vocab, %ld nnz\n", base_ds.n_rows, base_ds.n_cols, base_ds.nnz);

    if (!query_ds.load(data_dir + "/queries.dev.csr")) {
        printf("Error: Cannot load queries\n");
        return 1;
    }
    printf("  Queries: %ld queries, %ld nnz\n", query_ds.n_rows, query_ds.nnz);

    float avg_query_len = static_cast<float>(query_ds.nnz) / query_ds.n_rows;
    printf("  Avg query length: %.2f terms\n", avg_query_len);

    // Build index
    printf("\n[Building Index]\n");
    InvertedIndex idx;
    Timer build_timer;
    idx.build(base_ds);
    printf("  Build time: %.2f ms\n", build_timer.elapsed_ms());

    // Load queries
    auto queries = load_queries(query_ds);
    if (nq > static_cast<int>(queries.size()))
        nq = queries.size();
    printf("  Using %d queries\n", nq);

    // Get formulas
    auto formulas = get_formulas();

    // Warmup
    printf("\n[Warmup]\n");
    for (int i = 0; i < 10; ++i) {
        maxscore_search(idx, queries[i % nq], 10, formulas[0].second);
    }
    printf("  Done\n");

    // Run experiments
    printf("\n[Running Experiments]\n");
    printf("%-35s %10s %10s\n", "Formula", "Time (ms)", "QPS");
    printf("─────────────────────────────────────────────────────────────\n");

    std::vector<std::pair<std::string, double>> results;

    for (auto& [name, formula] : formulas) {
        Timer timer;
        for (int q = 0; q < nq; ++q) {
            maxscore_search(idx, queries[q], 10, formula);
        }
        double elapsed = timer.elapsed_ms();
        double qps = nq * 1000.0 / elapsed;
        printf("%-35s %10.2f %10.1f\n", name.c_str(), elapsed, qps);
        results.push_back({name, qps});
    }

    // Summary
    printf("\n[Summary - Speedup vs V1]\n");
    double v1_qps = results[0].second;
    for (auto& [name, qps] : results) {
        printf("%-35s %.2fx\n", name.c_str(), qps / v1_qps);
    }

    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Experiment Complete                                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    return 0;
}
