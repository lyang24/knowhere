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

// Benchmark for add_row_to_index optimization
// Compares original (2 loops) vs optimized (merged loop) implementation
//
// Build: g++ -O3 -std=c++17 -I../include -o benchmark_add_row benchmark_add_row.cpp
// Run: ./benchmark_add_row

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

// Minimal types needed for benchmark
using table_t = uint32_t;
using fp32 = float;

template <typename T>
struct SparseIdVal {
    table_t id;
    T val;
};

// Simplified SparseRow for benchmarking
template <typename T>
class SparseRow {
 public:
    SparseRow(const std::vector<std::pair<table_t, T>>& data) : data_(data) {
    }

    size_t
    size() const {
        return data_.size();
    }

    SparseIdVal<T>
    operator[](size_t i) const {
        return {data_[i].first, data_[i].second};
    }

    T val;  // dummy for compatibility

 private:
    std::vector<std::pair<table_t, T>> data_;
};

// Original implementation (2 loops for WAND/MAXSCORE)
class OriginalIndex {
 public:
    OriginalIndex(bool use_bm25) : use_bm25_(use_bm25) {
    }

    void
    add_row(const SparseRow<fp32>& row, table_t vec_id) {
        float row_sum = 0;
        for (size_t j = 0; j < row.size(); ++j) {
            auto [dim, val] = row[j];
            if (use_bm25_) {
                row_sum += val;
            }
            if (val == 0) {
                continue;
            }
            auto dim_it = dim_map_.find(dim);
            if (dim_it == dim_map_.cend()) {
                dim_it = dim_map_.insert({dim, next_dim_id_++}).first;
                inverted_index_ids_.emplace_back();
                inverted_index_vals_.emplace_back();
                max_score_in_dim_.emplace_back(0.0f);
            }
            inverted_index_ids_[dim_it->second].emplace_back(vec_id);
            inverted_index_vals_[dim_it->second].emplace_back(val);
        }

        // Second loop for max_score_in_dim_ (original behavior)
        for (size_t j = 0; j < row.size(); ++j) {
            auto [dim, val] = row[j];
            if (val == 0) {
                continue;
            }
            auto dim_it = dim_map_.find(dim);
            auto score = static_cast<float>(val);
            if (use_bm25_) {
                // Simulate BM25 score computation
                score = val * (1.5f + 1) / (val + 1.5f * (1 - 0.75f + 0.75f * (row_sum / 100.0f)));
            }
            max_score_in_dim_[dim_it->second] = std::max(max_score_in_dim_[dim_it->second], score);
        }

        if (use_bm25_) {
            row_sums_.emplace_back(row_sum);
        }
    }

    void
    clear() {
        dim_map_.clear();
        inverted_index_ids_.clear();
        inverted_index_vals_.clear();
        max_score_in_dim_.clear();
        row_sums_.clear();
        next_dim_id_ = 0;
    }

 private:
    bool use_bm25_;
    std::unordered_map<table_t, uint32_t> dim_map_;
    std::vector<std::vector<table_t>> inverted_index_ids_;
    std::vector<std::vector<fp32>> inverted_index_vals_;
    std::vector<float> max_score_in_dim_;
    std::vector<float> row_sums_;
    uint32_t next_dim_id_ = 0;
};

// Optimized implementation (merged loop)
class OptimizedIndex {
 public:
    OptimizedIndex(bool use_bm25) : use_bm25_(use_bm25) {
    }

    void
    add_row(const SparseRow<fp32>& row, table_t vec_id) {
        float row_sum = 0;

        // For BM25, pre-compute row_sum in a cheap loop first
        if (use_bm25_) {
            for (size_t j = 0; j < row.size(); ++j) {
                row_sum += row[j].val;
            }
        }

        // Single merged loop
        for (size_t j = 0; j < row.size(); ++j) {
            auto [dim, val] = row[j];
            if (val == 0) {
                continue;
            }
            auto dim_it = dim_map_.find(dim);
            if (dim_it == dim_map_.cend()) {
                dim_it = dim_map_.insert({dim, next_dim_id_++}).first;
                inverted_index_ids_.emplace_back();
                inverted_index_vals_.emplace_back();
                max_score_in_dim_.emplace_back(0.0f);
            }
            auto dim_id = dim_it->second;
            inverted_index_ids_[dim_id].emplace_back(vec_id);
            inverted_index_vals_[dim_id].emplace_back(val);

            // Update max_score_in_dim_ in the same loop
            auto score = static_cast<float>(val);
            if (use_bm25_) {
                // Simulate BM25 score computation
                score = val * (1.5f + 1) / (val + 1.5f * (1 - 0.75f + 0.75f * (row_sum / 100.0f)));
            }
            max_score_in_dim_[dim_id] = std::max(max_score_in_dim_[dim_id], score);
        }

        if (use_bm25_) {
            row_sums_.emplace_back(row_sum);
        }
    }

    void
    clear() {
        dim_map_.clear();
        inverted_index_ids_.clear();
        inverted_index_vals_.clear();
        max_score_in_dim_.clear();
        row_sums_.clear();
        next_dim_id_ = 0;
    }

 private:
    bool use_bm25_;
    std::unordered_map<table_t, uint32_t> dim_map_;
    std::vector<std::vector<table_t>> inverted_index_ids_;
    std::vector<std::vector<fp32>> inverted_index_vals_;
    std::vector<float> max_score_in_dim_;
    std::vector<float> row_sums_;
    uint32_t next_dim_id_ = 0;
};

// Generate random sparse vectors
std::vector<SparseRow<fp32>>
generate_sparse_vectors(size_t num_vectors, size_t avg_nnz, size_t max_dim) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> nnz_dist(avg_nnz / 2, avg_nnz * 3 / 2);
    std::uniform_int_distribution<table_t> dim_dist(0, max_dim - 1);
    std::uniform_real_distribution<fp32> val_dist(0.1f, 10.0f);

    std::vector<SparseRow<fp32>> vectors;
    vectors.reserve(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        size_t nnz = nnz_dist(gen);
        std::vector<std::pair<table_t, fp32>> data;
        data.reserve(nnz);

        std::vector<table_t> dims;
        for (size_t j = 0; j < nnz; ++j) {
            dims.push_back(dim_dist(gen));
        }
        std::sort(dims.begin(), dims.end());
        dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

        for (auto dim : dims) {
            data.emplace_back(dim, val_dist(gen));
        }
        vectors.emplace_back(data);
    }
    return vectors;
}

template <typename IndexType>
double
benchmark_add_rows(IndexType& index, const std::vector<SparseRow<fp32>>& vectors, int iterations) {
    double total_ms = 0;

    for (int iter = 0; iter < iterations; ++iter) {
        index.clear();
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < vectors.size(); ++i) {
            index.add_row(vectors[i], static_cast<table_t>(i));
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return total_ms / iterations;
}

int
main() {
    std::cout << "=== Benchmark: add_row_to_index optimization ===\n\n";

    // Test configurations
    struct Config {
        size_t num_vectors;
        size_t avg_nnz;
        size_t max_dim;
        const char* name;
    };

    std::vector<Config> configs = {
        {10000, 50, 10000, "Small (10K vectors, 50 nnz, 10K dims)"},
        {50000, 100, 50000, "Medium (50K vectors, 100 nnz, 50K dims)"},
        {100000, 200, 100000, "Large (100K vectors, 200 nnz, 100K dims)"},
    };

    const int iterations = 3;

    for (const auto& config : configs) {
        std::cout << "Config: " << config.name << "\n";

        auto vectors = generate_sparse_vectors(config.num_vectors, config.avg_nnz, config.max_dim);

        // Calculate total nnz
        size_t total_nnz = 0;
        for (const auto& v : vectors) {
            total_nnz += v.size();
        }
        std::cout << "  Total nnz: " << total_nnz << "\n";

        // Test IP metric
        {
            OriginalIndex orig_ip(false);
            OptimizedIndex opt_ip(false);

            double orig_ms = benchmark_add_rows(orig_ip, vectors, iterations);
            double opt_ms = benchmark_add_rows(opt_ip, vectors, iterations);

            std::cout << "  IP metric:\n";
            std::cout << "    Original:  " << orig_ms << " ms\n";
            std::cout << "    Optimized: " << opt_ms << " ms\n";
            std::cout << "    Speedup:   " << (orig_ms / opt_ms) << "x\n";
        }

        // Test BM25 metric
        {
            OriginalIndex orig_bm25(true);
            OptimizedIndex opt_bm25(true);

            double orig_ms = benchmark_add_rows(orig_bm25, vectors, iterations);
            double opt_ms = benchmark_add_rows(opt_bm25, vectors, iterations);

            std::cout << "  BM25 metric:\n";
            std::cout << "    Original:  " << orig_ms << " ms\n";
            std::cout << "    Optimized: " << opt_ms << " ms\n";
            std::cout << "    Speedup:   " << (orig_ms / opt_ms) << "x\n";
        }

        std::cout << "\n";
    }

    return 0;
}
