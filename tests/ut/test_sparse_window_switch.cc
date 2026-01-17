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

// Unit tests for the Window Switch optimization in sparse inverted index.
// The Window Switch strategy (from SINDI paper, VLDB 2026) improves cache locality
// by processing posting lists in fixed-size windows instead of random writes
// across the entire score array.

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

using namespace knowhere;

// Helper to generate sparse dataset with controlled characteristics
knowhere::DataSetPtr
GenSparseDataSetForWindowTest(int32_t nb, int32_t dim, float sparsity, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 10.0f);
    std::uniform_real_distribution<float> sparse_dist(0.0f, 1.0f);

    std::vector<std::map<int32_t, float>> data(nb);
    for (int32_t i = 0; i < nb; ++i) {
        for (int32_t j = 0; j < dim; ++j) {
            if (sparse_dist(gen) > sparsity) {
                data[i][j] = val_dist(gen);
            }
        }
    }
    return GenSparseDataSet(data, dim);
}

TEST_CASE("Test Window Switch Correctness - Small Dataset (No Windowing)", "[sparse window switch]") {
    // Small dataset below the windowed threshold - should use direct processing
    // This is a baseline test to ensure the code path works correctly

    auto nb = 1000;  // Below kWindowedProcessingThreshold (200000)
    auto dim = 500;
    auto doc_sparsity = 0.95f;  // 5% non-zero = avg 25 elements per doc
    auto query_sparsity = 0.97f;
    auto topk = 10;
    int64_t nq = 5;

    auto version = GenTestVersionList();

    auto train_ds = GenSparseDataSetForWindowTest(nb, dim, doc_sparsity, 12345);
    auto query_ds = GenSparseDataSetForWindowTest(nq, dim, query_sparsity, 54321);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0f;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "TAAT_NAIVE";

    // Build index
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    // Compute ground truth using brute force
    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
    REQUIRE(gt.has_value());

    // Search using index (uses compute_all_distances internally)
    auto results = idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());

    // Compare results - should be identical for drop_ratio_search = 0
    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall == 1.0f);
}

TEST_CASE("Test Window Switch Correctness - Large Dataset (With Windowing)", "[sparse window switch]") {
    // Large dataset above the windowed threshold - should use Window Switch processing
    // This tests that the windowed version produces correct results

    auto nb = 250000;  // Above kWindowedProcessingThreshold (200000)
    auto dim = 1000;
    auto doc_sparsity = 0.99f;  // 1% non-zero = avg 10 elements per doc
    auto query_sparsity = 0.995f;
    auto topk = 10;
    int64_t nq = 3;  // Fewer queries due to larger dataset

    auto version = GenTestVersionList();

    auto train_ds = GenSparseDataSetForWindowTest(nb, dim, doc_sparsity, 11111);
    auto query_ds = GenSparseDataSetForWindowTest(nq, dim, query_sparsity, 22222);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0f;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "TAAT_NAIVE";

    // Build index
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
    REQUIRE(idx.Count() == nb);

    // Compute ground truth using brute force
    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
    REQUIRE(gt.has_value());

    // Search using index (uses windowed compute_all_distances for IP metric)
    auto results = idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());

    // Compare results - should be identical for drop_ratio_search = 0
    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall == 1.0f);
}

TEST_CASE("Test Window Switch Consistency Across Dataset Sizes", "[sparse window switch]") {
    // Test that results are consistent across the windowing threshold boundary
    // by comparing results from two indices with different dataset sizes

    auto dim = 500;
    auto doc_sparsity = 0.98f;
    auto query_sparsity = 0.99f;
    auto topk = 10;
    int64_t nq = 3;

    auto version = GenTestVersionList();

    // Use same seed for both datasets to ensure query overlaps with data
    auto query_ds = GenSparseDataSetForWindowTest(nq, dim, query_sparsity, 99999);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0f;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "TAAT_NAIVE";

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
    };

    // Test with different dataset sizes spanning the threshold
    auto nb_values = GENERATE(50000, 100000, 200000, 250000, 300000);

    auto train_ds = GenSparseDataSetForWindowTest(nb_values, dim, doc_sparsity, 12345);

    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
    REQUIRE(gt.has_value());

    auto results = idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());

    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall == 1.0f);
}

TEST_CASE("Test Window Switch with Various Query Characteristics", "[sparse window switch]") {
    // Test Window Switch with different query term counts and weights

    auto nb = 250000;  // Above threshold
    auto dim = 2000;
    auto doc_sparsity = 0.995f;  // Very sparse: avg 10 elements
    auto topk = 10;
    int64_t nq = 2;

    auto version = GenTestVersionList();

    auto train_ds = GenSparseDataSetForWindowTest(nb, dim, doc_sparsity, 33333);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0f;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "TAAT_NAIVE";

    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
    };

    SECTION("Sparse query (few terms)") {
        auto query_ds = GenSparseDataSetForWindowTest(nq, dim, 0.998f, 44444);  // ~4 terms

        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        auto results = idx.Search(query_ds, json, nullptr);

        REQUIRE(gt.has_value());
        REQUIRE(results.has_value());
        REQUIRE(GetKNNRecall(*gt.value(), *results.value()) == 1.0f);
    }

    SECTION("Dense query (many terms)") {
        auto query_ds = GenSparseDataSetForWindowTest(nq, dim, 0.95f, 55555);  // ~100 terms

        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        auto results = idx.Search(query_ds, json, nullptr);

        REQUIRE(gt.has_value());
        REQUIRE(results.has_value());
        REQUIRE(GetKNNRecall(*gt.value(), *results.value()) == 1.0f);
    }
}

TEST_CASE("Test Window Switch with Bitset Filter", "[sparse window switch]") {
    // Test that Window Switch works correctly with bitset filtering

    auto nb = 250000;
    auto dim = 500;
    auto doc_sparsity = 0.98f;
    auto query_sparsity = 0.99f;
    auto topk = 10;
    int64_t nq = 3;

    auto version = GenTestVersionList();

    auto train_ds = GenSparseDataSetForWindowTest(nb, dim, doc_sparsity, 66666);
    auto query_ds = GenSparseDataSetForWindowTest(nq, dim, query_sparsity, 77777);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0f;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "TAAT_NAIVE";

    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
    };

    // Test with different filter ratios
    auto filter_ratio = GENERATE(0.1f, 0.5f, 0.9f);

    auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, static_cast<size_t>(filter_ratio * nb));
    knowhere::BitsetView bitset(bitset_data.data(), nb);

    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, bitset);
    REQUIRE(gt.has_value());

    auto results = idx.Search(query_ds, json, bitset);
    REQUIRE(results.has_value());

    // Verify all results respect the filter
    auto* ids = results.value()->GetIds();
    auto k = results.value()->GetDim();
    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            auto id = ids[i * k + j];
            if (id != -1) {
                REQUIRE(!bitset.test(id));
            }
        }
    }

    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall == 1.0f);
}
