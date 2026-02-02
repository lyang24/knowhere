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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "knowhere/bitsetview.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/sparse_utils.h"

// CSR format loader for MSMARCO/SPLADE data from big-ann-benchmarks
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
        if (!file) {
            printf("Error: Cannot open file %s\n", path.c_str());
            return false;
        }

        // Read header: n_rows, n_cols, nnz (all int64_t)
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(int64_t));

        printf("  Loading CSR: %ld rows, %ld cols, %ld nnz\n", n_rows, n_cols, nnz);

        // Read indptr (n_rows + 1 int64_t values)
        indptr.resize(n_rows + 1);
        file.read(reinterpret_cast<char*>(indptr.data()), (n_rows + 1) * sizeof(int64_t));

        // Read indices (nnz int32_t values)
        indices.resize(nnz);
        file.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int32_t));

        // Read data (nnz float values)
        data.resize(nnz);
        file.read(reinterpret_cast<char*>(data.data()), nnz * sizeof(float));

        if (!file) {
            printf("Error: Failed to read file %s\n", path.c_str());
            return false;
        }

        return true;
    }

    // Convert to knowhere SparseRow format
    std::unique_ptr<knowhere::sparse::SparseRow<float>[]>
    to_sparse_rows() const {
        auto rows = std::make_unique<knowhere::sparse::SparseRow<float>[]>(n_rows);
        for (int64_t i = 0; i < n_rows; ++i) {
            int64_t start = indptr[i];
            int64_t end = indptr[i + 1];
            int64_t len = end - start;
            rows[i] = knowhere::sparse::SparseRow<float>(len);
            for (int64_t j = 0; j < len; ++j) {
                rows[i].set_at(j, indices[start + j], data[start + j]);
            }
        }
        return rows;
    }
};

// Ground truth loader (binary format: nq x k int32_t)
struct GroundTruth {
    std::vector<std::vector<int32_t>> gt;
    int64_t nq = 0;
    int64_t k = 0;

    bool
    load(const std::string& path, int64_t expected_nq) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            printf("Error: Cannot open ground truth file %s\n", path.c_str());
            return false;
        }

        // Get file size to determine k
        file.seekg(0, std::ios::end);
        int64_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        nq = expected_nq;
        k = file_size / (nq * sizeof(int32_t));
        printf("  Loading GT: %ld queries, k=%ld\n", nq, k);

        gt.resize(nq);
        for (int64_t i = 0; i < nq; ++i) {
            gt[i].resize(k);
            file.read(reinterpret_cast<char*>(gt[i].data()), k * sizeof(int32_t));
        }

        return file.good();
    }

    float
    compute_recall(const int64_t* results, int64_t query_idx, int64_t result_k) const {
        if (query_idx >= nq)
            return 0.0f;
        int64_t check_k = std::min(result_k, k);
        int matches = 0;
        for (int64_t i = 0; i < check_k; ++i) {
            for (int64_t j = 0; j < check_k; ++j) {
                if (results[i] == gt[query_idx][j]) {
                    matches++;
                    break;
                }
            }
        }
        return static_cast<float>(matches) / check_k;
    }
};

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

    void
    reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

void
print_usage(const char* prog) {
    printf("Usage: %s --data-dir <path> [--topk <k>] [--nq <num_queries>]\n", prog);
    printf("\nExpected files in data-dir:\n");
    printf("  base_small.csr   - Base vectors in CSR format\n");
    printf("  queries.dev.csr  - Query vectors in CSR format\n");
    printf("  base_small.gt    - Ground truth (binary int32)\n");
    printf("\nDownload from:\n");
    printf("  wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.csr.gz\n");
    printf("  wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/queries.dev.csr.gz\n");
    printf("  wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.dev.gt\n");
}

int
main(int argc, char** argv) {
    std::string data_dir;
    int64_t topk = 10;
    int64_t nq = 0;  // 0 = use all queries

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            topk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc) {
            nq = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_dir.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    printf("==========================================================\n");
    printf("  Sparse Search Algorithm Benchmark (MaxScore vs MaxScore v2)\n");
    printf("==========================================================\n\n");

    // Initialize knowhere
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    // Load datasets
    printf("[Loading Data]\n");
    CSRDataset base, queries;

    if (!base.load(data_dir + "/base_small.csr")) {
        return 1;
    }
    if (!queries.load(data_dir + "/queries.dev.csr")) {
        return 1;
    }

    if (nq == 0 || nq > queries.n_rows) {
        nq = queries.n_rows;
    }
    printf("  Using %ld queries\n", nq);

    // Load ground truth
    GroundTruth gt;
    if (!gt.load(data_dir + "/base_small.gt", queries.n_rows)) {
        printf("Warning: Ground truth not loaded, recall will not be computed\n");
    }

    // Convert to sparse rows
    printf("\n[Converting to SparseRow format]\n");
    auto base_rows = base.to_sparse_rows();
    auto query_rows = queries.to_sparse_rows();
    printf("  Done\n");

    // Algorithms to benchmark
    std::vector<std::string> algos = {"DAAT_MAXSCORE", "DAAT_MAXSCORE_V2"};

    // Benchmark parameters following DSP paper methodology:
    // 5 runs, drop first 2 (warmup), average last 3
    const int total_runs = 5;
    const int warmup_runs = 2;

    printf("\n[Benchmark Configuration]\n");
    printf("  Base vectors: %ld\n", base.n_rows);
    printf("  Queries: %ld\n", nq);
    printf("  Top-k: %ld\n", topk);
    printf("  Runs: %d (warmup: %d)\n", total_runs, warmup_runs);

    for (const auto& algo : algos) {
        printf("\n----------------------------------------------------------\n");
        printf("  Algorithm: %s\n", algo.c_str());
        printf("----------------------------------------------------------\n");

        // Build index
        printf("\n[Building Index]\n");
        Timer build_timer;

        auto index_result = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            "SPARSE_INVERTED_INDEX", knowhere::Version::GetCurrentVersion().VersionNumber());
        if (!index_result.has_value()) {
            printf("Error: Failed to create index\n");
            continue;
        }
        auto index = index_result.value();

        knowhere::Json build_conf;
        build_conf["metric_type"] = "IP";
        build_conf["inverted_index_algo"] = algo;

        // Create dataset
        auto ds = knowhere::GenDataSet(base.n_rows, base.n_cols, nullptr);
        ds->SetIsSparse(true);
        ds->SetTensor(base_rows.get());

        auto status = index.Build(ds, build_conf);
        if (status != knowhere::Status::success) {
            printf("Error: Failed to build index: %s\n", knowhere::Status2String(status).c_str());
            continue;
        }

        double build_time = build_timer.elapsed_ms();
        printf("  Build time: %.2f ms\n", build_time);

        // Search configuration
        knowhere::Json search_conf;
        search_conf["metric_type"] = "IP";
        search_conf["drop_ratio_search"] = 0.0f;
        search_conf["topk"] = topk;

        // Run benchmark
        printf("\n[Running Search Benchmark]\n");
        std::vector<double> run_times;
        std::vector<int64_t> all_results(nq * topk);
        std::vector<float> all_distances(nq * topk);

        for (int run = 0; run < total_runs; ++run) {
            Timer search_timer;

            for (int64_t q = 0; q < nq; ++q) {
                auto query_ds = knowhere::GenDataSet(1, queries.n_cols, nullptr);
                query_ds->SetIsSparse(true);
                query_ds->SetTensor(&query_rows[q]);

                auto result = index.Search(query_ds, search_conf, knowhere::BitsetView());
                if (!result.has_value()) {
                    printf("Error: Search failed for query %ld\n", q);
                    continue;
                }

                auto ids = result.value()->GetIds();
                auto dists = result.value()->GetDistance();
                memcpy(&all_results[q * topk], ids, topk * sizeof(int64_t));
                memcpy(&all_distances[q * topk], dists, topk * sizeof(float));
            }

            double elapsed = search_timer.elapsed_ms();
            run_times.push_back(elapsed);
            printf("  Run %d: %.2f ms (%.1f QPS)\n", run + 1, elapsed, nq * 1000.0 / elapsed);
        }

        // Calculate average of last (total_runs - warmup_runs) runs
        double avg_time = 0;
        for (int i = warmup_runs; i < total_runs; ++i) {
            avg_time += run_times[i];
        }
        avg_time /= (total_runs - warmup_runs);

        // Compute recall
        float avg_recall = 0;
        if (gt.nq > 0) {
            for (int64_t q = 0; q < nq; ++q) {
                avg_recall += gt.compute_recall(&all_results[q * topk], q, topk);
            }
            avg_recall /= nq;
        }

        printf("\n[Results for %s]\n", algo.c_str());
        printf("  Avg search time: %.2f ms\n", avg_time);
        printf("  QPS: %.1f\n", nq * 1000.0 / avg_time);
        printf("  Recall@%ld: %.4f (%.2f%%)\n", topk, avg_recall, avg_recall * 100);
    }

    printf("\n==========================================================\n");
    printf("  Benchmark Complete\n");
    printf("==========================================================\n");

    return 0;
}
