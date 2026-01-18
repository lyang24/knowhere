# SINDI Algorithm Features - Deep Dive

## Overview

SINDI (Sparse Inverted Non-redundant Distance-calculation Index) from VLDB 2026 introduces three key algorithmic innovations beyond SIMD acceleration. This document explains each feature, its benefits, and tradeoffs.

---

## 1. Window Partitioning (Window Switch Strategy)

### What It Is

Instead of maintaining a single distance array of size N (total documents), SINDI partitions documents into fixed-size windows of λ documents (default λ=100K). During search, it processes one window at a time using a smaller distance array of size λ.

**Traditional Approach:**
```cpp
// Single large array for all N documents
std::vector<float> scores(N);  // N = 1M docs → 4MB array

for (auto& [dim, q_val] : query) {
    auto& posting_list = inverted_index[dim];
    for (auto& [doc_id, value] : posting_list) {
        scores[doc_id] += q_val * value;  // Random access to 4MB array
    }
}
```

**SINDI's Window Approach:**
```cpp
// Small array per window
constexpr size_t WINDOW_SIZE = 100000;  // λ = 100K → 400KB array
std::vector<float> window_scores(WINDOW_SIZE);

for (size_t win = 0; win < num_windows; ++win) {
    std::fill(window_scores.begin(), window_scores.end(), 0.0f);

    for (auto& [dim, q_val] : query) {
        auto& window = inverted_index[dim].get_window(win);
        for (auto& [doc_id_local, value] : window) {
            window_scores[doc_id_local] += q_val * value;  // Access to 400KB array
        }
    }

    // Collect top-k candidates from this window
    merge_window_results(window_scores, global_heap);
}
```

### Data Structure Change

**Traditional Inverted Index:**
```cpp
// One posting list per dimension
struct PostingList {
    std::vector<table_t> doc_ids;     // All documents
    std::vector<float> values;
};

std::vector<PostingList> inverted_index;  // Size = vocabulary_size
```

**SINDI Windowed Index:**
```cpp
// Posting lists partitioned into windows
struct SindiWindow {
    std::vector<table_t> doc_ids;     // Documents in this window
    std::vector<float> values;
};

struct SindiInvertedList {
    std::vector<SindiWindow> windows;  // One window per λ documents
};

std::vector<SindiInvertedList> inverted_index;  // Size = vocabulary_size
// Each dimension has num_windows = ceil(N / λ) windows
```

### Pros

✅ **1. Cache Efficiency - Main Benefit**
- **L3 Cache Fit**: Modern CPUs have 8-32MB L3 cache
  - 400KB score array (λ=100K × 4 bytes) easily fits
  - 4MB score array (1M docs × 4 bytes) may not fit
- **Reduced Cache Misses**: Random writes stay within cached memory
- **2-3× speedup** from cache efficiency alone (according to paper)

✅ **2. SIMD-Friendly**
- Gather/scatter operations are faster when target array is cached
- AVX-512 scatter to cached memory: ~10 cycles
- AVX-512 scatter to uncached memory: ~200+ cycles (cache miss)

✅ **3. Scalability**
- Performance doesn't degrade as dataset grows beyond cache size
- 1M docs vs 100M docs: Same per-window performance

✅ **4. Memory Access Patterns**
- Better memory bandwidth utilization
- Prefetchers work better on smaller, repeatedly accessed arrays

### Cons

❌ **1. Index Size Overhead**
- Each window stores its own doc_ids and values
- Memory overhead: ~10-20% compared to flat posting lists
- Reason: Posting list metadata duplicated across windows

❌ **2. Build Complexity**
- More complex to construct than flat inverted index
- Must partition documents and distribute into windows
- Example:
  ```cpp
  // For document with doc_id=150000:
  size_t window_idx = 150000 / 100000;  // window_idx = 1
  size_t local_id = 150000 % 100000;    // local_id = 50000
  inverted_index[dim].windows[window_idx].add(local_id, value);
  ```

❌ **3. Not Friendly to Incremental Updates**
- Adding new documents may require creating new windows
- Hard to rebalance windows dynamically
- Better suited for batch construction

❌ **4. Window Size Tuning Required**
- Optimal λ depends on CPU cache size
- Too small: More windows → more overhead
- Too large: Loses cache benefits
- Paper suggests λ=100K, but may need tuning per hardware

❌ **5. Multi-threading Overhead**
- Each window processed sequentially
- Can parallelize across windows, but needs coordination
- More complex than parallelizing flat inverted index search

### Optimal Window Size

From the paper (Table 2, page 7):

| λ (Window Size) | Index Size | Search Time | QPS |
|-----------------|------------|-------------|-----|
| 50K             | +5% overhead | 4.5ms | 220 |
| 100K (optimal)  | +10% overhead | 4.1ms | **241** |
| 200K            | +12% overhead | 5.2ms | 190 |
| No windows      | Baseline | 17.5ms | 57 |

**Recommendation**: λ = 100K for CPUs with 8MB+ L3 cache

---

## 2. Mass Ratio Pruning

### What It Is

Mass ratio pruning removes low-value non-zero entries that contribute little to the final similarity score. It retains only entries whose cumulative absolute value sums to α% of the total "mass".

**Mathematical Definition (Definition 6 from paper):**

Given sparse vector **v** and pruning ratio α ∈ [0,1]:

1. Compute total mass: `M = Σ|v_i|` for all dimensions i
2. Sort dimensions by |v_i| in descending order
3. Keep dimensions until cumulative sum ≥ α × M
4. Discard remaining dimensions

**Example:**
```
Original vector:
  dim 10: 0.8
  dim 25: 0.5
  dim 42: 0.3
  dim 67: 0.1
  dim 89: 0.05

Total mass M = 0.8 + 0.5 + 0.3 + 0.1 + 0.05 = 1.75

With α = 0.7 (keep 70% of mass):
  Target mass = 0.7 × 1.75 = 1.225

  Keep dim 10: 0.8    → cumulative = 0.8
  Keep dim 25: 0.5    → cumulative = 1.3 ≥ 1.225 ✓ STOP

  Discard dim 42, 67, 89

Pruned vector:
  dim 10: 0.8
  dim 25: 0.5
```

### Implementation

```cpp
template <typename T>
std::vector<std::pair<size_t, T>>
apply_mass_ratio_pruning(const SparseRow<T>& row, float alpha) {
    std::vector<std::pair<size_t, T>> result;

    if (alpha >= 1.0f) {
        // No pruning - return all entries
        for (size_t i = 0; i < row.size(); ++i) {
            result.emplace_back(row.get_indices()[i], row.get_data()[i]);
        }
        return result;
    }

    // Calculate total mass
    float total_mass = 0.0f;
    for (size_t i = 0; i < row.size(); ++i) {
        total_mass += std::abs(row.get_data()[i]);
    }

    // Sort by absolute value descending
    std::vector<IndexedValue<T>> indexed_values;
    for (size_t i = 0; i < row.size(); ++i) {
        indexed_values.push_back({row.get_indices()[i], row.get_data()[i]});
    }
    std::sort(indexed_values.begin(), indexed_values.end(),
              [](const auto& a, const auto& b) {
                  return std::abs(a.value) > std::abs(b.value);
              });

    // Keep entries until target mass reached
    float cumulative_mass = 0.0f;
    float target_mass = alpha * total_mass;

    for (const auto& iv : indexed_values) {
        result.emplace_back(iv.index, iv.value);
        cumulative_mass += std::abs(iv.value);
        if (cumulative_mass >= target_mass) {
            break;
        }
    }

    return result;
}
```

### Two Types of Pruning

**1. Document Pruning (α - alpha)**
- Applied to database vectors during index construction
- Prunes each document to keep α% of mass (e.g., α=0.5 = keep 50%)
- Reduces index size and search computation

**2. Query Pruning (β - beta)**
- Applied to query vectors at search time
- Prunes query to keep β% of mass (e.g., β=0.2 = keep 20%)
- Reduces number of posting lists accessed

### Pros

✅ **1. Reduced Computation - Main Benefit**
- Fewer non-zero entries to process
- Example: SPLADE vectors avg 126 nnz → α=0.5 → ~50 nnz
- **1.5-4× speedup** from reduced computation

✅ **2. Smaller Index Size**
- Document pruning reduces posting list lengths
- Example: α=0.5 → ~50% index size reduction
- Important for large-scale systems

✅ **3. Memory Bandwidth Savings**
- Less data to fetch from memory
- Posting lists are shorter → fewer cache lines loaded

✅ **4. Better Cache Utilization**
- Accessing fewer posting lists → better cache hit rate
- Query pruning (β) especially effective

✅ **5. Minimal Accuracy Loss**
- Paper shows: α=0.5, β=0.2 → Recall@50 = 99.1%
- Low-value entries contribute little to ranking
- Top-k results mostly unchanged

### Cons

❌ **1. Quality Degradation**
- Removes information → can miss relevant documents
- Tradeoff: More aggressive pruning (lower α, β) → lower recall
- Example from paper (SPLADE-1M dataset):
  | α    | β    | Recall@50 | QPS  |
  |------|------|-----------|------|
  | 1.0  | 1.0  | 100.0%    | 58   |
  | 0.7  | 0.3  | 99.8%     | 145  |
  | 0.5  | 0.2  | 99.1%     | 241  |
  | 0.3  | 0.1  | 95.2%     | 412  |

❌ **2. Parameter Tuning Required**
- Optimal α and β depend on:
  - Dataset characteristics (sparsity, value distribution)
  - Recall requirements
  - Performance targets
- No universal "best" values

❌ **3. Build Time Overhead**
- Sorting each vector by absolute value
- O(k log k) per vector where k = avg non-zeros
- Example: 8.8M docs × 126 nnz × log(126) ≈ 7.5 billion comparisons

❌ **4. Non-deterministic for Equal Values**
- If multiple dimensions have same |value|, tie-breaking is arbitrary
- Can lead to slightly different results across runs
- Usually not a problem in practice (values rarely exactly equal)

❌ **5. Not Suitable for All Metrics**
- Works well for Inner Product and Cosine similarity
- May not work for metrics where all dimensions matter equally
- Example: Hamming distance, Jaccard - pruning breaks semantics

❌ **6. Incompatible with Exact Search**
- Pruning inherently introduces approximation
- Cannot guarantee 100% recall
- Not suitable when exact results required

### Parameter Selection Guidelines

From the paper's experiments:

**Conservative (High Recall):**
- α = 0.7, β = 0.3
- Recall@50 ≈ 99.8%
- 2-3× speedup

**Balanced (Recommended):**
- α = 0.5, β = 0.2
- Recall@50 ≈ 99.0-99.5%
- 4-5× speedup

**Aggressive (High Speed):**
- α = 0.3, β = 0.1
- Recall@50 ≈ 95-97%
- 7-10× speedup

---

## 3. Two-Phase Search

### What It Is

Two-phase search splits the retrieval process into:

1. **Phase 1: Coarse Recall** - Fast approximate search using pruned index and pruned query to get γ candidates (γ > k)
2. **Phase 2: Exact Reordering** - Recompute exact distances for γ candidates using original unpruned vectors, return top-k

### Architecture

```cpp
class SindiIndex {
    // Phase 1: Pruned index (fast but approximate)
    std::vector<SindiInvertedList<float>> pruned_index_;  // Built with α pruning

    // Phase 2: Original vectors (slow but exact)
    std::vector<SparseRow<float>> original_vectors_;      // Unpruned documents
};
```

### Search Algorithm

```cpp
void Search(const SparseRow<float>& query, size_t k,
            float* distances, label_t* labels) {

    // === Phase 1: Coarse Recall ===
    // Prune query to keep β% of mass
    auto pruned_query = apply_mass_ratio_pruning(query, config_.query_pruning_ratio);

    // Search pruned index with pruned query
    // Get top-γ candidates (γ > k, e.g., γ=500 for k=50)
    MaxMinHeap<float> coarse_heap(config_.reorder_pool_size);

    for (size_t win = 0; win < num_windows_; ++win) {
        std::vector<float> window_scores(config_.window_size, 0.0f);

        // Compute approximate scores using pruned index
        for (auto& [dim, q_val] : pruned_query) {
            auto& window = pruned_index_[dim].get_window(win);
            // SIMD distance computation here...
        }

        // Collect top candidates from this window
        for (size_t i = 0; i < window_scores.size(); ++i) {
            if (window_scores[i] > 0) {
                coarse_heap.push(win * config_.window_size + i, window_scores[i]);
            }
        }
    }

    // === Phase 2: Exact Reordering ===
    // Get γ candidates from Phase 1
    std::vector<label_t> candidates = coarse_heap.get_labels();

    // Recompute exact distances using ORIGINAL vectors
    MaxMinHeap<float> exact_heap(k);
    for (auto doc_id : candidates) {
        float exact_score = compute_exact_distance(
            query,                      // Original query (unpruned)
            original_vectors_[doc_id]   // Original document (unpruned)
        );
        exact_heap.push(doc_id, exact_score);
    }

    // Return top-k from exact heap
    collect_result(exact_heap, distances, labels);
}
```

### Pros

✅ **1. Accuracy Recovery - Main Benefit**
- Phase 1 pruning causes some errors
- Phase 2 corrects errors by recomputing exact scores
- Achieves high recall (99%+) despite aggressive pruning

✅ **2. Speed with Quality Balance**
- Phase 1: Fast but approximate (processes γ candidates)
- Phase 2: Slow but exact (only k candidates matter)
- Example: γ=500, k=50
  - Phase 1 processes 500 candidates quickly
  - Phase 2 recomputes exact scores for 500 (not all N)
  - 10× faster than computing exact for all N

✅ **3. Tunable Tradeoff**
- Larger γ (reorder pool) → higher recall, slower search
- Smaller γ → lower recall, faster search
- Can adjust γ at query time (no index rebuild needed)

✅ **4. Robust to Pruning Errors**
- Even if Phase 1 ranks incorrectly, Phase 2 can fix
- As long as true top-k are in top-γ, final result is exact
- Paper shows: γ=500 for k=50 → 99%+ recall

✅ **5. Compatible with Other Optimizations**
- Works with any fast approximate method in Phase 1
- Can use WAND, MaxScore, or simple TAAT in Phase 1
- Phase 2 is embarrassingly parallel (independent score computations)

### Cons

❌ **1. Extra Memory Requirement**
- Must store BOTH pruned index and original vectors
- Memory overhead: ~1.5-2× compared to single index
- Example: SPLADE-1M with α=0.5
  - Pruned index: 4GB
  - Original vectors: 4GB
  - Total: 8GB (vs 4GB for single index)

❌ **2. Complexity in Implementation**
- Need to maintain two representations
- More code to manage and debug
- Serialization/deserialization more complex

❌ **3. Phase 2 Overhead**
- Recomputing γ distances is not free
- For small datasets or small γ, overhead can dominate
- Example: γ=500 docs × 126 nnz/doc × 49 nnz/query ≈ 3M operations
  - If Phase 1 is very fast, Phase 2 becomes bottleneck

❌ **4. Still Not 100% Exact**
- Only guarantees top-k are exact IF they're in top-γ from Phase 1
- If true top-k is ranked at position 600, but γ=500, it will be missed
- Choosing γ is critical

❌ **5. Load Balancing Issues**
- Phase 1 can be highly parallel
- Phase 2 is serial (in simple implementation) or needs coordination
- May not fully utilize all CPU cores

❌ **6. Cache Pollution**
- Phase 2 accesses original vectors randomly
- Can evict useful data from cache loaded in Phase 1
- Temporal locality is lost between phases

### Parameter Selection

From the paper (Table 4):

| k (top-k) | γ (reorder pool) | Recall@k | Phase 1 Time | Phase 2 Time |
|-----------|------------------|----------|--------------|--------------|
| 10        | 100              | 98.5%    | 3.2ms        | 0.3ms        |
| 50        | 500              | 99.2%    | 3.2ms        | 1.1ms        |
| 100       | 1000             | 99.6%    | 3.2ms        | 2.3ms        |

**General Rule**: γ = 10 × k is a good starting point

---

## Combined Impact

When all three features work together:

| Feature                  | Speedup | Recall Impact |
|--------------------------|---------|---------------|
| Window Partitioning      | 2-3×    | None (exact)  |
| Mass Ratio Pruning       | 1.5-4×  | -0.5% to -5%  |
| Two-Phase Search         | 1.2-1.5× | +3% to +5% (recovery) |
| SIMD (AVX-512)           | 1.5-3×  | None (exact)  |

**Total Speedup**: 2 × 1.5 × 1.2 × 1.5 = **5.4×** (conservative estimate)

Paper reports: **4-26× speedup** depending on dataset and parameters

---

## Comparison Table

| Aspect                | Window Partitioning | Mass Ratio Pruning | Two-Phase Search |
|-----------------------|---------------------|-------------------|------------------|
| **Type**              | Optimization        | Approximation     | Hybrid           |
| **Affects Accuracy**  | ❌ No              | ✅ Yes (reduces)  | ✅ Yes (recovers) |
| **Memory Overhead**   | +10-20%             | -50% (saves)      | +100% (doubles)  |
| **Build Complexity**  | Medium              | Low               | Medium           |
| **Runtime Tunable**   | ❌ No              | ✅ Yes (β)        | ✅ Yes (γ)       |
| **Works on All CPUs** | ✅ Yes             | ✅ Yes            | ✅ Yes           |
| **Main Benefit**      | Cache efficiency    | Less computation  | Accuracy recovery |
| **Speedup**           | 2-3×                | 1.5-4×            | 1.2-1.5×         |

---

## Implementation Recommendations

### When to Use Each Feature

**Window Partitioning:**
- ✅ Always use for large datasets (1M+ documents)
- ✅ Especially beneficial with SIMD
- ❌ Skip for small datasets (<100K docs) - overhead dominates

**Mass Ratio Pruning:**
- ✅ Use when recall requirements allow (95-99% acceptable)
- ✅ Essential for high-dimensional sparse vectors (SPLADE, BGE-M3)
- ❌ Skip when 100% recall required (exact search)

**Two-Phase Search:**
- ✅ Use when using aggressive pruning (α<0.7, β<0.3)
- ✅ When memory allows (need to store original vectors)
- ❌ Skip for very high recall requirements (99.9%+) - just use less pruning instead

### Suggested Configurations

**Maximum Speed (95% recall acceptable):**
```cpp
SindiConfig config;
config.window_size = 100000;
config.doc_pruning_ratio = 0.3;    // α = 0.3
config.query_pruning_ratio = 0.1;  // β = 0.1
config.reorder_pool_size = 1000;   // γ = 10k for k=100
config.enable_simd = true;
```

**Balanced (99% recall):**
```cpp
SindiConfig config;
config.window_size = 100000;
config.doc_pruning_ratio = 0.5;    // α = 0.5
config.query_pruning_ratio = 0.2;  // β = 0.2
config.reorder_pool_size = 500;    // γ = 5k for k=50
config.enable_simd = true;
```

**High Quality (99.5%+ recall):**
```cpp
SindiConfig config;
config.window_size = 100000;
config.doc_pruning_ratio = 0.7;    // α = 0.7
config.query_pruning_ratio = 0.3;  // β = 0.3
config.reorder_pool_size = 200;    // γ = 2k for k=10
config.enable_simd = true;
```

---

## References

- SINDI Paper: "Sparse Inverted Non-redundant Distance-calculation Index" (VLDB 2026)
- Algorithm 1 (page 5): Index construction with mass ratio pruning
- Algorithm 2 (page 6): Window-based search
- Algorithm 4 (page 7): Two-phase search with reordering
- Table 2 (page 7): Window size ablation study
- Table 4 (page 9): Reorder pool size impact
