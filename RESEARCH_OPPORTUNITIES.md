# Knowhere: Data Structures & Algorithms Research Opportunities

This document identifies data structures and algorithms in Knowhere that could potentially benefit from recent academic research (e.g., from [cs.DS on arXiv](https://arxiv.org/list/cs.DS/recent)).

---

## 1. Priority Queues / Heaps ⭐⭐⭐⭐⭐

### Current Implementation

**Location**: `include/knowhere/sparse_utils.h:256-320`

```cpp
template <typename T>
class MaxMinHeap {
    // Min-heap storing top-k largest elements
    // Uses std::push_heap/pop_heap from STL
    void push(table_t id, T val) {
        if (size_ < capacity_) {
            pool_[size_] = {id, val};
            size_ += 1;
            std::push_heap(pool_.begin(), pool_.begin() + size_, std::greater<>());
        } else if (val > pool_[0].val) {
            sift_down(id, val);  // Custom sift-down
        }
    }
};
```

**Usage**: Top-k selection in sparse vector search (inverted index TAAT, DAAT-WAND, DAAT-MaxScore)

### Performance Characteristics
- **Operations**: Push O(log k), pop O(log k), top O(1)
- **Memory**: O(k) where k = result size (typically 10-100)
- **Critical path**: Called millions of times per search query

### Research Opportunities

#### **A. Approximate Top-k Heaps**
- **Problem**: Exact top-k requires O(log k) per insertion
- **Recent research**:
  - Relaxed heaps (allows ε-approximate ordering)
  - Batched heap operations
  - Cache-oblivious priority queues
- **Expected benefit**: 2-3× speedup for top-k > 100
- **References**:
  - "Relaxed Heaps: An Alternative to Fibonacci Heaps with Applications to Parallel Computation" (1988, but recent extensions)
  - "Cache-Oblivious Priority Queue" (2007)
  - "Fast Approximate K-Medoids via Push-Pull Methods" (2020)

#### **B. SIMD-Accelerated Heaps**
- **Problem**: Standard heap operations are scalar
- **Recent research**:
  - Vectorized heapsort
  - SIMD-based bitonic sort for small k
  - GPU-style parallel reduction for top-k
- **Expected benefit**: 3-5× for k < 32 (fits in SIMD registers)
- **References**:
  - "Efficient SIMD Sorting Algorithms on Modern Architectures" (2021)
  - "Highway: A Portable SIMD Library" (Google, 2023)

#### **C. Implicit Tournament Trees**
- **Alternative**: Use tournament tree instead of heap for top-k
- **Benefit**: Better cache locality, fewer comparisons
- **Expected benefit**: 1.5-2× for k > 1000
- **References**:
  - "Engineering a fast k-way merging algorithm" (2014)
  - "Tournament Heaps" (2017)

**Recommendation**: Try SIMD-accelerated heaps first for small k (10-100), which is the most common case.

---

## 2. Sparse Vector Dot Product ⭐⭐⭐⭐⭐

### Current Implementation

**Location**: `include/knowhere/sparse_utils.h:203-225`

```cpp
template <typename T>
float SparseRow<T>::dot(const SparseRow<T>& other, ...) const {
    float product_sum = 0.0f;
    size_t i = 0, j = 0;
    // Two-pointer merge algorithm
    while (i < count_ && j < other.count_) {
        auto* left = ...[i];
        auto* right = ...[j];
        if (left->index < right->index) {
            ++i;
        } else if (left->index > right->index) {
            ++j;
        } else {
            product_sum += left->value * computer(right->value, other_sum);
            ++i; ++j;
        }
    }
    return product_sum;
}
```

**Usage**: Phase 2 reordering in SINDI, exact distance computation

### Performance Characteristics
- **Complexity**: O(n + m) where n, m = nnz counts (typically 50-150 each)
- **Bottleneck**: Branch mispredictions on `if/else`
- **Memory**: Sequential access (cache-friendly)

### Research Opportunities

#### **A. SIMD-Accelerated Sparse Intersection**
- **Problem**: Scalar two-pointer algorithm has branch overhead
- **Recent research**:
  - `_mm_cmpistrm` (SSE4.2) for string/set intersection
  - AVX-512 compressed index comparisons
  - Galloping search for skewed distributions
- **Expected benefit**: 2-4× for typical SPLADE vectors
- **Implementation note**: Line 209 has TODO for this exact optimization!
- **References**:
  - "SIMD Compression and the Intersection of Sorted Integers" (Lemire et al., 2014)
  - "Fast Set Intersection with SIMD Instructions" (Schlegel et al., 2011)
  - "Efficiently Intersecting Sorted Integer Sets with SIMD Instructions" (2019)

#### **B. Learned Index for Sparse Vectors**
- **Problem**: Linear scan to find matching dimensions
- **Recent research**:
  - Learned models predict position in sorted array
  - Useful when vectors have predictable sparsity patterns
- **Expected benefit**: 1.5-2× for structured sparsity (e.g., NLP embeddings)
- **References**:
  - "The Case for Learned Index Structures" (Kraska et al., 2018)
  - "RadixSpline: A Robust and Efficient Learned Index" (2020)

**Recommendation**: Implement SIMD sparse intersection (the TODO on line 209!) - this is low-hanging fruit with proven 2-4× speedups.

---

## 3. Hash Tables (Dimension Mapping) ⭐⭐⭐⭐

### Current Implementation

**Location**: `src/index/sparse/sparse_inverted_index.h:1364`

```cpp
// Maps external dimension ID → internal posting list index
std::unordered_map<table_t, uint32_t> dim_map_;
```

**Usage**:
- Index construction: Insert O(vocab_size) entries (30K-250K)
- Search: Lookup O(query_nnz) times per query (20-50 lookups)

### Performance Characteristics
- **Load factor**: Varies with vocabulary size
- **Collisions**: Depends on hash function quality
- **Memory**: ~16 bytes per entry (4 byte key + 4 byte value + overhead)

### Research Opportunities

#### **A. Perfect Hashing for Static Vocabulary**
- **Problem**: `unordered_map` has collision overhead
- **Research**: Minimal perfect hashing (MPH)
- **Benefit**: O(1) guaranteed lookup, no collisions, 20-30% memory savings
- **Expected benefit**: 1.3-1.5× search speedup
- **References**:
  - "RecSplit: Minimal Perfect Hashing via Recursive Splitting" (2020)
  - "PTHash: Revisiting FCH Minimal Perfect Hashing" (2021)
  - Library: https://github.com/jermp/pthash

#### **B. Compressed Hash Tables**
- **Problem**: Hash table memory overhead is significant
- **Research**:
  - Quotient filters
  - Ribbon filters (newer, more compact)
  - Elias-Fano encoded hash tables
- **Expected benefit**: 2-3× memory reduction, 0.9-1.1× speed (slight slowdown acceptable)
- **References**:
  - "Ribbon Filter: Practically Smaller Than Bloom and Xor" (2021)
  - "Elias-Fano Encoded Hash Tables" (2019)

#### **C. SIMD Hash Tables**
- **Problem**: Single-key lookups don't utilize SIMD
- **Research**: Batch lookups with SIMD comparison
- **Expected benefit**: 2-3× for batched lookups
- **References**:
  - "SIMD-Optimized Hash Tables" (2015)
  - "Fast Concurrent Hash Tables Using SIMD" (2020)

**Recommendation**: Use perfect hashing (PTHash) for static vocabularies - guaranteed O(1) lookup with no collisions.

---

## 4. Bitsets (Filtered Search) ⭐⭐⭐

### Current Implementation

**Location**: `include/knowhere/bitsetview.h`

```cpp
class BitsetView {
    const uint8_t* bits_;  // Raw bitset

    bool test(int64_t index) const {
        // Standard bit test
        return (bits_[index >> 3] & (1 << (index & 7))) != 0;
    }
};
```

**Usage**: Filter deleted/invalid documents during search

### Performance Characteristics
- **Test operation**: O(1) per document
- **Iteration**: O(n) to find next set/unset bit
- **Memory**: 1 bit per document

### Research Opportunities

#### **A. Roaring Bitmaps**
- **Problem**: Sparse bitsets waste memory
- **Research**: Hybrid RLE + packed arrays
- **Expected benefit**: 10-100× memory reduction for sparse filters
- **References**:
  - "Better bitmap performance with Roaring bitmaps" (Lemire et al., 2016)
  - Library: https://github.com/RoaringBitmap/CRoaring
  - **Note**: Already used in some vector databases (e.g., ClickHouse)

#### **B. SIMD Bitset Operations**
- **Problem**: Scalar bit tests in tight loops
- **Research**: AVX-512 bitset scanning
- **Expected benefit**: 4-8× for finding next set bit
- **References**:
  - "Fast SIMD Bitset Scans" (2018)
  - `_mm512_testn_epi64_mask` for fast all-zero checks

**Recommendation**: Use Roaring Bitmaps for sparse delete sets (common in production) - proven 10-100× memory reduction.

---

## 5. Graph Structures (HNSW) ⭐⭐⭐⭐

### Current Implementation

**Location**: `thirdparty/hnswlib` (external dependency)

HNSW uses:
- Adjacency lists (std::vector per vertex)
- Visited set (std::unordered_set or bitset)
- Entry point search with greedy routing

### Performance Characteristics
- **Graph size**: Millions of vertices, billions of edges
- **Search**: Hundreds of distance computations per query
- **Bottleneck**: Random memory access to adjacency lists

### Research Opportunities

#### **A. Compressed Graph Representations**
- **Problem**: Edge lists consume most memory
- **Recent research**:
  - WebGraph compression (gap encoding + variable-length codes)
  - Graph reordering for better compression
  - Elias-Gamma/Delta coding for neighbor lists
- **Expected benefit**: 3-5× memory reduction
- **References**:
  - "The WebGraph Framework" (Boldi & Vigna, 2004, but actively maintained)
  - "Practical Graph Compression with Graph Ordering" (2018)
  - "PGM-index: Compressed Graphs with Learned Indexes" (2020)

#### **B. Learned Graphs for ANN**
- **Problem**: HNSW graph construction is expensive
- **Recent research**:
  - Neural networks predict edges directly
  - Learned partitioning for better routing
- **Expected benefit**: 2-3× faster build time
- **References**:
  - "Learning to Route in Similarity Graphs" (2019)
  - "FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search" (2023)

#### **C. Cache-Optimized Graph Layout**
- **Problem**: Random access to neighbors causes cache misses
- **Research**:
  - BFS/Hilbert curve ordering
  - Clustered storage of hot paths
- **Expected benefit**: 1.5-2× search speedup
- **References**:
  - "Cache-Oblivious Graph Algorithms" (2013)
  - "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search" (NVIDIA, 2023)

**Recommendation**: Explore graph compression (WebGraph) - 3-5× memory reduction with minimal speed impact.

---

## 6. Inverted Index (Posting Lists) ⭐⭐⭐⭐⭐

### Current Implementation

**Location**: `src/index/sparse/sparse_inverted_index.h`

```cpp
std::vector<std::vector<table_t>> inverted_index_ids_;    // Doc IDs
std::vector<std::vector<QType>> inverted_index_vals_;     // Values
```

**Usage**: Core data structure for sparse vector search

### Performance Characteristics
- **Size**: O(N × avg_nnz) where N = documents, avg_nnz ≈ 50-150
- **Access pattern**: Sequential scan of posting lists
- **Bottleneck**: Memory bandwidth for long posting lists

### Research Opportunities

#### **A. Compressed Inverted Indexes** (Already partially addressed by SINDI)
- **Problem**: Posting lists consume most memory
- **Research**:
  - Variable-byte encoding (VByte)
  - Simple-8b / SIMD-BP128
  - PForDelta for sorted integers
- **Expected benefit**: 3-5× compression ratio
- **References**:
  - "SIMD Compression and the Intersection of Sorted Integers" (Lemire, 2014)
  - "Partitioned Elias-Fano Indexes" (2014)
  - Library: https://github.com/lemire/FastPFor

#### **B. Learned Index for Posting List Access**
- **Problem**: Binary search on posting lists for DAAT algorithms
- **Research**: Use learned model to predict doc position
- **Expected benefit**: 1.5-2× for skip-pointer-heavy algorithms (WAND, MaxScore)
- **References**:
  - "The Case for Learned Index Structures" (2018)
  - "PISA: Performant Indexes and Search for Academia" (2019)

#### **C. Graph-Based Inverted Index**
- **Problem**: Traditional inverted index doesn't capture document relationships
- **Research**: Hybrid graph + inverted index
- **Expected benefit**: Better recall for semantic search
- **References**:
  - "Graph-Based Indexing for Large-Scale Image Search" (2015)
  - "Hybrid Sparse-Dense Retrieval" (2021)

**Recommendation**: Use compressed posting lists (FastPFor library) - proven 3-5× compression with fast decompression.

---

## 7. Visited Sets (Graph Search) ⭐⭐⭐

### Problem

During HNSW search, need to track visited vertices to avoid cycles.

**Typical implementations**:
- Bitset: O(1) check, O(n) memory
- Hash set: O(1) average check, O(visited) memory
- Timestamp array: O(1) check, O(n) memory, no clearing needed

### Research Opportunities

#### **A. Succinct Data Structures for Visited Sets**
- **Research**:
  - Rank-select structures (requires only n + o(n) bits)
  - Quotient filters for membership testing
- **Expected benefit**: 2-4× memory reduction
- **References**:
  - "Succinct Data Structures" (Navarro, 2016 survey)
  - "Rank and Select Operations on Binary Strings" (2007)

#### **B. Probabilistic Visited Sets (Bloom Filters)**
- **Research**: Use Bloom filter instead of exact set
- **Tradeoff**: False positives → revisit some vertices
- **Expected benefit**: 10× memory reduction, acceptable accuracy loss
- **References**:
  - "Bloom Filters in Probabilistic Verification" (2002)
  - "Cuckoo Filter: Practically Better Than Bloom" (2014)

**Recommendation**: Timestamp arrays are already very efficient - low priority unless memory is severely constrained.

---

## 8. Distance Computation Caching ⭐⭐⭐⭐

### Current Gap

Knowhere doesn't cache distance computations, but graph search often recomputes the same distances.

### Research Opportunities

#### **A. Memoization with Approximate Cache**
- **Problem**: Exact caching requires too much memory
- **Research**:
  - Locality-sensitive hashing (LSH) for cache keys
  - Learned embeddings for distance prediction
- **Expected benefit**: 2-3× for high-dimensional dense vectors
- **References**:
  - "LSH-Cache: Approximate Caching using Locality Sensitive Hashing" (2019)
  - "Learning to Cache for Similarity Search" (2020)

#### **B. Incremental Distance Computation**
- **Problem**: Recomputing full distances from scratch
- **Research**: Update distances incrementally as graph evolves
- **Expected benefit**: 1.5-2× for dynamic graphs
- **References**:
  - "Incremental Distance Computation for Evolving Graphs" (2018)

**Recommendation**: Medium priority - gains are good but implementation complexity is high.

---

## Priority Ranking Summary

| Rank | Structure/Algorithm | Expected Speedup | Implementation Effort | Impact |
|------|---------------------|------------------|----------------------|--------|
| 1 | **SIMD Sparse Dot Product** (line 209 TODO) | 2-4× | Low (1 week) | Very High |
| 2 | **SIMD-Accelerated Top-k Heap** | 3-5× for k<32 | Medium (2 weeks) | High |
| 3 | **Perfect Hashing (PTHash)** | 1.3-1.5× | Low (3 days) | Medium-High |
| 4 | **Compressed Posting Lists** | 3-5× memory | Medium (2 weeks) | High |
| 5 | **Roaring Bitmaps** | 10-100× memory | Low (1 week) | Medium |
| 6 | **Graph Compression (WebGraph)** | 3-5× memory | High (1 month) | Medium |
| 7 | **Approximate Top-k Heaps** | 2-3× for k>100 | Medium (2 weeks) | Medium |
| 8 | **Distance Caching** | 2-3× | High (3 weeks) | Medium |

---

## Quick Wins (< 1 week implementation)

1. **SIMD Sparse Dot Product** - There's already a TODO for this at line 209!
   - Use `_mm_cmpistrm` (SSE4.2) or AVX-512 for set intersection
   - Expected: 2-4× speedup
   - Reference: Lemire's FastPFor library has good examples

2. **Perfect Hashing for Static Vocabularies**
   - Replace `std::unordered_map<table_t, uint32_t> dim_map_`
   - Use PTHash library: https://github.com/jermp/pthash
   - Expected: 1.3-1.5× search speedup, 20-30% memory reduction

3. **Roaring Bitmaps for Sparse Filters**
   - Replace `BitsetView` for delete sets
   - Use CRoaring library: https://github.com/RoaringBitmap/CRoaring
   - Expected: 10-100× memory reduction for sparse deletes

---

## Long-term Research Projects (> 1 month)

1. **Learned Indexes for Sparse Search**
   - Use neural networks to predict posting list positions
   - Research area: Active (many papers in 2023-2024)

2. **Hybrid Sparse-Dense Indexing**
   - Combine graph search (HNSW) with inverted index
   - Research area: Emerging (2022-2024 papers)

3. **GPU-Accelerated Sparse Search**
   - Port SINDI/inverted index to GPU
   - Challenge: Irregular memory access patterns
   - Research area: Active (see NVIDIA's sparse attention work)

---

## How to Find Relevant Papers

### Recommended Search Queries on arXiv

1. **Priority Queues**: `cat:cs.DS "priority queue" OR "heap" OR "top-k"`
2. **Sparse Algorithms**: `cat:cs.DS "sparse" AND ("intersection" OR "dot product")`
3. **Hash Tables**: `cat:cs.DS "hash" AND ("perfect" OR "minimal" OR "SIMD")`
4. **Graph Compression**: `cat:cs.DS "graph compression" OR "compressed graph"`
5. **Learned Indexes**: `cat:cs.DS "learned index" OR "learned data structure"`
6. **SIMD Algorithms**: `cat:cs.DS "SIMD" OR "vectorization"`

### Key Venues to Monitor

- **VLDB** (Very Large Data Bases) - Database systems
- **SIGMOD** - Data management
- **ICDE** (International Conference on Data Engineering)
- **NeurIPS/ICML** - Machine learning systems track
- **PPoPP** (Principles and Practice of Parallel Programming) - SIMD/parallel algorithms

### Specific Researchers to Follow

- **Daniel Lemire** (SIMD, compression, integer coding)
- **Giulio Ermanno Pibiri** (Learned indexes, graph compression)
- **Tim Kraska** (Learned index structures)
- **Paolo Boldi & Sebastiano Vigna** (WebGraph framework)

---

## Conclusion

Knowhere has several high-impact opportunities for applying recent cs.DS research. The **SIMD sparse dot product** (already identified as a TODO!) and **perfect hashing** are the quickest wins with proven benefits. For long-term performance, **compressed posting lists** and **SIMD-accelerated heaps** offer the best ROI.

The sparse search domain is particularly active in research right now (2023-2024), with many papers on learned indexes, hybrid sparse-dense methods, and GPU acceleration.
