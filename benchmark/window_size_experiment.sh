#!/bin/bash
# Window size vs data density experiment
# Tests different window sizes on sparse (BM25) and dense (SPLADE) data

set -e

KNOWHERE_DIR="/home/ubuntu/knowhere"
BUILD_DIR="$KNOWHERE_DIR/build/Release"
SPLADE_DATA="/home/ubuntu/data"        # ~127 nnz/doc
BM25_DATA="/home/ubuntu/data/bm25"     # ~25 nnz/doc

# Window sizes to test (in number of documents)
# 0 means no windowing (use n_rows)
WINDOW_SIZES=(8192 16384 32768 65536 131072 262144 0)

echo "=============================================="
echo "Window Size vs Data Density Experiment"
echo "=============================================="
echo ""
echo "Datasets:"
echo "  SPLADE: ~127 nnz/doc (dense)"
echo "  BM25:   ~25 nnz/doc (sparse)"
echo ""
echo "Window sizes: ${WINDOW_SIZES[*]}"
echo ""

# Results file
RESULTS_FILE="/tmp/window_experiment_results.csv"
echo "window_size,dataset,nnz_per_doc,metric,v2_qps" > $RESULTS_FILE

cd $KNOWHERE_DIR

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
    echo "=============================================="
    if [ "$WINDOW_SIZE" -eq 0 ]; then
        echo "Testing: NO WINDOW (full dataset)"
        # Use a very large number to effectively disable windowing
        WINDOW_FLAG="-DMAXSCORE_V2_WINDOW_SIZE=999999999"
    else
        echo "Testing: WINDOW_SIZE = $WINDOW_SIZE"
        WINDOW_FLAG="-DMAXSCORE_V2_WINDOW_SIZE=$WINDOW_SIZE"
    fi
    echo "=============================================="

    # Update cmake cache with new window size
    cd $BUILD_DIR
    sed -i "s|CMAKE_CXX_FLAGS:STRING=.*|CMAKE_CXX_FLAGS:STRING=$WINDOW_FLAG|" CMakeCache.txt

    # Rebuild
    echo "Building with $WINDOW_FLAG..."
    make -j8 benchmark_sparse_algo > /dev/null 2>&1

    # Test on SPLADE (dense) - IP only for speed
    echo "Running on SPLADE (dense, ~127 nnz/doc)..."
    SPLADE_RESULT=$(./benchmark/benchmark_sparse_algo --data-dir $SPLADE_DATA --nq 1000 2>&1 | grep -A5 "Algorithm: DAAT_MAXSCORE_V2 (IP)" | grep "Batch QPS" | awk '{print $3}')
    echo "  SPLADE IP V2 QPS: $SPLADE_RESULT"

    if [ "$WINDOW_SIZE" -eq 0 ]; then
        echo "0,splade,127,IP,$SPLADE_RESULT" >> $RESULTS_FILE
    else
        echo "$WINDOW_SIZE,splade,127,IP,$SPLADE_RESULT" >> $RESULTS_FILE
    fi

    # Test on BM25 (sparse) - IP only for speed
    echo "Running on BM25 (sparse, ~25 nnz/doc)..."
    BM25_RESULT=$(./benchmark/benchmark_sparse_algo --data-dir $BM25_DATA --nq 1000 2>&1 | grep -A5 "Algorithm: DAAT_MAXSCORE_V2 (IP)" | grep "Batch QPS" | awk '{print $3}')
    echo "  BM25 IP V2 QPS: $BM25_RESULT"

    if [ "$WINDOW_SIZE" -eq 0 ]; then
        echo "0,bm25,25,IP,$BM25_RESULT" >> $RESULTS_FILE
    else
        echo "$WINDOW_SIZE,bm25,25,IP,$BM25_RESULT" >> $RESULTS_FILE
    fi

    echo ""
done

echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="
echo ""
cat $RESULTS_FILE | column -t -s,
echo ""
echo "Results saved to: $RESULTS_FILE"
