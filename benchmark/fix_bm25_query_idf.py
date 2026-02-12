#!/usr/bin/env python3
"""Fix BM25 query CSR files by adding IDF weights.

The original build_bm25_dataset.py stored raw term frequencies as query weights,
but proper BM25 requires IDF * tf. This script:
1. Loads base CSR to compute document frequencies
2. Computes IDF for each term
3. Rewrites query CSR with IDF-weighted values
4. Regenerates ground truth using the corrected scoring

Usage:
    python3 fix_bm25_query_idf.py --data-dir ~/data/msmarco_full_bm25
"""

import argparse
import struct
import time
import numpy as np
from pathlib import Path
from scipy import sparse


def load_csr(path):
    with open(path, "rb") as f:
        n_rows, n_cols, nnz = struct.unpack("qqq", f.read(24))
        indptr = np.frombuffer(f.read((n_rows + 1) * 8), dtype=np.int64).copy()
        indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32).copy()
        data = np.frombuffer(f.read(nnz * 4), dtype=np.float32).copy()
    print(f"  Loaded {path}: {n_rows} rows, {n_cols} cols, {nnz} nnz")
    return n_rows, n_cols, nnz, indptr, indices, data


def save_csr(path, n_rows, n_cols, indptr, indices, data):
    nnz = len(indices)
    with open(path, "wb") as f:
        f.write(struct.pack("qqq", n_rows, n_cols, nnz))
        f.write(np.array(indptr, dtype=np.int64).tobytes())
        f.write(np.array(indices, dtype=np.int32).tobytes())
        f.write(np.array(data, dtype=np.float32).tobytes())
    print(f"  Saved {path}: {n_rows} rows, {n_cols} cols, {nnz} nnz")


def save_gt(path, gt_matrix, k):
    nq = gt_matrix.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("ii", nq, k))
        for i in range(nq):
            row = gt_matrix[i, :k].astype(np.int32)
            f.write(row.tobytes())
    print(f"  Saved {path}: {nq} queries, k={k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--gt-k", type=int, default=100)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load base and query CSR
    print("[Loading base CSR]")
    b_rows, b_cols, b_nnz, b_indptr, b_indices, b_data = load_csr(data_dir / "base_small.csr")

    print("[Loading query CSR]")
    q_rows, q_cols, q_nnz, q_indptr, q_indices, q_data = load_csr(data_dir / "queries.dev.csr")

    n_cols = max(b_cols, q_cols)
    N = b_rows

    # Compute document frequencies: count unique docs per term
    # In CSR, each (row, col) pair is unique, so bincount on indices gives
    # number of entries per column = number of docs containing each term = df
    print("[Computing document frequencies]")
    t0 = time.time()
    df = np.bincount(b_indices, minlength=n_cols).astype(np.float64)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Terms with df>0: {np.sum(df > 0)}")
    print(f"  Max df: {df.max():.0f}, median df (nonzero): {np.median(df[df>0]):.0f}")

    # Compute IDF (Lucene/Robertson variant, matches common BM25 implementations)
    # IDF(t) = log(1 + (N - df + 0.5) / (df + 0.5))
    idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
    idf[df == 0] = 0.0
    print(f"  IDF range: [{idf[idf>0].min():.3f}, {idf.max():.3f}]")

    # Rewrite query CSR with IDF * tf weights
    print("[Rewriting query CSR with IDF weights]")
    new_q_data = q_data.copy()
    for i in range(q_nnz):
        dim = q_indices[i]
        new_q_data[i] = q_data[i] * idf[dim]

    # Remove zero-weight query terms (terms not in corpus)
    # Rebuild CSR without them
    new_indptr = [0]
    new_indices = []
    new_data_list = []
    for i in range(q_rows):
        s, e = q_indptr[i], q_indptr[i + 1]
        for j in range(s, e):
            if new_q_data[j] > 0:
                new_indices.append(q_indices[j])
                new_data_list.append(new_q_data[j])
        new_indptr.append(len(new_indices))

    new_indices = np.array(new_indices, dtype=np.int32)
    new_data_arr = np.array(new_data_list, dtype=np.float32)
    new_indptr = np.array(new_indptr, dtype=np.int64)

    # Backup original
    orig_path = data_dir / "queries.dev.csr.orig_no_idf"
    if not orig_path.exists():
        import shutil
        shutil.copy(data_dir / "queries.dev.csr", orig_path)
        print(f"  Backed up original to {orig_path}")

    save_csr(data_dir / "queries.dev.csr", q_rows, n_cols, new_indptr, new_indices, new_data_arr)

    # Verify: print first query's weights
    s, e = new_indptr[0], new_indptr[1]
    print(f"  Query 0: {e-s} terms, weights: {new_data_arr[s:min(s+5,e)]}")

    # Regenerate ground truth with proper BM25 scoring
    # score(q,d) = Σ_t IDF(t) * qtf(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1*(1-b+b*(dl/avgdl)))
    print("\n[Regenerating ground truth]")
    k1 = args.bm25_k1
    b = args.bm25_b

    # Compute doc lengths using scipy (sum of raw TF values per row)
    print("  Computing doc lengths...")
    base_mat_raw = sparse.csr_matrix((b_data, b_indices, b_indptr), shape=(b_rows, n_cols))
    doc_lens = np.array(base_mat_raw.sum(axis=1)).ravel()
    avgdl = np.mean(doc_lens)
    print(f"  avgdl: {avgdl:.2f}")

    # Build normalized base matrix: TF_norm(tf, dl) = tf * (k1+1) / (tf + k1*(1-b+b*(dl/avgdl)))
    print("  Normalizing base matrix with BM25 TF formula...")
    t0 = time.time()
    # Vectorized: expand doc_lens to match each nonzero entry
    doc_ids_per_entry = np.repeat(np.arange(b_rows), np.diff(b_indptr))
    dl_per_entry = doc_lens[doc_ids_per_entry]
    norm_data = (b_data * (k1 + 1) / (b_data + k1 * (1 - b + b * (dl_per_entry / avgdl)))).astype(np.float32)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Build scipy sparse matrices
    print("  Building sparse matrices...")
    base_mat = sparse.csr_matrix((norm_data, b_indices, b_indptr), shape=(b_rows, n_cols))
    query_mat = sparse.csr_matrix((new_data_arr, new_indices, new_indptr), shape=(q_rows, n_cols))

    # Compute scores: query @ base.T
    print("  Computing query @ base.T ...")
    t0 = time.time()
    scores = query_mat @ base_mat.T  # (nq, n_docs)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Extract top-k
    print(f"  Extracting top-{args.gt_k}...")
    t0 = time.time()
    gt_k = min(args.gt_k, b_rows)
    gt_ids = np.zeros((q_rows, gt_k), dtype=np.int32)
    for i in range(q_rows):
        row = scores.getrow(i).toarray().ravel()
        top_idx = np.argpartition(row, -gt_k)[-gt_k:]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        gt_ids[i] = top_idx[:gt_k]
    print(f"  Done in {time.time()-t0:.1f}s")

    save_gt(data_dir / "base_small.dev.bm25.gt", gt_ids, gt_k)

    print("\nDone! Query CSR now has IDF weights, GT regenerated.")
    print(f"  avgdl: {avgdl:.2f} (pass --bm25-avgdl to benchmark)")


if __name__ == "__main__":
    main()
