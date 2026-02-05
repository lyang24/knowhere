#!/usr/bin/env python3
"""Generate BM25 ground truth using brute force search."""

import numpy as np
import struct
import argparse
from pathlib import Path


def load_csr(path):
    """Load CSR format file."""
    with open(path, 'rb') as f:
        n_rows = struct.unpack('q', f.read(8))[0]
        n_cols = struct.unpack('q', f.read(8))[0]
        nnz = struct.unpack('q', f.read(8))[0]

        indptr = np.frombuffer(f.read((n_rows + 1) * 8), dtype=np.int64)
        indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
        data = np.frombuffer(f.read(nnz * 4), dtype=np.float32)

    return n_rows, n_cols, nnz, indptr, indices, data


def compute_bm25_score(query_indices, query_values, doc_indices, doc_values, doc_len, avgdl, k1=1.2, b=0.75):
    """Compute BM25 score between query and document."""
    score = 0.0

    # Create a dict for fast lookup of document term frequencies
    doc_tf = dict(zip(doc_indices, doc_values))

    for q_idx, q_val in zip(query_indices, query_values):
        if q_idx in doc_tf:
            tf = doc_tf[q_idx]
            # BM25 formula: tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            term_score = tf * (k1 + 1) / denom
            score += q_val * term_score

    return score


def generate_ground_truth(base_path, query_path, output_path, k=10, k1=1.2, b=0.75, avgdl=100.0, nq=None):
    """Generate ground truth using brute force BM25."""
    print(f"Loading base vectors from {base_path}...")
    n_base, n_cols, base_nnz, base_indptr, base_indices, base_data = load_csr(base_path)
    print(f"  {n_base} docs, {n_cols} dims, {base_nnz} nnz")

    print(f"Loading queries from {query_path}...")
    n_queries, _, query_nnz, query_indptr, query_indices, query_data = load_csr(query_path)
    print(f"  {n_queries} queries, {query_nnz} nnz")

    if nq is not None and nq < n_queries:
        n_queries = nq
        print(f"  Using {n_queries} queries")

    # Compute document lengths (sum of term frequencies)
    print("Computing document lengths...")
    doc_lens = np.zeros(n_base, dtype=np.float32)
    for i in range(n_base):
        start, end = base_indptr[i], base_indptr[i+1]
        doc_lens[i] = np.sum(base_data[start:end])

    # Use provided avgdl or compute from data
    actual_avgdl = np.mean(doc_lens)
    print(f"  Avg doc length: {actual_avgdl:.2f} (using {avgdl} for BM25)")

    print(f"Computing BM25 scores for {n_queries} queries (brute force)...")
    ground_truth = np.full((n_queries, k), -1, dtype=np.int32)  # Initialize with -1 for padding

    for q in range(n_queries):
        if (q + 1) % 100 == 0:
            print(f"  Query {q+1}/{n_queries}")

        q_start, q_end = query_indptr[q], query_indptr[q+1]
        q_indices = query_indices[q_start:q_end]
        q_values = query_data[q_start:q_end]

        scores = np.zeros(n_base, dtype=np.float32)

        for d in range(n_base):
            d_start, d_end = base_indptr[d], base_indptr[d+1]
            d_indices = base_indices[d_start:d_end]
            d_values = base_data[d_start:d_end]

            scores[d] = compute_bm25_score(
                q_indices, q_values, d_indices, d_values,
                doc_lens[d], avgdl, k1, b
            )

        # Get top-k indices, excluding zero-score docs
        # Knowhere MaxScore only considers docs with query-term overlap (score > 0)
        # If fewer than k matches, pad with -1
        nonzero_mask = scores > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_scores = scores[nonzero_mask]

        if len(nonzero_indices) == 0:
            # No matches - fill with -1
            ground_truth[q] = -1
        else:
            # Sort by (score desc, doc_id asc) for deterministic tie-breaking
            sort_order = np.lexsort((nonzero_indices, -nonzero_scores))
            sorted_indices = nonzero_indices[sort_order]

            if len(sorted_indices) >= k:
                ground_truth[q] = sorted_indices[:k]
            else:
                # Fewer than k matches - pad with -1
                ground_truth[q, :len(sorted_indices)] = sorted_indices
                ground_truth[q, len(sorted_indices):] = -1

    print(f"Saving ground truth to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(struct.pack('i', n_queries))  # nq (int32)
        f.write(struct.pack('i', k))          # k (int32)
        f.write(ground_truth.tobytes())

    print("Done!")
    return ground_truth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate BM25 ground truth')
    parser.add_argument('--data-dir', required=True, help='Directory with CSR files')
    parser.add_argument('--k', type=int, default=10, help='Top-k')
    parser.add_argument('--nq', type=int, default=None, help='Number of queries (default: all)')
    parser.add_argument('--k1', type=float, default=1.2, help='BM25 k1 parameter')
    parser.add_argument('--b', type=float, default=0.75, help='BM25 b parameter')
    parser.add_argument('--avgdl', type=float, default=100.0, help='BM25 avgdl parameter')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    generate_ground_truth(
        data_dir / 'base_small.csr',
        data_dir / 'queries.dev.csr',
        data_dir / 'base_small.dev.gt',
        k=args.k,
        nq=args.nq,
        k1=args.k1,
        b=args.b,
        avgdl=args.avgdl
    )
