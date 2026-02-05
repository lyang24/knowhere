#!/usr/bin/env python3
"""
Prepare BM25 sparse datasets from MSMARCO passages.

This script:
1. Downloads MSMARCO passage collection
2. Tokenizes text and removes stop words
3. Builds vocabulary (term -> ID mapping)
4. Creates sparse vectors with term frequencies
5. Saves in CSR format compatible with benchmark_sparse_algo

Usage:
    python prepare_bm25_data.py --output-dir /path/to/output --num-docs 100000
"""

import argparse
import gzip
import os
import re
import struct
import urllib.request
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np

# MSMARCO passage collection URL (includes both collection and queries)
MSMARCO_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
MSMARCO_URL_MIRROR = "https://www.dropbox.com/s/9f54jg2f71ray3b/collectionandqueries.tar.gz?dl=1"

# Common English stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'can', 'could', 'may', 'might',
    'must', 'shall', 'should', 'would', 'now', 'or', 'if', 'then', 'also', 'into',
    'about', 'after', 'before', 'between', 'through', 'during', 'above', 'below',
    'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'once', 'here',
    'there', 'any', 'been', 'being', 'did', 'does', 'doing', 'having', 'her', 'him',
    'his', 'herself', 'himself', 'itself', 'me', 'my', 'myself', 'our', 'ours',
    'ourselves', 'she', 'their', 'theirs', 'them', 'themselves', 'these', 'those',
    'us', 'we', 'you', 'your', 'yours', 'yourself', 'yourselves', 'i', 'am', 'do'
}


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split on non-alphanumeric, filter stop words."""
    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    # Filter stop words and very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def download_file(url: str, output_path: str):
    """Download a file with progress indication."""
    print(f"Downloading {url}...")

    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        print(f"\r  Progress: {percent}%", end='', flush=True)

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print()


def ensure_msmarco_data(data_dir: str):
    """Download and extract MSMARCO data if not present."""
    collection_path = os.path.join(data_dir, "collection.tsv")
    queries_path = os.path.join(data_dir, "queries.dev.tsv")
    tar_path = os.path.join(data_dir, "collectionandqueries.tar.gz")

    if os.path.exists(collection_path) and os.path.exists(queries_path):
        return  # Already extracted

    if not os.path.exists(tar_path):
        try:
            download_file(MSMARCO_URL, tar_path)
        except Exception as e:
            print(f"  Primary URL failed: {e}")
            print("  Trying mirror...")
            download_file(MSMARCO_URL_MIRROR, tar_path)

    print("Extracting collectionandqueries.tar.gz...")
    import tarfile
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)


def load_msmarco_passages(data_dir: str, max_docs: int = None) -> List[str]:
    """Load MSMARCO passages from collection.tsv."""
    ensure_msmarco_data(data_dir)
    collection_path = os.path.join(data_dir, "collection.tsv")

    # Load passages
    print(f"Loading passages from {collection_path}...")
    passages = []
    with open(collection_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                passages.append(parts[1])  # passage text is second column
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1} passages...")

    print(f"  Total passages loaded: {len(passages)}")
    return passages


def load_msmarco_queries(data_dir: str, max_queries: int = None) -> List[str]:
    """Load MSMARCO queries."""
    ensure_msmarco_data(data_dir)
    queries_path = os.path.join(data_dir, "queries.dev.tsv")

    if not os.path.exists(queries_path):
        # Try queries.dev.small.tsv as fallback
        queries_path = os.path.join(data_dir, "queries.dev.small.tsv")

    print(f"Loading queries from {queries_path}...")
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_queries and i >= max_queries:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries.append(parts[1])  # query text is second column

    print(f"  Total queries loaded: {len(queries)}")
    return queries


def build_vocabulary(documents: List[str], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from documents, filtering rare terms."""
    print("Building vocabulary...")
    term_freq = Counter()

    for i, doc in enumerate(documents):
        tokens = tokenize(doc)
        term_freq.update(set(tokens))  # Count document frequency
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1} documents...")

    # Filter by minimum frequency and assign IDs
    vocab = {}
    for term, freq in term_freq.items():
        if freq >= min_freq:
            vocab[term] = len(vocab)

    print(f"  Vocabulary size: {len(vocab)} (filtered {len(term_freq) - len(vocab)} rare terms)")
    return vocab


def documents_to_sparse(documents: List[str], vocab: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert documents to sparse CSR format with term frequencies."""
    print("Converting documents to sparse vectors...")

    indptr = [0]
    indices = []
    data = []

    for i, doc in enumerate(documents):
        tokens = tokenize(doc)
        term_counts = Counter(tokens)

        # Sort by term ID for CSR format
        doc_terms = [(vocab[t], c) for t, c in term_counts.items() if t in vocab]
        doc_terms.sort(key=lambda x: x[0])

        for term_id, count in doc_terms:
            indices.append(term_id)
            data.append(float(count))  # Term frequency as float

        indptr.append(len(indices))

        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1} documents...")

    print(f"  Total non-zeros: {len(data)}")
    return np.array(indptr, dtype=np.int64), np.array(indices, dtype=np.int32), np.array(data, dtype=np.float32)


def save_csr(filepath: str, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n_cols: int):
    """Save sparse matrix in CSR binary format compatible with benchmark."""
    n_rows = len(indptr) - 1
    nnz = len(data)

    print(f"Saving to {filepath}...")
    print(f"  Shape: {n_rows} x {n_cols}, NNZ: {nnz}")

    with open(filepath, 'wb') as f:
        # Header: n_rows, n_cols, nnz (all int64)
        f.write(struct.pack('<qqq', n_rows, n_cols, nnz))
        # indptr: (n_rows + 1) int64 values
        f.write(indptr.astype(np.int64).tobytes())
        # indices: nnz int32 values
        f.write(indices.astype(np.int32).tobytes())
        # data: nnz float32 values
        f.write(data.astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(description='Prepare BM25 sparse datasets from MSMARCO')
    parser.add_argument('--output-dir', type=str, default='/tmp/bm25_data',
                        help='Output directory for processed data')
    parser.add_argument('--num-docs', type=int, default=100000,
                        help='Number of documents to process (default: 100000)')
    parser.add_argument('--num-queries', type=int, default=10000,
                        help='Number of queries to process (default: 10000)')
    parser.add_argument('--min-freq', type=int, default=2,
                        help='Minimum document frequency for vocabulary (default: 2)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    passages = load_msmarco_passages(args.output_dir, args.num_docs)
    queries = load_msmarco_queries(args.output_dir, args.num_queries)

    # Build vocabulary from passages
    vocab = build_vocabulary(passages, min_freq=args.min_freq)
    n_cols = len(vocab)

    # Save vocabulary for reference
    vocab_path = os.path.join(args.output_dir, "vocabulary.tsv")
    print(f"Saving vocabulary to {vocab_path}...")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for term, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{term}\n")

    # Convert passages to sparse
    p_indptr, p_indices, p_data = documents_to_sparse(passages, vocab)

    # Determine output filename based on size
    if args.num_docs >= 1000000:
        base_filename = "base_1M_bm25.csr"
    elif args.num_docs >= 100000:
        base_filename = "base_100k_bm25.csr"
    else:
        base_filename = f"base_{args.num_docs}_bm25.csr"

    save_csr(
        os.path.join(args.output_dir, base_filename),
        p_indptr, p_indices, p_data, n_cols
    )

    # Convert queries to sparse
    q_indptr, q_indices, q_data = documents_to_sparse(queries, vocab)
    save_csr(
        os.path.join(args.output_dir, "queries_bm25.csr"),
        q_indptr, q_indices, q_data, n_cols
    )

    print("\nDone!")
    print(f"Output files in {args.output_dir}:")
    print(f"  - {base_filename}: {len(passages)} passages")
    print(f"  - queries_bm25.csr: {len(queries)} queries")
    print(f"  - vocabulary.tsv: {n_cols} terms")


if __name__ == '__main__':
    main()
