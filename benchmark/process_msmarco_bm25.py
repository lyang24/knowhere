#!/usr/bin/env python3
"""Process MSMARCO collection into BM25 sparse vectors (CSR format)."""

import argparse
import struct
import re
from collections import Counter
from pathlib import Path

# Common English stop words
STOP_WORDS = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'can', 'just', 'should', 'now', 'i',
    'you', 'your', 'we', 'our', 'their', 'them', 'his', 'her', 'she', 'him',
    'my', 'me', 'do', 'does', 'did', 'doing', 'would', 'could', 'might', 'must',
    'shall', 'if', 'or', 'because', 'until', 'while', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'any', 'been', 'being', 'am',
])

def tokenize(text):
    """Simple tokenization: lowercase, split on non-alphanumeric, remove stop words."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens

def process_collection(collection_path, output_dir, max_docs=None):
    """Process collection.tsv into CSR format."""
    print(f"Processing {collection_path}...")

    # First pass: build vocabulary and count docs
    vocab = {}
    doc_count = 0

    print("Pass 1: Building vocabulary...")
    with open(collection_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue
            doc_id, text = parts
            tokens = tokenize(text)
            for token in set(tokens):  # unique tokens per doc
                if token not in vocab:
                    vocab[token] = len(vocab)
            doc_count += 1
            if doc_count % 100000 == 0:
                print(f"  Processed {doc_count} docs, vocab size: {len(vocab)}")
            if max_docs and doc_count >= max_docs:
                break

    print(f"  Total: {doc_count} docs, {len(vocab)} unique terms")

    # Second pass: build CSR matrix
    print("Pass 2: Building CSR matrix...")
    indptr = [0]
    indices = []
    data = []

    with open(collection_path, 'r', encoding='utf-8', errors='ignore') as f:
        processed = 0
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue
            doc_id, text = parts
            tokens = tokenize(text)
            tf = Counter(tokens)

            # Sort by term index for CSR format
            sorted_terms = sorted([(vocab[t], c) for t, c in tf.items() if t in vocab])

            for term_idx, count in sorted_terms:
                indices.append(term_idx)
                data.append(float(count))

            indptr.append(len(indices))
            processed += 1

            if processed % 100000 == 0:
                print(f"  Processed {processed} docs")
            if max_docs and processed >= max_docs:
                break

    n_rows = len(indptr) - 1
    n_cols = len(vocab)
    nnz = len(indices)

    print(f"  CSR matrix: {n_rows} x {n_cols}, {nnz} nnz")

    # Save CSR format
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csr_path = output_path / 'base_small.csr'
    print(f"Saving to {csr_path}...")

    with open(csr_path, 'wb') as f:
        f.write(struct.pack('q', n_rows))
        f.write(struct.pack('q', n_cols))
        f.write(struct.pack('q', nnz))
        for ptr in indptr:
            f.write(struct.pack('q', ptr))
        for idx in indices:
            f.write(struct.pack('i', idx))
        for val in data:
            f.write(struct.pack('f', val))

    # Save vocabulary
    vocab_path = output_path / 'vocabulary.tsv'
    print(f"Saving vocabulary to {vocab_path}...")
    with open(vocab_path, 'w') as f:
        for term, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{term}\n")

    return vocab, n_rows

def process_queries(queries_path, output_path, vocab):
    """Process queries into CSR format using existing vocabulary."""
    print(f"Processing queries from {queries_path}...")

    indptr = [0]
    indices = []
    data = []
    query_count = 0

    with open(queries_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue
            qid, text = parts
            tokens = tokenize(text)
            tf = Counter(tokens)

            # Only use terms in vocabulary
            sorted_terms = sorted([(vocab[t], c) for t, c in tf.items() if t in vocab])

            for term_idx, count in sorted_terms:
                indices.append(term_idx)
                data.append(float(count))

            indptr.append(len(indices))
            query_count += 1

    n_rows = len(indptr) - 1
    n_cols = len(vocab)
    nnz = len(indices)

    print(f"  Queries: {n_rows}, nnz: {nnz}, avg terms: {nnz/n_rows:.1f}")

    with open(output_path, 'wb') as f:
        f.write(struct.pack('q', n_rows))
        f.write(struct.pack('q', n_cols))
        f.write(struct.pack('q', nnz))
        for ptr in indptr:
            f.write(struct.pack('q', ptr))
        for idx in indices:
            f.write(struct.pack('i', idx))
        for val in data:
            f.write(struct.pack('f', val))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MSMARCO to BM25 sparse vectors')
    parser.add_argument('--input-dir', required=True, help='Directory with collection.tsv and queries')
    parser.add_argument('--output-dir', required=True, help='Output directory for CSR files')
    parser.add_argument('--max-docs', type=int, default=None, help='Max documents to process')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Process collection
    vocab, n_docs = process_collection(
        input_dir / 'collection.tsv',
        output_dir,
        max_docs=args.max_docs
    )

    # Process queries
    queries_file = input_dir / 'queries.dev.tsv'
    if queries_file.exists():
        process_queries(queries_file, output_dir / 'queries.dev.csr', vocab)

    print("\nDone!")
    print(f"Output files in {output_dir}:")
    print(f"  base_small.csr - {n_docs} documents")
    print(f"  queries.dev.csr - queries")
    print(f"  vocabulary.tsv - {len(vocab)} terms")
