#!/usr/bin/env python3
"""Prepare HotpotQA data for BM25 benchmark.

Converts HotpotQA JSON to CSR format suitable for sparse retrieval benchmarking.
Each paragraph becomes a document, questions become queries.
Ground truth is based on supporting_facts annotations.
"""

import json
import numpy as np
import struct
import argparse
import re
from pathlib import Path
from collections import defaultdict


def tokenize(text):
    """Simple word tokenization for BM25."""
    # Lowercase and split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return tokens


def compute_term_frequencies(tokens):
    """Compute term frequencies for a document."""
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    return tf


def save_csr(path, n_rows, n_cols, indptr, indices, data):
    """Save CSR format file."""
    nnz = len(indices)
    with open(path, 'wb') as f:
        f.write(struct.pack('q', n_rows))
        f.write(struct.pack('q', n_cols))
        f.write(struct.pack('q', nnz))
        f.write(np.array(indptr, dtype=np.int64).tobytes())
        f.write(np.array(indices, dtype=np.int32).tobytes())
        f.write(np.array(data, dtype=np.float32).tobytes())
    print(f"  Saved {path}: {n_rows} rows, {n_cols} cols, {nnz} nnz")


def save_ground_truth(path, gt, nq, k):
    """Save ground truth file."""
    with open(path, 'wb') as f:
        f.write(struct.pack('i', nq))
        f.write(struct.pack('i', k))
        f.write(np.array(gt, dtype=np.int32).tobytes())
    print(f"  Saved {path}: {nq} queries, k={k}")


def prepare_hotpotqa(json_path, output_dir, max_docs=None, max_queries=None, k=10):
    """Prepare HotpotQA data for BM25 benchmark."""
    print(f"Loading HotpotQA from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} QA pairs")

    # Step 1: Extract all unique paragraphs as documents
    # Each paragraph is identified by (title, paragraph_index)
    # For fullwiki, each question has 10 paragraphs in context
    print("Extracting documents (paragraphs)...")
    doc_id_map = {}  # (title) -> doc_id (we use title as unique key since each title appears once per context)
    documents = []  # List of (title, text)

    # First pass: collect all unique paragraphs across all questions
    all_paragraphs = {}  # title -> full text (concatenated sentences)
    for item in data:
        for para in item['context']:
            title = para[0]
            sentences = para[1]
            full_text = ' '.join(sentences)
            # Keep the longest version if title appears multiple times
            if title not in all_paragraphs or len(full_text) > len(all_paragraphs[title]):
                all_paragraphs[title] = full_text

    # Build document list
    for title, text in all_paragraphs.items():
        doc_id = len(documents)
        doc_id_map[title] = doc_id
        documents.append((title, text))

    if max_docs and len(documents) > max_docs:
        print(f"  Limiting to {max_docs} documents")
        documents = documents[:max_docs]
        # Update doc_id_map
        doc_id_map = {title: i for i, (title, _) in enumerate(documents)}

    print(f"  Found {len(documents)} unique paragraphs")

    # Step 2: Build vocabulary from documents
    print("Building vocabulary...")
    vocab = {}  # token -> term_id
    doc_term_freqs = []  # List of {term_id: freq} for each doc

    for doc_id, (title, text) in enumerate(documents):
        tokens = tokenize(title + ' ' + text)
        tf = compute_term_frequencies(tokens)

        term_freq_dict = {}
        for token, freq in tf.items():
            if token not in vocab:
                vocab[token] = len(vocab)
            term_freq_dict[vocab[token]] = freq
        doc_term_freqs.append(term_freq_dict)

        if (doc_id + 1) % 10000 == 0:
            print(f"    Processed {doc_id + 1}/{len(documents)} docs")

    print(f"  Vocabulary size: {len(vocab)}")

    # Step 3: Extract queries and ground truth
    print("Extracting queries and ground truth...")
    queries = []  # List of question texts
    query_term_freqs = []  # List of {term_id: freq} for each query
    ground_truth = []  # List of relevant doc_ids for each query

    for item in data:
        question = item['question']
        supporting_facts = item.get('supporting_facts', [])

        # Get relevant doc IDs from supporting facts
        relevant_titles = set(sf[0] for sf in supporting_facts)
        relevant_doc_ids = []
        for title in relevant_titles:
            if title in doc_id_map:
                relevant_doc_ids.append(doc_id_map[title])

        # Skip queries with no relevant docs in our document set
        if not relevant_doc_ids:
            continue

        # Tokenize query
        tokens = tokenize(question)
        tf = compute_term_frequencies(tokens)

        term_freq_dict = {}
        for token, freq in tf.items():
            if token in vocab:  # Only use known terms
                term_freq_dict[vocab[token]] = freq

        # Skip queries with no terms in vocabulary
        if not term_freq_dict:
            continue

        queries.append(question)
        query_term_freqs.append(term_freq_dict)
        ground_truth.append(relevant_doc_ids)

        if max_queries and len(queries) >= max_queries:
            break

    print(f"  Extracted {len(queries)} valid queries")

    # Step 4: Convert to CSR format
    print("Converting to CSR format...")
    n_cols = len(vocab)

    # Documents CSR
    doc_indptr = [0]
    doc_indices = []
    doc_data = []
    for tf_dict in doc_term_freqs:
        for term_id, freq in sorted(tf_dict.items()):
            doc_indices.append(term_id)
            doc_data.append(float(freq))
        doc_indptr.append(len(doc_indices))

    # Queries CSR
    query_indptr = [0]
    query_indices = []
    query_data = []
    for tf_dict in query_term_freqs:
        for term_id, freq in sorted(tf_dict.items()):
            query_indices.append(term_id)
            query_data.append(float(freq))
        query_indptr.append(len(query_indices))

    # Step 5: Save files
    print("Saving files...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csr(
        output_dir / 'base_small.csr',
        len(documents), n_cols,
        doc_indptr, doc_indices, doc_data
    )

    save_csr(
        output_dir / 'queries.dev.csr',
        len(queries), n_cols,
        query_indptr, query_indices, query_data
    )

    # Save ground truth (for recall verification during development)
    # Note: This is based on supporting_facts, not BM25 scores
    # We'll generate proper BM25 GT separately using generate_bm25_gt.py
    gt_array = np.full((len(queries), k), -1, dtype=np.int32)
    for q_idx, rel_docs in enumerate(ground_truth):
        for j, doc_id in enumerate(rel_docs[:k]):
            gt_array[q_idx, j] = doc_id

    save_ground_truth(
        output_dir / 'supporting_facts.gt',
        gt_array.flatten(), len(queries), k
    )

    # Save metadata
    with open(output_dir / 'metadata.txt', 'w') as f:
        f.write(f"Source: HotpotQA fullwiki dev set\n")
        f.write(f"Documents: {len(documents)}\n")
        f.write(f"Queries: {len(queries)}\n")
        f.write(f"Vocabulary: {len(vocab)}\n")
        f.write(f"Avg doc length: {np.mean([sum(tf.values()) for tf in doc_term_freqs]):.2f}\n")
        f.write(f"Avg query length: {np.mean([sum(tf.values()) for tf in query_term_freqs]):.2f}\n")

    print("\nDataset statistics:")
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Vocabulary: {len(vocab)}")
    print(f"  Avg doc nnz: {np.mean([len(tf) for tf in doc_term_freqs]):.2f}")
    print(f"  Avg query nnz: {np.mean([len(tf) for tf in query_term_freqs]):.2f}")
    print(f"  Avg relevant docs per query: {np.mean([len(r) for r in ground_truth]):.2f}")

    print("\nDone! Now run generate_bm25_gt.py to create BM25-based ground truth.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare HotpotQA for BM25 benchmark')
    parser.add_argument('--input', required=True, help='Path to HotpotQA JSON file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--max-docs', type=int, default=None, help='Max documents to use')
    parser.add_argument('--max-queries', type=int, default=None, help='Max queries to use')
    parser.add_argument('--k', type=int, default=10, help='Top-k for ground truth')
    args = parser.parse_args()

    prepare_hotpotqa(args.input, args.output, args.max_docs, args.max_queries, args.k)
