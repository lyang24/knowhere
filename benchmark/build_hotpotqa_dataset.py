#!/usr/bin/env python3
"""Convert HotpotQA Wikipedia abstracts + dev questions to BM25 CSR format.

Usage:
    python3 build_hotpotqa_dataset.py \
        --abstracts-dir ~/data/hotpotqa_raw/enwiki-20171001-pages-meta-current-withlinks-abstracts \
        --questions ~/data/hotpotqa_raw/hotpot_dev_fullwiki_v1.json \
        --output-dir ~/data/hotpotqa_full_bm25
"""

import argparse
import bz2
import json
import os
import struct
import time
import numpy as np
from pathlib import Path


def load_abstracts(abstracts_dir):
    """Load all Wikipedia abstracts from HotpotQA bz2 files.
    Each doc = title + first paragraph sentences joined."""
    docs = []
    abstracts_path = Path(abstracts_dir)
    subdirs = sorted([d for d in abstracts_path.iterdir() if d.is_dir()])
    for subdir in subdirs:
        bz2_files = sorted(subdir.glob("wiki_*.bz2"))
        for bz2_file in bz2_files:
            with bz2.open(bz2_file, "rt") as f:
                for line in f:
                    doc = json.loads(line.strip())
                    title = doc.get("title", "")
                    text_paragraphs = doc.get("text", [])
                    # Join all paragraph sentences into one text
                    sentences = []
                    for para in text_paragraphs:
                        sentences.extend(para)
                    full_text = title + " " + " ".join(sentences)
                    docs.append(full_text)
        print(f"  Loaded {subdir.name}: {len(docs)} docs total")
    return docs


def load_questions(questions_path):
    """Load HotpotQA questions from JSON."""
    with open(questions_path) as f:
        data = json.load(f)
    return [item["question"] for item in data]


def save_csr(path, n_rows, n_cols, indptr, indices, data):
    """Save in CSR binary format."""
    nnz = len(indices)
    with open(path, "wb") as f:
        f.write(struct.pack("qqq", n_rows, n_cols, nnz))
        f.write(np.array(indptr, dtype=np.int64).tobytes())
        f.write(np.array(indices, dtype=np.int32).tobytes())
        f.write(np.array(data, dtype=np.float32).tobytes())
    print(f"  Saved {path}: {n_rows} rows, {n_cols} cols, {nnz} nnz")


def token_ids_to_csr(id_lists, n_vocab, weights=None):
    """Convert list of token-ID lists to CSR format (term frequencies).

    Args:
        weights: optional per-term weights (e.g. IDF). If provided, each tf
                 value is multiplied by weights[term_id].
    """
    indptr = [0]
    indices = []
    data = []
    for ids in id_lists:
        tf = {}
        for tid in ids:
            tf[tid] = tf.get(tid, 0) + 1
        for tid in sorted(tf.keys()):
            indices.append(tid)
            val = float(tf[tid])
            if weights is not None:
                val *= weights[tid]
            data.append(val)
        indptr.append(len(indices))
    return len(id_lists), n_vocab, indptr, indices, data


def main():
    import bm25s

    parser = argparse.ArgumentParser(description="Build HotpotQA BM25 dataset")
    parser.add_argument("--abstracts-dir", required=True, help="Path to extracted abstracts directory")
    parser.add_argument("--questions", required=True, help="Path to HotpotQA dev JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    print("[Loading Wikipedia abstracts]")
    t0 = time.time()
    docs = load_abstracts(args.abstracts_dir)
    print(f"  Total: {len(docs)} documents in {time.time()-t0:.1f}s")

    # Load questions
    print("\n[Loading questions]")
    questions = load_questions(args.questions)
    print(f"  {len(questions)} questions")

    # Tokenize
    print("\n[Tokenizing with bm25s]")
    t0 = time.time()
    corpus_tokenized = bm25s.tokenize(docs, stopwords="en")
    query_tokenized = bm25s.tokenize(questions, stopwords="en")
    print(f"  Tokenized in {time.time()-t0:.1f}s")

    corpus_ids = corpus_tokenized.ids
    query_ids = query_tokenized.ids
    vocab = corpus_tokenized.vocab
    n_vocab = len(vocab)
    print(f"  Vocabulary size: {n_vocab}")

    # Remap query tokens to corpus vocabulary
    inv_query_vocab = {v: k for k, v in query_tokenized.vocab.items()}
    remapped_query_ids = []
    for qids in query_ids:
        remapped = []
        for qid in qids:
            token_str = inv_query_vocab.get(qid)
            if token_str and token_str in vocab:
                remapped.append(vocab[token_str])
        remapped_query_ids.append(remapped)
    query_ids = remapped_query_ids

    # Compute document frequencies and IDF from corpus
    print("\n[Computing IDF]")
    t0 = time.time()
    df = np.zeros(n_vocab, dtype=np.float64)
    for doc_ids in corpus_ids:
        for tid in set(doc_ids):
            df[tid] += 1
    N = len(corpus_ids)
    idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
    idf[df == 0] = 0.0
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Terms with df>0: {int(np.sum(df > 0))}")
    print(f"  IDF range: [{idf[idf>0].min():.3f}, {idf.max():.3f}]")

    # Convert to CSR
    print("\n[Building CSR vectors]")
    t0 = time.time()
    # Base: raw tf (knowhere applies BM25 TF normalization at query time)
    base_n, _, base_indptr, base_indices, base_data = token_ids_to_csr(corpus_ids, n_vocab)
    save_csr(output_dir / "base_small.csr", base_n, n_vocab, base_indptr, base_indices, base_data)

    # Queries: IDF * tf (proper BM25 query weighting)
    query_n, _, query_indptr, query_indices, query_data = token_ids_to_csr(query_ids, n_vocab, weights=idf)
    save_csr(output_dir / "queries.dev.csr", query_n, n_vocab, query_indptr, query_indices, query_data)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Compute avgdl
    doc_lens = []
    for i in range(base_n):
        s, e = base_indptr[i], base_indptr[i+1]
        doc_lens.append(sum(base_data[s:e]))
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0
    print(f"  avgdl: {avgdl:.2f}")

    # Generate ground truth with proper BM25 scoring
    print("\n[Generating ground truth]")
    from scipy import sparse as sp

    k1, b_param = 1.2, 0.75
    gt_k = 100

    # BM25 TF normalization on base
    print("  Normalizing base with BM25 TF formula...")
    t0 = time.time()
    dl = np.array(doc_lens, dtype=np.float64)
    norm_data = np.empty(len(base_data), dtype=np.float32)
    for i in range(base_n):
        s, e = base_indptr[i], base_indptr[i+1]
        tf_vals = np.array(base_data[s:e], dtype=np.float64)
        norm_data[s:e] = (tf_vals * (k1 + 1) / (tf_vals + k1 * (1 - b_param + b_param * (dl[i] / avgdl)))).astype(np.float32)
    print(f"  Done in {time.time()-t0:.1f}s")

    base_mat = sp.csr_matrix((norm_data, np.array(base_indices, dtype=np.int32),
                              np.array(base_indptr, dtype=np.int64)), shape=(base_n, n_vocab))
    query_mat = sp.csr_matrix((np.array(query_data, dtype=np.float32),
                               np.array(query_indices, dtype=np.int32),
                               np.array(query_indptr, dtype=np.int64)), shape=(query_n, n_vocab))

    print(f"  Computing query @ base.T ({query_n} x {base_n})...")
    t0 = time.time()
    scores = query_mat @ base_mat.T
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"  Extracting top-{gt_k}...")
    t0 = time.time()
    gt_ids = np.zeros((query_n, gt_k), dtype=np.int32)
    for i in range(query_n):
        row = scores.getrow(i).toarray().ravel()
        top_idx = np.argpartition(row, -gt_k)[-gt_k:]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        gt_ids[i] = top_idx[:gt_k]
    print(f"  Done in {time.time()-t0:.1f}s")

    gt_path = output_dir / "base_small.dev.bm25.gt"
    with open(gt_path, "wb") as f:
        f.write(struct.pack("ii", query_n, gt_k))
        f.write(gt_ids.tobytes())
    print(f"  Saved {gt_path}")

    # Save metadata
    meta_path = output_dir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"Source: HotpotQA Wikipedia abstracts\n")
        f.write(f"Documents: {base_n}\n")
        f.write(f"Queries: {query_n}\n")
        f.write(f"Vocabulary: {n_vocab}\n")
        f.write(f"Avg doc length: {avgdl:.2f}\n")
    print(f"  Saved metadata to {meta_path}")

    print(f"\nDone! Output in {output_dir}")


if __name__ == "__main__":
    main()
