#!/usr/bin/env python3
"""
benchmark_hybrid.py

Evaluates and compares retrieval models:
1. Dense Only (BGE-M3)
2. BM25 Only (Lexical)
3. Hybrid (Dense + BM25 combined via Min-Max Normalization)

Performs an alpha sweep from 0.0 (BM25 only) to 1.0 (Dense only) in steps of 0.1,
and reports metrics (Recall@1, Recall@5, Recall@10, MRR) overall and per target language.
"""

import os
import sys
import json
import re
import argparse
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Set
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.exists(path):
        print(f"Error: Chunk file not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))
    return chunks


def load_queries(path: str) -> List[Dict[str, Any]]:
    queries = []
    if not os.path.exists(path):
        print(f"Error: Queries file not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            queries.append(json.loads(line))
    return queries


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text by extracting alphanumeric word runs.
    Works for English and Indic languages (Devanagari, Bengali, etc.).
    """
    return re.findall(r'\w+', text.lower())


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalizes scores to [0, 1] range. Handles division by zero safely.
    """
    s_min = scores.min()
    s_max = scores.max()
    denominator = s_max - s_min
    if denominator < 1e-9:
        return np.zeros_like(scores)
    return (scores - s_min) / denominator


def calculate_mrr(retrieved_documents: List[str], relevant_documents: Set[str]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) based on the 1-indexed rank of the first relevant document.
    """
    if not relevant_documents:
        return 0.0
    for rank_idx, doc_id in enumerate(retrieved_documents):
        if doc_id in relevant_documents:
            return 1.0 / (rank_idx + 1)
    return 0.0


def calculate_recall(retrieved_documents: List[str], relevant_documents: Set[str], k: int) -> float:
    """
    Computes whether any of the top k retrieved unique documents are in relevant documents.
    """
    if not relevant_documents:
        return 0.0
    return float(bool(set(retrieved_documents[:k]) & relevant_documents))


def get_ranked_documents(scores: np.ndarray, chunks: List[Dict[str, Any]], max_depth: int = 100) -> List[str]:
    """
    Sorts corpus chunks by score, extracts their parent document_ids, and deduplicates
    them while preserving order to yield the top unique document list.
    """
    top_indices = np.argsort(scores)[::-1]
    retrieved_documents = []
    seen = set()
    for index in top_indices:
        doc_id = chunks[index]["document_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            retrieved_documents.append(doc_id)
            if len(retrieved_documents) >= max_depth:
                break
    return retrieved_documents


def main():
    parser = argparse.ArgumentParser(description="VaaniRAG Hybrid Retrieval Benchmarking with Alpha Sweep.")
    parser.add_argument("--strategy", type=str, default="adaptive", help="Chunking strategy to load (passage, adaptive, overlap, sentence, semantic)")
    args = parser.parse_args()

    chunks_path = f"data/chunks_{args.strategy}.jsonl"
    queries_path = "data/evaluation_queries.jsonl"

    print("=================================================================")
    print("      VaaniRAG Phase 2: Hybrid Retrieval & Alpha Sweep")
    print("=================================================================")
    print(f"Strategy: {args.strategy}")

    chunks = load_chunks(chunks_path)
    queries = load_queries(queries_path)
    print(f"Loaded {len(chunks)} passage chunks.")
    print(f"Loaded {len(queries)} evaluation queries.")

    # Get unique languages present in queries
    languages = sorted(list(set(q["language"] for q in queries)))
    print(f"Detected query languages: {languages}")

    # 1. Initialize BM25 lexical indexer
    print("Initializing BM25 Lexical Indexer...")
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # 2. Initialize BGE-M3 dense model & encode
    print("Loading BGE-M3 model & encoding passages...")
    model = SentenceTransformer("BAAI/bge-m3")
    
    passage_texts = [c["text"] for c in chunks]
    passage_embeddings = model.encode(
        passage_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Encoding evaluation queries...")
    query_texts = [q["query"] for q in queries]
    query_embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # 3. Precompute dense and bm25 raw scores for all queries to make the alpha sweep instant
    print("Pre-calculating raw dense and BM25 scores...")
    num_queries = len(queries)
    num_chunks = len(chunks)
    
    raw_dense_scores = []
    raw_bm25_scores = []
    
    for idx, query in enumerate(queries):
        # Dense similarity
        d_scores = passage_embeddings @ query_embeddings[idx]
        raw_dense_scores.append(min_max_normalize(d_scores))
        
        # BM25 similarity
        query_tokens = tokenize(query["query"])
        b_scores = np.array(bm25.get_scores(query_tokens))
        raw_bm25_scores.append(min_max_normalize(b_scores))

    # 4. Perform Alpha Sweep (0.0 to 1.0)
    sweep_results = []
    lang_sweep_results = {lang: [] for lang in languages}
    
    alphas = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]

    print("\nRunning Alpha Sweep...")
    for alpha in alphas:
        alpha_label = f"{alpha:.1f}"
        if alpha == 0.0:
            label = "0.0 (BM25 only)"
        elif alpha == 1.0:
            label = "1.0 (Dense only)"
        else:
            label = alpha_label
            
        # Overall tracking
        recall_counts = {1: 0.0, 5: 0.0, 10: 0.0}
        total_mrr = 0.0
        
        # Language breakdown tracking
        lang_recall_counts = {lang: {1: 0.0, 5: 0.0, 10: 0.0} for lang in languages}
        lang_total_mrr = {lang: 0.0 for lang in languages}
        lang_counts = {lang: 0 for lang in languages}

        for idx, query in enumerate(queries):
            lang = query["language"]
            relevant_docs = set(query["relevant_document_ids"])
            lang_counts[lang] += 1
            
            # Combine pre-normalized scores
            combined_scores = alpha * raw_dense_scores[idx] + (1.0 - alpha) * raw_bm25_scores[idx]
            
            # Retrieve unique document ranking list
            retrieved_docs = get_ranked_documents(combined_scores, chunks, max_depth=20)
            
            # Compute metrics overall and for specific language
            for k in [1, 5, 10]:
                rec_val = calculate_recall(retrieved_docs, relevant_docs, k)
                recall_counts[k] += rec_val
                lang_recall_counts[lang][k] += rec_val
                
            mrr_val = calculate_mrr(retrieved_docs, relevant_docs)
            total_mrr += mrr_val
            lang_total_mrr[lang] += mrr_val

        # Add overall results
        sweep_results.append({
            "Alpha": label,
            "R@1": round(recall_counts[1] / num_queries, 4),
            "R@5": round(recall_counts[5] / num_queries, 4),
            "R@10": round(recall_counts[10] / num_queries, 4),
            "MRR": round(total_mrr / num_queries, 4)
        })

        # Add language-specific results
        for lang in languages:
            count = lang_counts[lang]
            if count > 0:
                lang_sweep_results[lang].append({
                    "Alpha": label,
                    "R@1": round(lang_recall_counts[lang][1] / count, 4),
                    "R@5": round(lang_recall_counts[lang][5] / count, 4),
                    "R@10": round(lang_recall_counts[lang][10] / count, 4),
                    "MRR": round(lang_total_mrr[lang] / count, 4)
                })

    # 5. Print sweep reports
    print("\n==================================================================")
    print(f"        OVERALL Hybrid Retrieval Sweep Report ({args.strategy.upper()})")
    print("==================================================================")
    df_report = pd.DataFrame(sweep_results)
    print(df_report.to_string(index=False))

    # Print breakdown per language
    for lang in languages:
        print(f"\n==================================================================")
        print(f"        Language: {lang.upper()} Sweep Report ({args.strategy.upper()})")
        print("==================================================================")
        df_lang_report = pd.DataFrame(lang_sweep_results[lang])
        print(df_lang_report.to_string(index=False))


if __name__ == "__main__":
    main()
