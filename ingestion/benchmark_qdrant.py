#!/usr/bin/env python3
"""
benchmark_qdrant.py

Evaluates the dense retrieval performance of the Qdrant backend (local or cloud)
across all 400 queries, reporting Recall@1, Recall@5, Recall@10, and MRR.
Sweeps different values of k (5, 10, 20) to compare accuracy and network latency.
"""

import os
import sys
import argparse
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.dense_retriever import QdrantDenseRetriever
from retrieval.qdrant_client import QDRANT_COLLECTION


def load_queries(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Error: Queries file not found at {path}", file=sys.stderr)
        sys.exit(1)
    queries = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            queries.append(json.loads(line))
    return queries


def calculate_mrr(retrieved_documents: List[str], relevant_documents: Set[str]) -> float:
    if not relevant_documents:
        return 0.0
    for rank_idx, doc_id in enumerate(retrieved_documents):
        if doc_id in relevant_documents:
            return 1.0 / (rank_idx + 1)
    return 0.0


def calculate_recall(retrieved_documents: List[str], relevant_documents: Set[str], k: int) -> float:
    if not relevant_documents:
        return 0.0
    return float(bool(set(retrieved_documents[:k]) & relevant_documents))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qdrant retrieval performance.")
    parser.add_argument("--collection", type=str, default=QDRANT_COLLECTION, help="Qdrant collection to evaluate")
    parser.add_argument("--filter-language", action="store_true", help="Apply target language filter to vector search payload")
    args = parser.parse_args()

    queries_path = "data/evaluation_queries.jsonl"
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} evaluation queries.")

    print("\nInitializing QdrantDenseRetriever (will load BGE-M3 model)...")
    retriever = QdrantDenseRetriever(collection_name=args.collection)

    # 1. Precompute BGE-M3 embeddings for all 400 queries to speed up execution
    print("\nPre-computing query embeddings using BGE-M3...")
    start_embed_all = time.perf_counter()
    
    query_vectors = []
    embed_latencies = []
    
    for query in queries:
        start_emb = time.perf_counter()
        vector = retriever.embed_query(query["query"])
        end_emb = time.perf_counter()
        query_vectors.append(vector)
        embed_latencies.append((end_emb - start_emb) * 1000)
        
    end_embed_all = time.perf_counter()
    print(f"Pre-encoded {len(queries)} queries in {end_embed_all - start_embed_all:.2f} seconds.")

    # Show query embedding generation latency
    p50_emb = np.percentile(embed_latencies, 50)
    p70_emb = np.percentile(embed_latencies, 70)
    p95_emb = np.percentile(embed_latencies, 95)
    p100_emb = np.max(embed_latencies)
    print("\n========================================================")
    print("        Query Embedding Latency Statistics")
    print("========================================================")
    print(f"  P50 (Median): {p50_emb:.2f} ms")
    print(f"  P70:          {p70_emb:.2f} ms")
    print(f"  P95:          {p95_emb:.2f} ms")
    print(f"  P100 (Max):   {p100_emb:.2f} ms")

    # 2. Sweep different K values for Qdrant retrieval
    k_values = [5, 10, 20]
    sweep_results = []
    latency_results = []

    for k in k_values:
        print(f"\nEvaluating Qdrant Search Sweep with K = {k} (Language filtering: {args.filter_language})...")
        
        recall_counts = {1: 0.0, 5: 0.0, 10: 0.0}
        total_mrr = 0.0
        qdrant_latencies = []
        
        for idx, query in enumerate(queries):
            lang = query["language"]
            relevant_docs = set(query["relevant_document_ids"])
            query_vector = query_vectors[idx]

            # Measure Qdrant Search Latency
            lang_filter = lang if args.filter_language else None
            start_qdrant = time.perf_counter()
            hits = retriever.search_vector(query_vector, k=k, language_filter=lang_filter)
            end_qdrant = time.perf_counter()
            
            qdrant_ms = (end_qdrant - start_qdrant) * 1000
            qdrant_latencies.append(qdrant_ms)
            
            # Deduplicate retrieved parent document IDs
            retrieved_docs = []
            seen = set()
            for hit in hits:
                doc_id = hit["document_id"]
                if doc_id not in seen:
                    seen.add(doc_id)
                    retrieved_docs.append(doc_id)

            # Compute metrics (Recall@1, @5, @10 and MRR)
            for eval_k in [1, 5, 10]:
                recall_counts[eval_k] += calculate_recall(retrieved_docs, relevant_docs, eval_k)

            mrr_val = calculate_mrr(retrieved_docs, relevant_docs)
            total_mrr += mrr_val

        # Record overall accuracy metrics for this K
        num_queries = len(queries)
        sweep_results.append({
            "K Depth": k,
            "R@1": round(recall_counts[1] / num_queries, 4),
            "R@5": round(recall_counts[5] / num_queries, 4),
            "R@10": round(recall_counts[10] / num_queries, 4),
            "MRR": round(total_mrr / num_queries, 4)
        })

        # Calculate search latency percentiles for this K
        p50_qd = np.percentile(qdrant_latencies, 50)
        p70_qd = np.percentile(qdrant_latencies, 70)
        p95_qd = np.percentile(qdrant_latencies, 95)
        p100_qd = np.max(qdrant_latencies)
        
        latency_results.append({
            "K Depth": k,
            "P50 (ms)": round(p50_qd, 2),
            "P70 (ms)": round(p70_qd, 2),
            "P95 (ms)": round(p95_qd, 2),
            "P100 (Max ms)": round(p100_qd, 2)
        })

    # Print final comparison reports
    print("\n========================================================")
    print("        Retrieval Accuracy Comparative Sweep")
    print("========================================================")
    df_acc = pd.DataFrame(sweep_results)
    print(df_acc.to_string(index=False))

    print("\n========================================================")
    print("        Qdrant Search Latency Comparative Sweep")
    print("========================================================")
    df_lat = pd.DataFrame(latency_results)
    print(df_lat.to_string(index=False))


if __name__ == "__main__":
    main()
