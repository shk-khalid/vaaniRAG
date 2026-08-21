#!/usr/bin/env python3
"""
benchmark_reranker.py

Evaluates and compares:
1. Qdrant Dense Only retrieval
2. Qdrant Dense + Cross-Encoder Reranked retrieval

Outputs Recall@1, Recall@5, Recall@10, and MRR.
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import List, Dict, Any, Set
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.dense_retriever import QdrantDenseRetriever
from retrieval.reranker import CrossEncoderReranker
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
    parser = argparse.ArgumentParser(description="Benchmark retrieval with Cross-Encoder reranker.")
    parser.add_argument("--collection", type=str, default=QDRANT_COLLECTION, help="Qdrant collection to evaluate")
    args = parser.parse_args()

    queries_path = "data/evaluation_queries.jsonl"
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} evaluation queries.")

    # Share models to save memory and avoid multiple loads
    print("\nLoading BGE-M3 Dense Embedding Model...")
    embed_model = SentenceTransformer("BAAI/bge-m3")
    
    print("Loading Cross-Encoder Reranker Model (MiniLM)...")
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    retriever = QdrantDenseRetriever(collection_name=args.collection, model=embed_model)
    reranker = CrossEncoderReranker(model=cross_model)

    # We evaluate two configurations:
    # 1. Qdrant Dense Only (top 20)
    # 2. Qdrant Dense (top 20) + Reranker (top 20)
    modes = ["dense_only", "dense_reranked"]
    results = []

    for mode in modes:
        print(f"\nEvaluating mode: {mode.upper()}...")
        
        recall_counts = {1: 0.0, 5: 0.0, 10: 0.0}
        total_mrr = 0.0

        for idx, query in enumerate(queries):
            relevant_docs = set(query["relevant_document_ids"])

            # Step 1: Retrieve top 20 candidate chunks from Qdrant
            candidates = retriever.search(query["query"], k=20)

            # Step 2: Apply Reranker if mode is dense_reranked
            if mode == "dense_reranked":
                candidates = reranker.rerank(query["query"], candidates, top_n=20)

            # Step 3: Deduplicate parent document IDs to evaluate unique document recall
            retrieved_docs = []
            seen = set()
            for hit in candidates:
                doc_id = hit["document_id"]
                if doc_id not in seen:
                    seen.add(doc_id)
                    retrieved_docs.append(doc_id)

            # Step 4: Compute metrics
            for k in [1, 5, 10]:
                recall_counts[k] += calculate_recall(retrieved_docs, relevant_docs, k)
            total_mrr += calculate_mrr(retrieved_docs, relevant_docs)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(queries):
                print(f"  Processed {idx + 1}/{len(queries)} queries...")

        num_queries = len(queries)
        results.append({
            "Retriever": mode.upper(),
            "R@1": round(recall_counts[1] / num_queries, 4),
            "R@5": round(recall_counts[5] / num_queries, 4),
            "R@10": round(recall_counts[10] / num_queries, 4),
            "MRR": round(total_mrr / num_queries, 4)
        })

    # Print final comparison report
    print("\n========================================================")
    print("        Qdrant Dense Only vs Qdrant + Reranker")
    print("========================================================")
    df_report = pd.DataFrame(results)
    print(df_report.to_string(index=False))


if __name__ == "__main__":
    main()
