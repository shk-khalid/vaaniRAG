#!/usr/bin/env python3
"""
benchmark_qdrant.py

Evaluates the dense retrieval performance of the Qdrant backend (local or cloud)
across all 400 queries, reporting Recall@1, Recall@5, Recall@10, and MRR.
"""

import os
import sys
import argparse
import json
import pandas as pd
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
    args = parser.parse_args()

    queries_path = "data/evaluation_queries.jsonl"
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} evaluation queries.")

    print("\nInitializing QdrantDenseRetriever (will load BGE-M3 model)...")
    retriever = QdrantDenseRetriever(collection_name=args.collection)

    print("\nEvaluating dense retrieval performance against Qdrant backend...")
    
    recall_counts = {1: 0.0, 5: 0.0, 10: 0.0}
    total_mrr = 0.0
    
    # Track per-language performance
    languages = sorted(list(set(q["language"] for q in queries)))
    lang_recall_counts = {lang: {1: 0.0, 5: 0.0, 10: 0.0} for lang in languages}
    lang_total_mrr = {lang: 0.0 for lang in languages}
    lang_counts = {lang: 0 for lang in languages}

    for idx, query in enumerate(queries):
        lang = query["language"]
        relevant_docs = set(query["relevant_document_ids"])
        lang_counts[lang] += 1

        # Search Qdrant for top matches (fetch top 20 to allow unique document extraction)
        hits = retriever.search(query["query"], k=20)
        
        # Deduplicate retrieved parent document IDs
        retrieved_docs = []
        seen = set()
        for hit in hits:
            doc_id = hit["document_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                retrieved_docs.append(doc_id)

        # Compute metrics
        for k in [1, 5, 10]:
            rec_val = calculate_recall(retrieved_docs, relevant_docs, k)
            recall_counts[k] += rec_val
            lang_recall_counts[lang][k] += rec_val

        mrr_val = calculate_mrr(retrieved_docs, relevant_docs)
        total_mrr += mrr_val
        lang_total_mrr[lang] += mrr_val

        if (idx + 1) % 50 == 0 or (idx + 1) == len(queries):
            print(f"  Processed {idx + 1}/{len(queries)} queries...")

    # Format reports
    num_queries = len(queries)
    
    overall_results = [{
        "Retriever": "QDRANT DENSE",
        "R@1": round(recall_counts[1] / num_queries, 4),
        "R@5": round(recall_counts[5] / num_queries, 4),
        "R@10": round(recall_counts[10] / num_queries, 4),
        "MRR": round(total_mrr / num_queries, 4)
    }]

    lang_results = []
    for lang in languages:
        count = lang_counts[lang]
        if count > 0:
            lang_results.append({
                "Language": lang.upper(),
                "R@1": round(lang_recall_counts[lang][1] / count, 4),
                "R@5": round(lang_recall_counts[lang][5] / count, 4),
                "R@10": round(lang_recall_counts[lang][10] / count, 4),
                "MRR": round(lang_total_mrr[lang] / count, 4)
            })

    print("\n========================================================")
    print(f"        Qdrant Dense Retrieval OVERALL Performance")
    print("========================================================")
    df_overall = pd.DataFrame(overall_results)
    print(df_overall.to_string(index=False))

    print("\n========================================================")
    print(f"        Qdrant Dense Retrieval PER-LANGUAGE Performance")
    print("========================================================")
    df_lang = pd.DataFrame(lang_results)
    print(df_lang.to_string(index=False))


if __name__ == "__main__":
    main()
