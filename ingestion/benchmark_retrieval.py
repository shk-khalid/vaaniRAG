#!/usr/bin/env python3
"""
benchmark_retrieval.py

Evaluates and compares BGE-M3 dense retrieval performance across 5 different
chunking strategies (passage, adaptive, overlap, sentence, semantic) using
Recall@1, Recall@5, and Recall@10 metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_chunks(path: str):
    chunks = []
    if not os.path.exists(path):
        print(f"Error: Chunk file not found at {path}", file=sys.stderr)
        return None
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))
    return chunks


def load_queries(path: str):
    queries = []
    if not os.path.exists(path):
        print(f"Error: Queries file not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            queries.append(json.loads(line))
    return queries


def retrieve_top_k(
    query_embedding,
    passage_embeddings,
    chunks,
    k: int,
):
    # Vectorized dot product for cosine similarity
    scores = passage_embeddings @ query_embedding
    top_indices = np.argsort(scores)[-k:][::-1]
    return [
        {
            "document_id": chunks[index]["document_id"],
            "score": float(scores[index]),
        }
        for index in top_indices
    ]


def calculate_recall(retrieved_documents, relevant_documents):
    if not relevant_documents:
        return False
    return bool(set(retrieved_documents) & set(relevant_documents))


def main():
    queries_path = "data/evaluation_queries.jsonl"
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} evaluation queries.")
    print(f"Average number of relevant documents per query: {sum(len(q['relevant_document_ids']) for q in queries) / len(queries):.4f}")

    print("\nLoading BGE-M3 SentenceTransformer model...")
    model = SentenceTransformer("BAAI/bge-m3")

    strategies = ["passage", "adaptive", "overlap", "sentence", "semantic"]
    comparison_results = []

    # Pre-encode all queries once to save time
    print("\nPre-encoding all evaluation queries...")
    query_texts = [q["query"] for q in queries]
    query_embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    for strategy in strategies:
        chunks_path = f"data/chunks_{strategy}.jsonl"
        print(f"\n==================================================")
        print(f" Evaluating Strategy: {strategy.upper()}")
        print(f"==================================================")
        
        chunks = load_chunks(chunks_path)
        if chunks is None:
            print(f"Skipping strategy {strategy} (missing chunks file).")
            continue
            
        print(f"Loaded {len(chunks)} chunks.")

        # Encode passages
        print(f"Encoding passage chunks for '{strategy}'...")
        passage_texts = [chunk["text"] for chunk in chunks]
        passage_embeddings = model.encode(
            passage_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        print(f"Passage embeddings shape: {passage_embeddings.shape}")

        # Initialize recall hits tracking for this strategy
        recall_counts = {1: 0, 5: 0, 10: 0}

        # Evaluate queries
        for idx, query in enumerate(queries):
            query_emb = query_embeddings[idx]
            relevant_documents = set(query["relevant_document_ids"])

            for k in [1, 5, 10]:
                results = retrieve_top_k(
                    query_emb,
                    passage_embeddings,
                    chunks,
                    k,
                )

                # Deduplicate retrieved document IDs to evaluate unique document-level recall
                retrieved_documents = []
                for res in results:
                    doc_id = res["document_id"]
                    if doc_id not in retrieved_documents:
                        retrieved_documents.append(doc_id)

                if calculate_recall(retrieved_documents, relevant_documents):
                    recall_counts[k] += 1

        # Calculate metrics
        r_1 = recall_counts[1] / len(queries)
        r_5 = recall_counts[5] / len(queries)
        r_10 = recall_counts[10] / len(queries)

        print(f"Recall@1: {r_1:.4f} | Recall@5: {r_5:.4f} | Recall@10: {r_10:.4f}")
        
        comparison_results.append({
            "Strategy": strategy.capitalize(),
            "R@1": round(r_1, 4),
            "R@5": round(r_5, 4),
            "R@10": round(r_10, 4)
        })

    print("\n==================================================")
    print("        Chunking Strategy Comparison Report")
    print("==================================================")
    df_report = pd.DataFrame(comparison_results)
    print(df_report.to_string(index=False))


if __name__ == "__main__":
    main()