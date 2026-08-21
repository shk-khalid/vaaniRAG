#!/usr/bin/env python3
"""
test_qdrant_retrieval.py

Performs a manual search query against Qdrant to verify indexing and retrieval works.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.dense_retriever import QdrantDenseRetriever


def main():
    parser = argparse.ArgumentParser(description="Test Qdrant retrieval.")
    parser.add_argument("--query", type=str, default="কৰ্পোৰেচন কি?", help="Query text to search")
    parser.add_argument("--k", type=int, default=5, help="Top k matches to retrieve")
    parser.add_argument("--lang", type=str, default=None, help="Language code to filter (e.g. asm_Beng, hin_Deva)")
    args = parser.parse_args()

    print(f"Initializing QdrantDenseRetriever...")
    retriever = QdrantDenseRetriever()

    print(f"\nSearching for: '{args.query}' (Filter: {args.lang})")
    results = retriever.search(args.query, k=args.k, language_filter=args.lang)

    print(f"\nRetrieved {len(results)} results:")
    for idx, hit in enumerate(results):
        print(f"\n  [{idx+1}] Score: {hit['score']:.4f}")
        print(f"      Chunk ID:  {hit['chunk_id']}")
        print(f"      Doc ID:    {hit['document_id']}")
        print(f"      Language:  {hit['language']}")
        print(f"      Selected:  {hit['is_selected']}")
        text_snippet = hit["text"][:150].replace('\n', ' ')
        print(f"      Text:      {text_snippet}...")


if __name__ == "__main__":
    main()
