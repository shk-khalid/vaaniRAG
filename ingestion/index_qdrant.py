#!/usr/bin/env python3
"""
index_qdrant.py

Loads chunks from data/chunks_<strategy>.jsonl, generates embeddings using BGE-M3,
and uploads them to Qdrant database.
"""

import os
import sys
import uuid
import argparse
import json
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models

# Import Qdrant client helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.qdrant_client import get_qdrant_client, init_collection, QDRANT_COLLECTION


def load_chunks(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Error: Chunk file not found at '{path}'. Run process_chunks.py first.", file=sys.stderr)
        sys.exit(1)
    chunks = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))
    return chunks


def build_stable_uuid(chunk_id: str) -> str:
    """
    Generates a stable, reproducible UUIDv5 from the chunk_id string.
    This prevents duplicate entries if indexing is rerun.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def index_chunks(chunks: List[Dict[str, Any]], collection_name: str, batch_size: int = 64) -> None:
    client = get_qdrant_client()
    init_collection(client, collection_name, vector_size=1024)

    print("Loading SentenceTransformer model 'BAAI/bge-m3'...")
    model = SentenceTransformer("BAAI/bge-m3")

    total_chunks = len(chunks)
    print(f"Generating embeddings and indexing {total_chunks} chunks to collection '{collection_name}'...")

    for i in tqdm(range(0, total_chunks, batch_size)):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [c["text"] for c in batch_chunks]

        # Generate BGE-M3 normalized embeddings
        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        points = []
        for idx, chunk in enumerate(batch_chunks):
            point_id = build_stable_uuid(chunk["chunk_id"])
            vector = embeddings[idx].tolist()
            
            # Format payload structure
            payload = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "query_id": chunk["query_id"],
                "text": chunk["text"],
                "language": chunk["language"],
                "is_selected": chunk["is_selected"],
                "strategy": chunk["strategy"],
                "source_passage_index": chunk["source_passage_index"]
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        # Upload batch
        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"\nIndexing successfully completed! Added {total_chunks} points to Qdrant collection '{collection_name}'.")


def main():
    parser = argparse.ArgumentParser(description="Index chunks to Qdrant.")
    parser.add_argument("--strategy", type=str, default="adaptive", help="Chunking strategy to index (passage, adaptive, overlap, sentence, semantic)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding and upsert batch size")
    args = parser.parse_args()

    chunks_path = f"data/chunks_{args.strategy}.jsonl"
    print(f"Loading chunks from: {chunks_path}")
    chunks = load_chunks(chunks_path)

    index_chunks(chunks, collection_name=QDRANT_COLLECTION, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
