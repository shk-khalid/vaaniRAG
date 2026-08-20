import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks(path: str):
    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks

chunks = load_chunks("data/chunks_passage.jsonl")

print(f"Loaded {len(chunks)} chunks")

def load_queries(path: str):
    queries = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            queries.append(json.loads(line))

    return queries

queries = load_queries("data/evaluation_queries.jsonl")
print(f"Loaded {len(queries)} queries")
print(f"Average number of relevant documents per query: {sum(len(q['relevant_document_ids']) for q in queries) / len(queries)}")


model = SentenceTransformer("BAAI/bge-m3")

passage_texts = [chunk["text"] for chunk in chunks]

passage_embeddings = model.encode(
    passage_texts, 
    normalize_embeddings=True,
    show_progress_bar=True
)
print(f"Passage embeddings shape: {passage_embeddings.shape}")


query = queries[0]

query_text = query["query"]

query_embedding = model.encode([query_text], normalize_embeddings=True)

print(f"Query: {query_text}")
print(f"Query embedding shape: {query_embedding.shape}")
print(f"Query embedding: {query_embedding}")

# Compute cosine similarity scores via dot product
scores = np.dot(passage_embeddings, query_embedding[0])

top_k = 5
top_indices = np.argsort(scores)[-top_k:][::-1]

for index in top_indices:
    chunk = chunks[index]

    print(
        f"\nScore: {scores[index]:.4f}"
        f"\nChunk ID: {chunk['chunk_id']}"
        f"\nSelected: {chunk['is_selected']}"
        f"\nText: {chunk['text'][:200]}"
    )