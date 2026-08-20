from embeddings import EmbeddingModel

def main():
    model = EmbeddingModel()
    
    query = "What is a corporation?"

    relevant_passage = (
        "A corporation is a company or group of people "
        "authorized to act as a single entity and recognized as such in law."
    )

    unrelated_passage = "How to cook rice properly using a pressure cooker."

    query_vector = model.encode_queries([query][0])

    passage_vectors = model.encode_documents([relevant_passage, unrelated_passage])

    relevant_similarity = query_vector @ passage_vectors[0]
    unrelated_similarity = query_vector @ passage_vectors[1]

    print(f"Query vector dimensions: {len(query_vector)}")
    print(f"Relevant passage similarity: {relevant_similarity:.4f}")
    print(f"Unrelated passage similarity: {unrelated_similarity:.4f}")


if __name__ == "__main__":
    main()