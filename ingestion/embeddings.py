from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings = True, show_progress_bar = True)

    def encode_queries(self, queries: list[str]):
        return self.model.encode(queries, normalize_embeddings = True, show_progress_bar = True)
