import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

class AgenticRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents: List[str]):
        self.documents = documents
        embeddings = self.model.encode(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        return [{"doc": self.documents[i], "score": float(distances[0][j])} for j, i in enumerate(indices[0])]
