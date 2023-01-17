from typing import List
from .retriever import AgenticRetriever

class RAGOrchestrator:
    def __init__(self, retriever: AgenticRetriever):
        self.retriever = retriever

    def handle_query(self, query: str) -> str:
        context = self.retriever.retrieve(query)
        # In a real scenario, this would call an LLM with the context
        return f"Synthesized answer based on {len(context)} documents."
