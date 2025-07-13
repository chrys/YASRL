from datetime import datetime
from typing import Optional

from llama_index.core.schema import Document
from llama_index.core.text_splitter import SentenceSplitter

from yasrl.models import Node
from yasrl.providers.embeddings import EmbeddingProviderFactory


class TextProcessor:
    def __init__(self, chunk_size: int, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_documents(self, documents: list[Document]) -> list[Node]:
        nodes = []
        for document in documents:
            nodes.extend(
                self.create_nodes_from_text(
                    text=document.text, metadata=document.metadata
                )
            )
        return nodes

    def create_nodes_from_text(self, text: str, metadata: dict) -> list[Node]:
        nodes = []
        text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        text_chunks = text_splitter.split_text(text)
        for i, text_chunk in enumerate(text_chunks):
            node_metadata = {
                **metadata,
                "chunk_position": i,
                "total_chunks": len(text_chunks),
                "created_at": datetime.now().isoformat(),
            }
            nodes.append(Node(text=text_chunk, metadata=node_metadata))
        return nodes

    @staticmethod
    def optimize_chunk_size(
        embedding_provider: str, chunk_size: Optional[int] = None
    ) -> int:
        if chunk_size:
            return chunk_size
        provider = EmbeddingProviderFactory.get_embedding_provider(embedding_provider)
        return provider.get_max_chunk_size()
