import logging
from typing import List, Optional

from .models import SourceChunk
from .vector_store import VectorStoreManager
from .providers.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class QueryProcessor:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_manager: VectorStoreManager,
    ):
        self.embedding_provider = embedding_provider
        self.db_manager = db_manager

    async def process_query(self, query: str, top_k: int = 5) -> List[SourceChunk]:
        if not query:
            raise ValueError("Query cannot be empty.")

        query_embedding = await self.embedding_provider.get_embedding_model().get_text_embedding(query)

        retrieved_chunks = self.db_manager.query(embedding=query_embedding, top_k=top_k)

        if not retrieved_chunks:
            logger.warning("No relevant chunks found for the query.")
            return []

        source_chunks = [
            SourceChunk(
                text=chunk.text,
                metadata=chunk.metadata,
                score=chunk.score
            ) for chunk in retrieved_chunks
        ]

        return source_chunks
