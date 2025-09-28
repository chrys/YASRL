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
        """
        Process a query and return relevant source chunks.
        
        Args:
            query: The query string
            top_k: Number of top results to return
            
        Returns:
            List of SourceChunk objects
            
        Raises:
            ValueError: If query is empty
            RetrievalError: If retrieval fails
        """
        if not query:
            raise ValueError("Query cannot be empty.")

        try:
            # Generate embedding for the query - REMOVE await here
            query_embedding = self.embedding_provider.get_embedding_model().get_text_embedding(query)

            # Retrieve similar chunks from the database - use retrieve_chunks method
            retrieved_chunks = self.db_manager.retrieve_chunks(query_embedding, top_k=top_k)

            if not retrieved_chunks:
                logger.warning("No relevant chunks found for the query.")
                return []

            # Convert results to SourceChunk objects
            source_chunks = []
            for node in retrieved_chunks:
                source_chunk = SourceChunk(
                    text=node.text,
                    metadata=node.metadata or {},
                    score=getattr(node, 'score', 0.0) # Safely get score, default to 0.0
                )
                source_chunks.append(source_chunk)
            
            for retrieved_metadata in [chunk.metadata for chunk in source_chunks]:
                if 'source' in retrieved_metadata:
                    logger.info(f"Retrieved source: {retrieved_metadata['source']}")
                if 'title' in retrieved_metadata:
                    logger.info(f"Retrieved title: {retrieved_metadata['title']}")
                if 'url' in retrieved_metadata:
                    logger.info(f"Retrieved URL: {retrieved_metadata['url']}")
            logger.info(f"Retrieved metadata {source_chunks[0].metadata} for the first chunk.")
            
            
            logger.info(f"Retrieved {len(source_chunks)} chunks for query: {query}")
            return source_chunks
            
        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")
            from .exceptions import RetrievalError
            raise RetrievalError(f"Failed to process query: {e}") from e