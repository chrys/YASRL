import logging
from typing import List

from llama_index.core.postprocessor import CohereRerank
from llama_index.core.schema import NodeWithScore, QueryBundle
from sentence_transformers import SentenceTransformer

from yasrl.exceptions import RetrievalError
from yasrl.models import SourceChunk
from yasrl.providers.embeddings import EmbeddingProvider
from yasrl.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Processes queries by retrieving and re-ranking source chunks.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_provider: EmbeddingProvider,
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.reranker_model = reranker_model
        self.reranker = CohereRerank(model=reranker_model)
        self.embedding_model = SentenceTransformer(embedding_provider.model_name)

    def process_query(
        self, query: str, top_k: int = 10, rerank_top_k: int = 5
    ) -> List[SourceChunk]:
        """
        Processes a query and returns a list of re-ranked source chunks.

        Args:
            query: The query string.
            top_k: The number of chunks to retrieve from the vector store.
            rerank_top_k: The number of chunks to return after re-ranking.

        Returns:
            A list of re-ranked source chunks.
        """
        logger.info(f"Processing query: {query}")
        query_embedding = self.embed_query(query)

        try:
            retrieved_chunks = self.vector_store.retrieve(
                query_embedding, top_k=top_k
            )
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve chunks from vector store: {e}")

        if not retrieved_chunks:
            return []

        reranked_chunks = self.rerank_chunks(query, retrieved_chunks)
        return reranked_chunks[:rerank_top_k]

    def embed_query(self, query: str) -> List[float]:
        """
        Converts a query to an embedding.

        Args:
            query: The query string.

        Returns:
            The query embedding.
        """
        logger.info(f"Embedding query: {query}")
        try:
            return self.embedding_model.encode(query).tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []

    def rerank_chunks(
        self, query: str, chunks: List[SourceChunk]
    ) -> List[SourceChunk]:
        """
        Re-ranks a list of source chunks based on a query.

        Args:
            query: The query string.
            chunks: The list of source chunks to re-rank.

        Returns:
            The re-ranked list of source chunks.
        """
        logger.info(f"Re-ranking {len(chunks)} chunks.")
        if not chunks:
            return []

        try:
            nodes = [
                NodeWithScore(
                    node_id=str(i),
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=chunk.score,
                )
                for i, chunk in enumerate(chunks)
            ]
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes, query_bundle=QueryBundle(query_str=query)
            )
            return [
                SourceChunk(
                    text=node.text,
                    metadata=node.metadata,
                    score=node.score,
                )
                for node in reranked_nodes
            ]
        except Exception as e:
            logger.warning(f"Re-ranking failed, returning original chunks: {e}")
            return chunks
