import asyncio
import logging
import os
from time import perf_counter
from typing import Optional, List

from .config.app_config import AppConfig
from .indexer import Indexer
from .query_engine import QueryEngine
from .providers.embeddings import EmbeddingProviderFactory
from .providers.llm import LLMProviderFactory
from .text_processor import TextProcessor
from .vector_store import VectorStoreManager
from .query_processor import QueryProcessor
from .models import QueryResult
from .reranker import ReRanker

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    The main class for the YasRL RAG pipeline.

    This class orchestrates the entire RAG process, from indexing documents to
    asking questions and retrieving answers.

    Args:
        llm: The name of the language model provider (e.g., "openai").
        embed_model: The name of the embedding model provider (e.g., "gemini").
        db_manager: Optional VectorStoreManager instance.
    """

    def __init__(
        self, llm: str, embed_model: str, db_manager: Optional[VectorStoreManager] = None
    ):
        """
        Initializes the RAG pipeline.
        """
        logger.info("Initializing RAG pipeline...")
        start_time = perf_counter()

        self.app_config = AppConfig(llm=llm, embed_model=embed_model)
        self.config = self.app_config.config

        self.llm_provider = LLMProviderFactory.create_provider(
            llm, self.app_config.config_manager
        )
        self.embedding_provider = EmbeddingProviderFactory.create_provider(
            embed_model, self.app_config.config_manager
        )

        sample_embedding = self.embedding_provider.get_embedding_model().get_text_embedding(
            "test"
        )
        actual_embed_dim = len(sample_embedding)
        logger.info(f"Using embedding dimensions: {actual_embed_dim}")

        if db_manager:
            self.db_manager = db_manager
        else:
            self.db_manager = VectorStoreManager(
                postgres_uri=self.config.database.postgres_uri,
                vector_dimensions=actual_embed_dim,
                table_prefix=self.config.database.table_prefix,
            )

        text_processor = TextProcessor(
            chunk_size=int(os.getenv("TEXT_CHUNK_SIZE", 1000))
        )
        self.indexer = Indexer(
            embedding_provider=self.embedding_provider,
            db_manager=self.db_manager,
            text_processor=text_processor,
        )

        query_processor = QueryProcessor(
            embedding_provider=self.embedding_provider, db_manager=self.db_manager
        )
        reranker = ReRanker(
            model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2", top_n=1
        )
        self.query_engine = QueryEngine(
            query_processor=query_processor,
            reranker=reranker,
            llm_provider=self.llm_provider,
        )

        end_time = perf_counter()
        logger.info(
            "RAG pipeline initialized in %.2f seconds.", end_time - start_time
        )

    async def _ainit(self):
        """Asynchronously initializes the database connection pool."""
        await self.db_manager.ainit()

    @classmethod
    async def create(
        cls, llm: str, embed_model: str, db_manager: Optional[VectorStoreManager] = None
    ) -> "RAGPipeline":
        """Factory method to create and asynchronously initialize the pipeline."""
        pipeline = cls(llm, embed_model, db_manager=db_manager)
        await pipeline._ainit()
        return pipeline

    async def __aenter__(self) -> "RAGPipeline":
        await self._ainit()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def cleanup(self):
        """Cleans up resources used by the pipeline."""
        logger.info("Cleaning up pipeline resources...")
        await self.db_manager.close()

    async def health_check(self) -> bool:
        """Checks the health of the pipeline."""
        logger.info("Performing health check...")
        return await self.db_manager.check_connection()

    async def get_statistics(self, project_id: Optional[str] = None) -> dict:
        """Gets statistics about the pipeline or a specific project."""
        logger.info("Getting pipeline statistics...")
        document_count = await self.db_manager.get_document_count(
            project_id=project_id
        )
        stats: dict = {"indexed_documents": document_count}
        if project_id:
            stats["project_id"] = project_id
        return stats

    async def index(
        self, source: str | List[str], project_id: Optional[str] = None
    ) -> None:
        """
        Indexes documents from a source.

        Args:
            source: The source to index (file path, URL, etc.).
            project_id: Optional project ID to associate with the documents.
        """
        await self.indexer.index(source, project_id=project_id)

    async def ask(
        self, query: str, conversation_history: Optional[List[dict]] = None
    ) -> QueryResult:
        """
        Asks a question and returns the answer.

        Args:
            query: The question to ask.
            conversation_history: Optional conversation history.

        Returns:
            The query result.
        """
        return await self.query_engine.query(query, conversation_history)