import asyncio
import logging
import os
from time import perf_counter
from typing import Optional

from .config.manager import ConfigurationManager
from .exceptions import ConfigurationError
from .providers.embeddings import EmbeddingProviderFactory
from .providers.llm import LLMProviderFactory
from .text_processor import TextProcessor
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, llm: str, embed_model: str):
        """
        Initializes the RAG pipeline.

        Args:
            llm: The name of the language model provider.
            embed_model: The name of the embedding model provider.
        """
        self._setup_logging()
        logger.info("Initializing RAG pipeline...")
        start_time = perf_counter()

        self.config_manager = ConfigurationManager()
        config = self.config_manager.load_config()
        self._validate_env_vars(llm, embed_model)

        self.llm_provider = LLMProviderFactory.create_provider(llm, self.config_manager)
        self.embedding_provider = EmbeddingProviderFactory.create_provider(
            embed_model, self.config_manager
        )

        self.db_manager = VectorStoreManager(
            postgres_uri= config.database.postgres_uri,
            vector_dimensions= config.database.vector_dimensions,
            table_prefix=os.getenv("TABLE_PREFIX", "yasrl"),
        )
        self.text_processor = TextProcessor(
            chunk_size=int(os.getenv("TEXT_CHUNK_SIZE", 1000))
        )
        # self.query_processor = QueryProcessor(self.llm_provider, self.db_manager) # QueryProcessor is not defined yet

        end_time = perf_counter()
        logger.info(
            "RAG pipeline initialized in %.2f seconds.", end_time - start_time
        )

    def _setup_logging(self):
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _validate_env_vars(self, llm: str, embed_model: str):
        """Validates that the required environment variables are set."""
        config = self.config_manager.load_config()
        missing_vars = []

        # Example checks (customize as needed for your config structure)
        if llm == "openai" and not os.getenv("OPENAI_API_KEY"):
            missing_vars.append("OPENAI_API_KEY")
        if embed_model == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            missing_vars.append("GOOGLE_API_KEY")
        if not config.database.postgres_uri:
            missing_vars.append("POSTGRES_URI")

        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    async def _ainit(self):
        """Asynchronously initializes the database connection."""
        try:
            conn = self.db_manager._get_connection()
            conn.close()
            logger.info("Database connection initialized.")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise ConfigurationError(f"Database connection failed: {e}")

    @classmethod
    async def create(cls, llm: str, embed_model: str) -> "RAGPipeline":
        """Factory method to create and asynchronously initialize the pipeline."""
        pipeline = cls(llm, embed_model)
        await pipeline._ainit()
        return pipeline

    async def close(self):
        """Closes the database connection."""
        if hasattr(self.db_manager, "_connection") and self.db_manager._connection:
            self.db_manager._connection.close()
            logger.info("Database connection closed.")

    async def __aenter__(self):
        await self._ainit()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
