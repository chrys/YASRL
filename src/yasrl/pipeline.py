import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter

from yasrl.config.manager import ConfigManager
from yasrl.exceptions import ConfigurationError
from yasrl.providers.embeddings import EmbeddingProviderFactory
from yasrl.providers.llm import LLMProviderFactory
from yasrl.text_processor import TextProcessor
from yasrl.vector_store import VectorStoreManager

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, llm: str, embed_model: str):
        """
        Initializes the RAGPipeline.

        Args:
            llm (str): The name of the language model provider.
            embed_model (str): The name of the embedding model provider.
        """
        self.llm_name = llm
        self.embed_model_name = embed_model
        self._is_initialized = False
        self.config_manager = None
        self.llm_provider = None
        self.embedding_provider = None
        self.vector_store_manager = None
        self.text_processor = None
        self.query_processor = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def initialize(self):
        """
        Asynchronously initializes the pipeline components.
        """
        if self._is_initialized:
            logger.info("Pipeline already initialized.")
            return

        logger.info("Initializing RAGPipeline...")
        start_time = perf_counter()

        try:
            # Initialize ConfigManager and validate environment variables
            self.config_manager = ConfigManager()
            self.config_manager.validate_config(self.llm_name, self.embed_model_name)
            logger.info("Configuration validated.")

            # Create LLM and embedding providers
            self.llm_provider = LLMProviderFactory.create(self.llm_name)
            self.embedding_provider = EmbeddingProviderFactory.create(self.embed_model_name)
            logger.info(
                f"LLM provider '{self.llm_name}' and embedding provider '{self.embed_model_name}' created."
            )

            # Initialize VectorStoreManager
            db_config = self.config_manager.get_database_config()
            self.vector_store_manager = VectorStoreManager(
                user=db_config["user"],
                password=db_config["password"],
                host=db_config["host"],
                port=db_config["port"],
                dbname=db_config["dbname"],
            )
            await self.vector_store_manager.initialize()
            logger.info("VectorStoreManager initialized.")

            # Set up TextProcessor
            self.text_processor = TextProcessor(chunk_size=1024)
            logger.info("TextProcessor initialized.")

            # Set up QueryProcessor
            # self.query_processor = QueryProcessor(self.llm_provider, self.vector_store_manager)
            # logger.info("QueryProcessor initialized.")

            self._is_initialized = True
            end_time = perf_counter()
            logger.info(
                f"RAGPipeline initialization complete in {end_time - start_time:.2f} seconds."
            )

        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize RAGPipeline: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """
        Cleans up resources, like database connections.
        """
        if self.vector_store_manager:
            await self.vector_store_manager.close()
            logger.info("Database connections closed.")
        self._is_initialized = False

    @asynccontextmanager
    async def managed_pipeline(self):
        """
        A context manager for the RAGPipeline.
        """
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()
