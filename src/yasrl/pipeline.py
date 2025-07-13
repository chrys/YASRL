import asyncio
import logging
import os
from time import perf_counter
from typing import Optional

from .config.manager import ConfigurationManager
from .exceptions import ConfigurationError, IndexingError
from .loaders import DocumentLoader
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

    async def index(self, source: str | list[str]) -> None:
        """
        Indexes documents from a source, handling upserts and batching.

        Args:
            source: The source to index (file, directory, URL, or list of URLs).
        """
        logger.info(f"Starting indexing process for source: {source}")
        start_time = perf_counter()
        document_loader = DocumentLoader()

        try:
            documents = document_loader.load_documents(source)
            if not documents:
                logger.warning(f"No documents found for source: {source}")
                return
        except IndexingError as e:
            logger.error(f"Failed to load documents from source: {source}. Error: {e}")
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred during document loading: {e}")
            return

        total_docs = len(documents)
        for i, doc in enumerate(documents):
            doc_id = doc.id_ or document_loader.generate_document_id(
                doc.metadata.get("file_path") or doc.metadata.get("extra_info", {}).get("Source", "")
            )
            logger.info(f"Processing document {i + 1}/{total_docs}: {doc_id}")

            try:
                # Process the document into chunks
                nodes = self.text_processor.process_documents([doc])

                # Generate embeddings for the chunks
                texts = [node.text for node in nodes]
                embeddings = await self.embedding_provider.get_embedding_model().get_text_embedding_batch(texts)

                for node, embedding in zip(nodes, embeddings):
                    if hasattr(node, "set_embedding"):
                        node.set_embedding(embedding)
                    elif hasattr(node, "with_embedding"):
                        node = node.with_embedding(embedding)
                    else:
                        logger.warning(f"Cannot set embedding for node: {getattr(node, 'id_', None)} (read-only attribute)")

                # Upsert the chunks into the vector store
                self.db_manager.upsert_documents(document_id=doc_id, chunks=nodes)
                logger.info(f"Successfully indexed document: {doc_id}")

            except Exception as e:
                logger.error(f"Failed to process document {doc_id}. Error: {e}")
                # Continue processing other documents

        end_time = perf_counter()
        logger.info(
            f"Indexing completed in {end_time - start_time:.2f} seconds. "
            f"Processed {total_docs} documents."
        )
