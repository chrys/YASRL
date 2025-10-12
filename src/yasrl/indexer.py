import logging
from time import perf_counter
from typing import List

from .exceptions import IndexingError
from .loaders import DocumentLoader
from .providers.embeddings import EmbeddingProvider
from .text_processor import TextProcessor
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class Indexer:
    """
    Handles the indexing process, including document loading, chunking,
    embedding, and upserting into the vector store.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_manager: VectorStoreManager,
        text_processor: TextProcessor,
    ):
        """
        Initializes the Indexer.

        Args:
            embedding_provider: The embedding provider instance.
            db_manager: The vector store manager instance.
            text_processor: The text processor instance.
        """
        self.embedding_provider = embedding_provider
        self.db_manager = db_manager
        self.text_processor = text_processor
        self.document_loader = DocumentLoader()

    async def index(
        self, source: str | List[str], project_id: str | None = None
    ) -> None:
        """
        Indexes documents from a source, handling upserts and batching.

        Args:
            source: A file path, directory path, URL, or list of URLs.
            project_id: Optional ID to associate the indexed documents with a project.
        """
        logger.info(f"Starting indexing process for source: {source}")
        start_time = perf_counter()

        try:
            documents = self.document_loader.load_documents(source)
            if not documents:
                logger.warning(f"No documents found for source: {source}")
                return
        except IndexingError as e:
            logger.error(f"Failed to load documents: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during document loading: {e}")
            raise IndexingError(f"Failed to load documents from {source}") from e

        total_docs = len(documents)
        logger.info(f"Loaded {total_docs} documents.")

        for i, doc in enumerate(documents):
            doc_id = doc.id_ or self.document_loader.generate_document_id(
                doc.metadata.get("file_path")
                or doc.metadata.get("extra_info", {}).get("Source", "")
            )
            logger.info(f"Processing document {i + 1}/{total_docs}: {doc_id}")

            try:
                # Process the document into chunks (nodes)
                nodes = self.text_processor.process_documents([doc])

                if project_id:
                    for node in nodes:
                        node.metadata["project_id"] = project_id

                # Generate embeddings for the chunks
                texts = [node.text for node in nodes]
                embeddings = (
                    self.embedding_provider.get_embedding_model().get_text_embedding_batch(
                        texts
                    )
                )

                for node, embedding in zip(nodes, embeddings):
                    node.embedding = embedding

                # Upsert the chunks into the vector store
                self.db_manager.upsert_documents(document_id=doc_id, chunks=nodes)
                logger.info(f"Successfully indexed document: {doc_id}")

            except Exception as e:
                logger.error(f"Failed to process document {doc_id}: {e}")
                # Optionally re-raise or handle to stop/continue indexing
                continue

        end_time = perf_counter()
        logger.info(
            f"Indexing completed in {end_time - start_time:.2f} seconds. "
            f"Processed {total_docs} documents."
        )