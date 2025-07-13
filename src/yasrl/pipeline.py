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
from .query_processor import QueryProcessor
from .models import QueryResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    The main class for the YasRL RAG pipeline.

    This class orchestrates the entire RAG process, from indexing documents to
    asking questions and retrieving answers.

    Args:
        llm: The name of the language model provider (e.g., "openai").
        embed_model: The name of the embedding model provider (e.g., "gemini").

    Examples:
        >>> from yasrl import RAGPipeline
        >>>
        >>> async def main():
        ...     pipeline = await RAGPipeline.create(llm="openai", embed_model="gemini")
        ...     await pipeline.index("https://example.com")
        ...     result = await pipeline.ask("What is the capital of France?")
        ...     print(result.answer)
        ...     await pipeline.cleanup()
        >>>
        >>> if __name__ == "__main__":
        ...     import asyncio
        ...     asyncio.run(main())
    """

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
            postgres_uri=config.database.postgres_uri,
            vector_dimensions=config.database.vector_dimensions,
            table_prefix=os.getenv("TABLE_PREFIX", "yasrl"),
        )
        self.text_processor = TextProcessor(
            chunk_size=int(os.getenv("TEXT_CHUNK_SIZE", 1000))
        )

        self.query_processor = QueryProcessor(
            embedding_provider=self.embedding_provider,
            db_manager=self.db_manager,
        )

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
        """Asynchronously initializes the database connection pool."""
        await self.db_manager.ainit()

    @classmethod
    async def create(cls, llm: str, embed_model: str) -> "RAGPipeline":
        """Factory method to create and asynchronously initialize the pipeline."""
        pipeline = cls(llm, embed_model)
        await pipeline._ainit()
        return pipeline

    async def __aenter__(self) -> "RAGPipeline":
        """
        Asynchronously enters the runtime context for the pipeline.

        Initializes the database connection pool.

        Returns:
            The pipeline instance.

        Example:
            >>> async with await RAGPipeline.create("openai", "gemini") as pipeline:
            ...     # Do something with the pipeline
            ...     pass
        """
        await self._ainit()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronously exits the runtime context for the pipeline.

        Cleans up resources, such as closing the database connection pool.
        """
        await self.cleanup()

    async def cleanup(self):
        """
        Cleans up resources used by the pipeline.

        This method should be called when the pipeline is no longer needed
        to ensure graceful shutdown of database connections and other resources.
        """
        logger.info("Cleaning up pipeline resources...")
        await self.db_manager.close()

    async def health_check(self) -> bool:
        """
        Checks the health of the pipeline.

        Returns:
            True if the pipeline is healthy, False otherwise.
        """
        logger.info("Performing health check...")
        return await self.db_manager.check_connection()

    async def get_statistics(self) -> dict:
        """
        Gets statistics about the pipeline.

        Returns:
            A dictionary containing pipeline statistics, such as the number of
            indexed documents.
        """
        logger.info("Getting pipeline statistics...")
        document_count = await self.db_manager.get_document_count()
        return {"indexed_documents": document_count}

    async def index(self, source: str | list[str]) -> None:
        """
        Indexes documents from a source, handling upserts and batching.

        This method loads documents from the given source, processes them into
        chunks, generates embeddings, and upserts them into the vector store.

        Args:
            source: The source to index. Can be a file path, a directory path,
                a URL, or a list of URLs.

        Example:
            >>> await pipeline.index("./my_documents")
            >>> await pipeline.index("https://en.wikipedia.org/wiki/RAG")
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

    async def ask(self, query: str, conversation_history: list[dict] | None = None) -> QueryResult:
        """
        Asks a question to the RAG pipeline.

        This method takes a query, retrieves relevant context from the indexed
        documents, and generates an answer using the language model.

        Args:
            query: The question to ask.
            conversation_history: A list of previous conversation turns to provide
                context to the language model. Each turn is a dictionary with
                "role" and "content" keys.

        Returns:
            A QueryResult object containing the answer and the source chunks
            used to generate the answer.

        Example:
            >>> result = await pipeline.ask("What is RAG?")
            >>> print(result.answer)
            >>> for chunk in result.source_chunks:
            ...     print(chunk.text)
        """
        if not query:
            raise ValueError("Query cannot be empty.")

        source_chunks = await self.query_processor.process_query(query)

        prompt = self._format_prompt(query, source_chunks, conversation_history)

        try:
            # Limit conversation history to the last 5 turns to avoid exceeding token limits
            if conversation_history and len(conversation_history) > 5:
                conversation_history = conversation_history[-5:]

            response = await self.llm_provider.get_llm().achat(prompt)
            answer = response.message.content

            if not answer:
                raise ValueError("LLM returned an empty response.")

        except Exception as e:
            logger.error(f"Error getting response from LLM: {e}")
            return QueryResult(answer="Error getting response from LLM.", source_chunks=[])

        return QueryResult(answer=answer, source_chunks=source_chunks)

    def _format_prompt(self, query: str, context: list, conversation_history: list[dict] | None = None) -> str:
        """
        Formats the prompt for the LLM.
        """
        system_message = (
            "You are a helpful AI assistant. Answer the user's query based on the "
            "provided context. If the context does not contain the answer, say so. "
            "Cite the sources used to answer the query by adding [Source X] to the end of the sentence, where X is the number of the source."
        )

        history_str = ""
        if conversation_history:
            for turn in conversation_history:
                history_str += f"{turn['role']}: {turn['content']}\n"

        context_str = ""
        if context:
            for i, chunk in enumerate(context):
                source = chunk.metadata.get("source", "Unknown")
                context_str += f"Source {i+1} ({source}): {chunk.text}\n\n"
        else:
            context_str = "No context provided."


        prompt = f"""
        {system_message}

        Conversation History:
        {history_str}

        Context:
        {context_str}

        Query: {query}
        """
        return prompt
