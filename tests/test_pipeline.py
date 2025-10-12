import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from yasrl.config.app_config import AppConfig
from yasrl.exceptions import ConfigurationError
from yasrl.indexer import Indexer
from yasrl.models import QueryResult
from yasrl.pipeline import RAGPipeline
from yasrl.query_engine import QueryEngine


class TestAppConfig(unittest.TestCase):
    """Tests for the AppConfig class."""

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
            "POSTGRES_URI": "postgresql://user:pass@localhost:5432/testdb",
        },
    )
    def test_successful_initialization(self):
        """Test successful initialization with required environment variables."""
        try:
            config = AppConfig(llm="openai", embed_model="gemini")
            self.assertIsNotNone(config)
        except ConfigurationError:
            self.fail("AppConfig initialization failed unexpectedly.")

    def test_missing_env_vars(self):
        """Test that ConfigurationError is raised when environment variables are missing."""
        with self.assertRaises(ConfigurationError):
            AppConfig(llm="openai", embed_model="gemini")


class TestIndexer(unittest.IsolatedAsyncioTestCase):
    """Tests for the Indexer class."""

    async def test_successful_indexing(self):
        """Test that the indexer successfully processes and upserts documents."""
        mock_embedding_provider = MagicMock()
        mock_db_manager = MagicMock()
        mock_text_processor = MagicMock()

        indexer = Indexer(
            embedding_provider=mock_embedding_provider,
            db_manager=mock_db_manager,
            text_processor=mock_text_processor,
        )

        with patch.object(
            indexer.document_loader, "load_documents", return_value=[MagicMock()]
        ) as mock_load:
            await indexer.index("dummy_source")
            mock_load.assert_called_once_with("dummy_source")
            mock_text_processor.process_documents.assert_called()
            mock_embedding_provider.get_embedding_model().get_text_embedding_batch.assert_called()
            mock_db_manager.upsert_documents.assert_called()


class TestQueryEngine(unittest.IsolatedAsyncioTestCase):
    """Tests for the QueryEngine class."""

    async def test_successful_query(self):
        """Test a successful query that returns a valid result."""
        mock_query_processor = MagicMock()
        mock_reranker = MagicMock()
        mock_llm_provider = MagicMock()

        # Ensure async methods are awaitable
        mock_query_processor.process_query = AsyncMock(return_value=[MagicMock()])
        mock_llm = AsyncMock()
        mock_llm.achat.return_value.message.content = "Test answer"
        mock_llm_provider.get_llm.return_value = mock_llm

        query_engine = QueryEngine(
            query_processor=mock_query_processor,
            reranker=mock_reranker,
            llm_provider=mock_llm_provider,
        )

        result = await query_engine.query("What is love?")
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.answer, "Test answer")


class TestRAGPipeline(unittest.IsolatedAsyncioTestCase):
    """Tests for the refactored RAGPipeline class."""

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
            "POSTGRES_URI": "postgresql://user:pass@localhost:5432/testdb",
        },
    )
    @patch("yasrl.pipeline.EmbeddingProviderFactory")
    @patch("yasrl.pipeline.LLMProviderFactory")
    @patch("yasrl.pipeline.VectorStoreManager")
    async def test_pipeline_initialization(
        self, mock_db_manager, mock_llm_factory, mock_embedding_factory
    ):
        """Test that the RAGPipeline initializes all components correctly."""
        # Arrange
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.get_embedding_model.return_value.get_text_embedding.return_value = [
            0.1
        ] * 10
        mock_embedding_factory.create_provider.return_value = mock_embedding_provider
        mock_db_manager.return_value.ainit = AsyncMock()

        # Act
        pipeline = await RAGPipeline.create(llm="openai", embed_model="gemini")

        # Assert
        self.assertIsInstance(pipeline.app_config, AppConfig)
        self.assertIsNotNone(pipeline.indexer)
        self.assertIsNotNone(pipeline.query_engine)
        mock_db_manager.assert_called()


if __name__ == "__main__":
    unittest.main()