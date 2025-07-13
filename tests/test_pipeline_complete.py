import asyncio
import os
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from yasrl.pipeline import RAGPipeline
from yasrl.models import QueryResult, SourceChunk
from yasrl.exceptions import ConfigurationError, yasrlError


class TestRAGPipelineComplete(unittest.TestCase):
    def setUp(self):
        self.llm = "openai"
        self.embed_model = "gemini"

    @patch("yasrl.pipeline.ConfigurationManager")
    @patch("yasrl.pipeline.LLMProviderFactory")
    @patch("yasrl.pipeline.EmbeddingProviderFactory")
    @patch("yasrl.pipeline.VectorStoreManager")
    @patch("yasrl.pipeline.TextProcessor")
    @patch("yasrl.pipeline.QueryProcessor")
    def test_pipeline_initialization(
        self,
        mock_query_processor,
        mock_text_processor,
        mock_db_manager,
        mock_embedding_provider_factory,
        mock_llm_provider_factory,
        mock_config_manager,
    ):
        """Test that the RAGPipeline initializes all its components correctly."""
        # Arrange
        mock_config_manager.return_value.load_config.return_value.database.postgres_uri = "test_uri"
        mock_config_manager.return_value.load_config.return_value.database.vector_dimensions = 768
        mock_db_manager.return_value.ainit = AsyncMock()

        # Patch required environment variables for validation
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "GOOGLE_API_KEY": "test-gemini-key",
            "POSTGRES_URI": "test_uri"
        }, clear=True):
            # Act
            pipeline = asyncio.run(RAGPipeline.create(llm=self.llm, embed_model=self.embed_model))

        # Assert
        mock_config_manager.assert_called_once()
        mock_llm_provider_factory.create_provider.assert_called_with(self.llm, mock_config_manager.return_value)
        mock_embedding_provider_factory.create_provider.assert_called_with(self.embed_model, mock_config_manager.return_value)
        mock_db_manager.assert_called_once()
        mock_text_processor.assert_called_once()
        mock_query_processor.assert_called_once()
        pipeline.db_manager.ainit.assert_called_once()

    @patch("yasrl.pipeline.ConfigurationManager")
    @patch("yasrl.pipeline.RAGPipeline._ainit")
    @patch("yasrl.pipeline.RAGPipeline.cleanup")
    def test_context_manager(self, mock_cleanup, mock_ainit, mock_config_manager):
        """Test that the async context manager (__aenter__ and __aexit__) works correctly."""
        # Arrange
        mock_config_manager.return_value.load_config.return_value.database.postgres_uri = "test_uri"
        # Patch required environment variables for validation
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "GOOGLE_API_KEY": "test-gemini-key",
            "POSTGRES_URI": "test_uri"
        }, clear=True):
            pipeline = RAGPipeline(llm=self.llm, embed_model=self.embed_model)
            mock_ainit.return_value = None
            mock_cleanup.return_value = None

            async def run_test():
                async with pipeline as p:
                    self.assertIs(p, pipeline)
                    mock_ainit.assert_called_once()
                    mock_cleanup.assert_not_called()
                mock_cleanup.assert_called_once()

            # Act & Assert
            asyncio.run(run_test())

    @patch("yasrl.pipeline.ConfigurationManager")
    @patch("yasrl.pipeline.VectorStoreManager")
    def test_health_check(self, mock_db_manager, mock_config_manager):
        """Test the health check method."""
        # Arrange
        mock_config_manager.return_value.load_config.return_value.database.postgres_uri = "test_uri"
        # Patch required environment variables for validation
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "GOOGLE_API_KEY": "test-gemini-key",
            "POSTGRES_URI": "test_uri"
        }, clear=True):
            pipeline = RAGPipeline(llm=self.llm, embed_model=self.embed_model)
            pipeline.db_manager = mock_db_manager
            pipeline.db_manager.check_connection = AsyncMock(return_value=True)

            # Act
            is_healthy = asyncio.run(pipeline.health_check())

            # Assert
            self.assertTrue(is_healthy)
            pipeline.db_manager.check_connection.assert_called_once()

    @patch("yasrl.pipeline.ConfigurationManager")
    @patch("yasrl.pipeline.VectorStoreManager")
    def test_get_statistics(self, mock_db_manager, mock_config_manager):
        """Test the get_statistics method."""
        # Arrange
        mock_config_manager.return_value.load_config.return_value.database.postgres_uri = "test_uri"
        # Patch required environment variables for validation
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "GOOGLE_API_KEY": "test-gemini-key",
            "POSTGRES_URI": "test_uri"
        }, clear=True):
            pipeline = RAGPipeline(llm=self.llm, embed_model=self.embed_model)
            pipeline.db_manager = mock_db_manager
            pipeline.db_manager.get_document_count = AsyncMock(return_value=10)

            # Act
            stats = asyncio.run(pipeline.get_statistics())

            # Assert
            self.assertEqual(stats, {"indexed_documents": 10})
            pipeline.db_manager.get_document_count.assert_called_once()

    def test_public_api_exports(self):
        """Verify that the public API exports work correctly."""
        import yasrl

        self.assertTrue(hasattr(yasrl, "RAGPipeline"))
        self.assertTrue(hasattr(yasrl, "QueryResult"))
        self.assertTrue(hasattr(yasrl, "SourceChunk"))
        self.assertTrue(hasattr(yasrl, "yasrlError"))
        self.assertTrue(hasattr(yasrl, "ConfigurationError"))
        self.assertTrue(hasattr(yasrl, "IndexingError"))
        self.assertTrue(hasattr(yasrl, "RetrievalError"))


if __name__ == "__main__":
    unittest.main()