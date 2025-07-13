import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from yasrl.exceptions import ConfigurationError
from yasrl.pipeline import RAGPipeline


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set mock environment variables."""
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")


@pytest.mark.asyncio
async def test_pipeline_initialization_success(mock_env_vars):
    """Tests successful initialization of the RAGPipeline."""
    with patch("yasrl.config.manager.ConfigManager") as mock_config_manager, patch(
        "yasrl.providers.llm.LLMProviderFactory"
    ) as mock_llm_factory, patch(
        "yasrl.providers.embeddings.EmbeddingProviderFactory"
    ) as mock_embedding_factory, patch(
        "yasrl.vector_store.VectorStoreManager"
    ) as mock_vector_store_manager:
        # Arrange
        mock_config_instance = MagicMock()
        mock_config_instance.get_database_config.return_value = {
            "user": "testuser",
            "password": "testpass",
            "host": "localhost",
            "port": "5432",
            "dbname": "testdb",
        }
        mock_config_manager.return_value = mock_config_instance

        mock_llm_provider = MagicMock()
        mock_llm_factory.create.return_value = mock_llm_provider

        mock_embedding_provider = MagicMock()
        mock_embedding_factory.create.return_value = mock_embedding_provider

        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.initialize = asyncio.coroutine(MagicMock())
        mock_vector_store_manager.return_value = mock_vector_store_instance

        # Act
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        await pipeline.initialize()

        # Assert
        mock_config_manager.assert_called_once()
        mock_config_instance.validate_config.assert_called_with("openai", "openai")
        mock_llm_factory.create.assert_called_with("openai")
        mock_embedding_factory.create.assert_called_with("openai")
        mock_vector_store_manager.assert_called_once()
        mock_vector_store_instance.initialize.assert_called_once()
        assert pipeline.llm_provider == mock_llm_provider
        assert pipeline.embedding_provider == mock_embedding_provider
        assert pipeline.vector_store_manager == mock_vector_store_instance
        assert pipeline._is_initialized


@pytest.mark.asyncio
async def test_pipeline_initialization_missing_env_vars():
    """Tests that ConfigurationError is raised for missing environment variables."""
    with pytest.raises(ConfigurationError):
        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        await pipeline.initialize()


@pytest.mark.asyncio
async def test_pipeline_context_manager(mock_env_vars):
    """Tests the async context manager functionality."""
    with patch("yasrl.pipeline.RAGPipeline.initialize") as mock_initialize, patch(
        "yasrl.pipeline.RAGPipeline.cleanup"
    ) as mock_cleanup:
        mock_initialize.return_value = asyncio.coroutine(MagicMock())()
        mock_cleanup.return_value = asyncio.coroutine(MagicMock())()

        pipeline = RAGPipeline(llm="openai", embed_model="openai")
        async with pipeline as p:
            assert p is pipeline
            mock_initialize.assert_called_once()
        mock_cleanup.assert_called_once()


def test_logging_configuration(monkeypatch):
    """Tests that logging is configured based on environment variables."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    # This is tricky to test without capturing output, but we can check the logger's level
    import logging

    # Re-importing pipeline to re-trigger logging configuration
    from yasrl import pipeline

    assert logging.getLogger(pipeline.__name__).level == logging.DEBUG
    monkeypatch.delenv("LOG_LEVEL")
