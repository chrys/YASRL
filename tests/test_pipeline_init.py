import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yasrl.exceptions import ConfigurationError
from yasrl.pipeline import RAGPipeline


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/testdb")
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TEXT_CHUNK_SIZE", "500")


@patch("yasrl.pipeline.LLMProviderFactory")
@patch("yasrl.pipeline.EmbeddingProviderFactory")
@patch("yasrl.pipeline.VectorStoreManager")
@patch("yasrl.pipeline.ConfigurationManager")
@patch("yasrl.pipeline.logging")

def test_pipeline_init_success(
    mock_logging,
    MockConfigManager,
    MockVectorStoreManager,
    MockEmbeddingProviderFactory,
    MockLLMProviderFactory,
):
    """Tests successful initialization of the RAGPipeline."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_api_key",
        "POSTGRES_URI": "postgresql://user:pass@localhost:5432/testdb",
        "LOG_LEVEL": "DEBUG",
        "TEXT_CHUNK_SIZE": "500"
    }, clear=True):
        mock_config = MockConfigManager.return_value
        mock_config.get_required_variables.return_value = ["OPENAI_API_KEY"]

        mock_db_manager = MockVectorStoreManager.return_value
        mock_db_manager.initialize = AsyncMock()

        pipeline = RAGPipeline(llm="openai", embed_model="openai")

        assert pipeline.llm_provider is not None
        assert pipeline.embedding_provider is not None
        assert pipeline.db_manager is not None
        assert pipeline.text_processor is not None
        assert pipeline.text_processor.chunk_size == 500

        MockLLMProviderFactory.create_provider.assert_called_once_with("openai", mock_config)
        MockEmbeddingProviderFactory.create_provider.assert_called_once_with("openai", mock_config)


def test_pipeline_init_missing_env_vars():
    """Tests that ConfigurationError is raised for missing environment variables."""
    with pytest.raises(ConfigurationError) as excinfo:
        RAGPipeline(llm="openai", embed_model="openai")
    assert "Database postgres_uri cannot be empty" in str(excinfo.value)


@patch("yasrl.pipeline.LLMProviderFactory")
@patch("yasrl.pipeline.EmbeddingProviderFactory")
@patch("yasrl.pipeline.VectorStoreManager")
@patch("yasrl.pipeline.ConfigurationManager")
@pytest.mark.asyncio
async def test_pipeline_context_manager(
    MockConfigManager,
    MockVectorStoreManager,
    MockEmbeddingProviderFactory,
    MockLLMProviderFactory,
    mock_env_vars,
):
    """Tests the async context manager functionality."""
    mock_db_manager = MockVectorStoreManager.return_value
    mock_db_manager.ainit = AsyncMock()
    mock_db_manager.close = AsyncMock()

    async with await RAGPipeline.create(llm="openai", embed_model="openai") as pipeline:
        assert pipeline is not None
        assert isinstance(pipeline, RAGPipeline)

@patch("yasrl.pipeline.ConfigurationManager")
@patch("yasrl.pipeline.logging")
def test_logging_setup(mock_logging, MockConfigManager):
    """Tests that logging is configured correctly."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_api_key",
        "POSTGRES_URI": "postgresql://user:pass@localhost:5432/testdb",
        "LOG_LEVEL": "DEBUG"
    }, clear=True):
        mock_config = MockConfigManager.return_value
        mock_config.openai_api_key = "test_api_key"
        RAGPipeline(llm="openai", embed_model="openai")
        mock_logging.basicConfig.assert_called_once()
        args, kwargs = mock_logging.basicConfig.call_args
        assert kwargs["level"] == "DEBUG"
