import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.yasrl.exceptions import ConfigurationError
from src.yasrl.pipeline import RAGPipeline


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TEXT_CHUNK_SIZE", "500")


@patch("src.yasrl.pipeline.LLMProviderFactory")
@patch("src.yasrl.pipeline.EmbeddingProviderFactory")
@patch("src.yasrl.pipeline.VectorStoreManager")
@patch("src.yasrl.pipeline.ConfigurationManager")
def test_pipeline_init_success(
    MockConfigManager,
    MockVectorStoreManager,
    MockEmbeddingProviderFactory,
    MockLLMProviderFactory,
    mock_env_vars,
):
    """Tests successful initialization of the RAGPipeline."""
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

    MockLLMProviderFactory.create.assert_called_once_with("openai")
    MockEmbeddingProviderFactory.create.assert_called_once_with("openai")


def test_pipeline_init_missing_env_vars():
    """Tests that ConfigurationError is raised for missing environment variables."""
    with pytest.raises(ConfigurationError) as excinfo:
        RAGPipeline(llm="openai", embed_model="openai")
    assert "Missing required environment variables" in str(excinfo.value)


@patch("src.yasrl.pipeline.LLMProviderFactory")
@patch("src.yasrl.pipeline.EmbeddingProviderFactory")
@patch("src.yasrl.pipeline.VectorStoreManager")
@patch("src.yasrl.pipeline.ConfigurationManager")
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
    mock_db_manager.initialize = AsyncMock()
    mock_db_manager.close = AsyncMock()

    async with await RAGPipeline.create(llm="openai", embed_model="openai") as pipeline:
        assert pipeline is not None
        mock_db_manager.initialize.assert_awaited_once()

    mock_db_manager.close.assert_awaited_once()


@patch("src.yasrl.pipeline.logging")
def test_logging_setup(mock_logging, mock_env_vars):
    """Tests that logging is configured correctly."""
    RAGPipeline(llm="openai", embed_model="openai")
    mock_logging.basicConfig.assert_called_once()
    args, kwargs = mock_logging.basicConfig.call_args
    assert kwargs["level"] == "DEBUG"
