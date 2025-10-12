import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from yasrl.query_engine import QueryEngine
from yasrl.models import QueryResult
from llama_index.core.llms import ChatMessage

@pytest.fixture
def mock_query_processor():
    """Fixture for a mocked QueryProcessor."""
    mock = MagicMock()
    # Simulate process_query returning two relevant chunks
    mock.process_query = AsyncMock(return_value=[
        MagicMock(text="Relevant chunk 1"),
        MagicMock(text="Relevant chunk 2")
    ])
    return mock

@pytest.fixture
def mock_reranker():
    """Fixture for a mocked ReRanker."""
    mock = MagicMock()
    # Simulate reranker returning the same chunks it received
    mock.rerank = MagicMock(side_effect=lambda query, chunks: chunks)
    return mock

@pytest.fixture
def mock_llm_provider():
    """Fixture for a mocked LLMProvider."""
    mock = MagicMock()
    return mock

@pytest.mark.asyncio
async def test_query_with_achat_interface(mock_query_processor, mock_reranker, mock_llm_provider):
    """Test the query flow when the LLM has an 'achat' method."""
    # Setup LLM mock for the 'achat' interface
    mock_llm = MagicMock()
    mock_llm.achat = AsyncMock(return_value=MagicMock(message=MagicMock(content="Achat answer")))
    mock_llm_provider.get_llm.return_value = mock_llm

    engine = QueryEngine(mock_query_processor, mock_reranker, mock_llm_provider)
    result = await engine.query("What is test?")

    assert isinstance(result, QueryResult)
    assert "Achat answer" in result.answer
    assert len(result.source_chunks) == 2
    mock_llm.achat.assert_awaited_once()

@pytest.mark.asyncio
async def test_query_with_agenerate_interface(mock_query_processor, mock_reranker, mock_llm_provider):
    """Test the query flow when the LLM has an 'agenerate' method."""
    # Setup LLM mock for the 'agenerate' interface
    mock_llm = MagicMock()
    del mock_llm.achat # Ensure 'achat' is not present
    mock_llm.agenerate = AsyncMock(return_value=MagicMock(generations=[[MagicMock(text="Agenerate answer")]]))
    mock_llm_provider.get_llm.return_value = mock_llm

    engine = QueryEngine(mock_query_processor, mock_reranker, mock_llm_provider)
    result = await engine.query("What is test?")

    assert "Agenerate answer" in result.answer
    mock_llm.agenerate.assert_awaited_once()

@pytest.mark.asyncio
async def test_query_with_sync_generate_interface(mock_query_processor, mock_reranker, mock_llm_provider):
    """Test the query flow when the LLM has a synchronous 'generate' method."""
    # Setup LLM mock for the 'generate' interface
    mock_llm = MagicMock()
    del mock_llm.achat, mock_llm.agenerate
    mock_llm.generate = MagicMock(return_value=MagicMock(generations=[[MagicMock(text="Generate answer")]]))
    mock_llm_provider.get_llm.return_value = mock_llm

    engine = QueryEngine(mock_query_processor, mock_reranker, mock_llm_provider)
    result = await engine.query("What is test?")

    assert "Generate answer" in result.answer
    mock_llm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_query_with_acomplete_interface(mock_query_processor, mock_reranker, mock_llm_provider):
    """Test the query flow when the LLM has an 'acomplete' method."""
    # Setup LLM mock for the 'acomplete' interface
    mock_llm = MagicMock()
    del mock_llm.achat, mock_llm.agenerate, mock_llm.generate
    mock_llm.acomplete = AsyncMock(return_value="Acomplete answer")
    mock_llm_provider.get_llm.return_value = mock_llm

    engine = QueryEngine(mock_query_processor, mock_reranker, mock_llm_provider)
    result = await engine.query("What is test?")

    assert "Acomplete answer" in result.answer
    mock_llm.acomplete.assert_awaited_once()

@pytest.mark.asyncio
async def test_query_with_sync_complete_interface(mock_query_processor, mock_reranker, mock_llm_provider):
    """Test the query flow when the LLM has only a synchronous 'complete' method."""
    # Setup LLM mock for the 'complete' interface
    mock_llm = MagicMock()
    del mock_llm.achat, mock_llm.agenerate, mock_llm.generate, mock_llm.acomplete
    mock_llm.complete = MagicMock(return_value="Complete answer")
    mock_llm_provider.get_llm.return_value = mock_llm

    engine = QueryEngine(mock_query_processor, mock_reranker, mock_llm_provider)
    result = await engine.query("What is test?")

    assert "Complete answer" in result.answer
    mock_llm.complete.assert_called_once()

@pytest.mark.asyncio
async def test_query_with_unknown_llm_interface(mock_query_processor, mock_reranker, mock_llm_provider):
    """Test the fallback behavior for an unknown LLM interface."""
    # Setup LLM mock with no known generation methods
    mock_llm = MagicMock()
    del mock_llm.achat, mock_llm.agenerate, mock_llm.generate, mock_llm.complete, mock_llm.acomplete
    mock_llm_provider.get_llm.return_value = mock_llm

    engine = QueryEngine(mock_query_processor, mock_reranker, mock_llm_provider)
    result = await engine.query("What is test?")

    assert "cannot generate a response" in result.answer

@pytest.mark.asyncio
async def test_query_handles_llm_exception(mock_query_processor, mock_reranker, mock_llm_provider):
    """Test error handling when the LLM call fails."""
    # Setup LLM mock to raise an exception
    mock_llm = MagicMock()
    mock_llm.achat = AsyncMock(side_effect=Exception("LLM API error"))
    mock_llm_provider.get_llm.return_value = mock_llm

    engine = QueryEngine(mock_query_processor, mock_reranker, mock_llm_provider)
    result = await engine.query("What is test?")

    assert "encountered an error generating the response" in result.answer
    assert "LLM API error" in result.answer

@pytest.mark.asyncio
async def test_query_handles_query_processor_exception(mock_query_processor, mock_reranker, mock_llm_provider):
    """Test error handling when the query processor fails."""
    # Query processor raises an exception
    mock_query_processor.process_query.side_effect = Exception("Database connection failed")
    engine = QueryEngine(mock_query_processor, mock_reranker, mock_llm_provider)
    result = await engine.query("What is test?")

    assert "encountered an error while processing your question" in result.answer
    assert "Database connection failed" in result.answer