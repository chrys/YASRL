import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from yasrl.pipeline import RAGPipeline
from yasrl.models import QueryResult, SourceChunk

@pytest.fixture
def mock_pipeline():
    with patch('yasrl.pipeline.ConfigurationManager'), \
         patch('yasrl.pipeline.LLMProviderFactory'), \
         patch('yasrl.pipeline.EmbeddingProviderFactory'), \
         patch('yasrl.pipeline.VectorStoreManager'), \
         patch('yasrl.pipeline.TextProcessor'):

        pipeline = RAGPipeline(llm="mock_llm", embed_model="mock_embed")
        pipeline.query_processor = AsyncMock()
        pipeline.llm_provider = MagicMock()
        return pipeline

@pytest.mark.asyncio
async def test_ask_successful(mock_pipeline):
    mock_pipeline.query_processor.process_query.return_value = [
        SourceChunk(text="chunk1", metadata={"source": "doc1"})
    ]
    mock_llm = AsyncMock()
    mock_llm.achat.return_value = MagicMock(message=MagicMock(content="answer"))
    mock_pipeline.llm_provider.get_llm.return_value = mock_llm

    result = await mock_pipeline.ask("query")

    assert isinstance(result, QueryResult)
    assert result.answer == "answer"
    assert len(result.source_chunks) == 1

@pytest.mark.asyncio
async def test_ask_empty_query(mock_pipeline):
    with pytest.raises(ValueError):
        await mock_pipeline.ask("")

@pytest.mark.asyncio
async def test_ask_no_chunks_found(mock_pipeline):
    mock_pipeline.query_processor.process_query.return_value = []
    mock_llm = AsyncMock()
    mock_llm.achat.return_value = MagicMock(message=MagicMock(content="answer"))
    mock_pipeline.llm_provider.get_llm.return_value = mock_llm

    result = await mock_pipeline.ask("query")

    assert result.answer == "answer"
    assert len(result.source_chunks) == 0

@pytest.mark.asyncio
async def test_ask_llm_error(mock_pipeline):
    mock_pipeline.query_processor.process_query.return_value = [
        SourceChunk(text="chunk1", metadata={"source": "doc1"})
    ]
    mock_llm = AsyncMock()
    mock_llm.achat.side_effect = Exception("LLM error")
    mock_pipeline.llm_provider.get_llm.return_value = mock_llm

    result = await mock_pipeline.ask("query")

    assert "Error" in result.answer
    assert len(result.source_chunks) == 0

@pytest.mark.asyncio
async def test_ask_with_conversation_history(mock_pipeline):
    mock_pipeline.query_processor.process_query.return_value = []
    mock_llm = AsyncMock()
    mock_llm.achat.return_value = MagicMock(message=MagicMock(content="answer"))
    mock_pipeline.llm_provider.get_llm.return_value = mock_llm

    history = [{"role": "user", "content": "previous query"}]
    await mock_pipeline.ask("query", conversation_history=history)

    mock_pipeline.llm_provider.get_llm.return_value.achat.assert_called_once()
    prompt = mock_pipeline.llm_provider.get_llm.return_value.achat.call_args[0][0]
    assert "previous query" in prompt
