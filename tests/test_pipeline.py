import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from yasrl.pipeline import RAGPipeline
from yasrl.vector_store import VectorStoreManager
from yasrl.models import QueryResult

@pytest.fixture
def mock_config_manager():
    mock = MagicMock()
    mock.load_config.return_value = MagicMock(
        database=MagicMock(
            postgres_uri="postgresql://test",
            table_prefix="test_prefix"
        ),
        log_level="INFO",
        log_output="console",
        log_file="yasrl.log"
    )
    return mock

@pytest.fixture
def mock_embedding_provider():
    mock = MagicMock()
    mock.get_embedding_model.return_value.get_text_embedding.return_value = [0.1, 0.2, 0.3]
    mock.get_embedding_model.return_value.get_text_embedding_batch.return_value = [
        [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]
    ]
    return mock

@pytest.fixture
def mock_llm_provider():
    mock = MagicMock()
    mock.get_llm.return_value.generate.return_value.generations = [[MagicMock(text="Test answer")]]
    return mock

@pytest.fixture
def mock_db_manager():
    mock = MagicMock(spec=VectorStoreManager)
    mock.ainit = AsyncMock()
    mock.close = AsyncMock()
    mock.check_connection = AsyncMock(return_value=True)
    mock.get_document_count = AsyncMock(return_value=42)
    return mock

@pytest.fixture
def pipeline(monkeypatch, mock_config_manager, mock_embedding_provider, mock_llm_provider, mock_db_manager):
    with patch("yasrl.pipeline.ConfigurationManager", return_value=mock_config_manager), \
         patch("yasrl.pipeline.EmbeddingProviderFactory.create_provider", return_value=mock_embedding_provider), \
         patch("yasrl.pipeline.LLMProviderFactory.create_provider", return_value=mock_llm_provider), \
         patch("yasrl.pipeline.VectorStoreManager", return_value=mock_db_manager), \
         patch("yasrl.pipeline.TextProcessor"), \
         patch("yasrl.pipeline.QueryProcessor"), \
         patch("yasrl.pipeline.ReRanker"):
        return RAGPipeline(llm="openai", embed_model="gemini", db_manager=mock_db_manager)

@pytest.mark.asyncio
async def test_create_and_cleanup(pipeline, mock_db_manager):
    # Test async creation and cleanup
    with patch.object(RAGPipeline, "_ainit", AsyncMock()):
        p = await RAGPipeline.create("openai", "gemini", db_manager=mock_db_manager)
        assert isinstance(p, RAGPipeline)
        await p.cleanup()
        mock_db_manager.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_health_check(pipeline, mock_db_manager):
    result = await pipeline.health_check()
    assert result is True
    mock_db_manager.check_connection.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_statistics_all(pipeline, mock_db_manager):
    stats = await pipeline.get_statistics()
    assert stats == {"indexed_documents": 42}
    mock_db_manager.get_document_count.assert_awaited_once_with()

@pytest.mark.asyncio
async def test_get_statistics_project(pipeline, mock_db_manager):
    stats = await pipeline.get_statistics(project_id="proj1")
    assert stats == {"indexed_documents": 42, "project_id": "proj1"}
    mock_db_manager.get_document_count.assert_awaited_once_with(project_id="proj1")

@pytest.mark.asyncio
async def test_index_handles_no_documents(pipeline):
    with patch("yasrl.pipeline.DocumentLoader") as MockLoader:
        loader = MockLoader.return_value
        loader.load_documents.return_value = []
        # Should log a warning and return
        await pipeline.index("dummy_source")

@pytest.mark.asyncio
async def test_index_handles_indexing_error(pipeline):
    with patch("yasrl.pipeline.DocumentLoader") as MockLoader:
        loader = MockLoader.return_value
        loader.load_documents.side_effect = Exception("Indexing failed")
        await pipeline.index("dummy_source")


@pytest.mark.asyncio
async def test_ask_handles_no_reranked_chunks(pipeline):
    pipeline.query_processor.process_query = AsyncMock(return_value=[])
    pipeline.reranker.rerank = MagicMock(return_value=[])
    result = await pipeline.ask("What is test?")
    assert "couldn't find any relevant information" in result.answer

@pytest.mark.asyncio
async def test_ask_handles_llm_error(pipeline):
    mock_chunk = MagicMock()
    mock_chunk.text = "context chunk"
    pipeline.query_processor.process_query = AsyncMock(return_value=[mock_chunk])
    pipeline.reranker.rerank = MagicMock(return_value=[mock_chunk])
    pipeline.llm_provider.get_llm.return_value.generate.side_effect = Exception("LLM error")
    result = await pipeline.ask("What is test?")
    assert "encountered an error" in result.answer

def test_format_prompt(pipeline):
    query = "What is Python?"
    context = [MagicMock(text="Python is a programming language.", metadata={"source": "Wikipedia"})]
    conversation_history = [{"role": "user", "content": "Tell me about programming."}]
    prompt = pipeline._format_prompt(query, context, conversation_history)
    assert "Python is a programming language." in prompt
    assert "Tell me about programming." in prompt
    assert "What is Python?" in prompt