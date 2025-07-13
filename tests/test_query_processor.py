import pytest
from unittest.mock import MagicMock, patch
from yasrl.query_processor import QueryProcessor
from yasrl.vector_store import VectorStoreManager
from yasrl.providers.embeddings import EmbeddingProvider
from yasrl.models import SourceChunk
from yasrl.exceptions import RetrievalError


@pytest.fixture
def mock_embedding_provider():
    provider = MagicMock(spec=EmbeddingProvider)
    provider.model_name = "mock-model"
    return provider


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStoreManager)
    return store


def test_query_processor_init(mock_vector_store, mock_embedding_provider):
    with patch("yasrl.query_processor.CohereRerank") as mock_rerank, patch(
        "yasrl.query_processor.SentenceTransformer"
    ) as mock_transformer:
        processor = QueryProcessor(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
        )
        assert processor.vector_store == mock_vector_store
        assert processor.embedding_provider == mock_embedding_provider
        mock_rerank.assert_called_once_with(model="BAAI/bge-reranker-base")
        mock_transformer.assert_called_once_with("mock-model")


def test_process_query(mock_vector_store, mock_embedding_provider):
    processor = QueryProcessor(
        vector_store=mock_vector_store, embedding_provider=mock_embedding_provider
    )

    chunks = [
        SourceChunk(text="chunk1", metadata={"source": "doc1"}, score=0.9),
        SourceChunk(text="chunk2", metadata={"source": "doc2"}, score=0.8),
    ]
    mock_vector_store.retrieve.return_value = chunks
    processor.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    processor.rerank_chunks = MagicMock(return_value=chunks)

    result = processor.process_query("test query")

    processor.embed_query.assert_called_once_with("test query")
    mock_vector_store.retrieve.assert_called_once_with([0.1, 0.2, 0.3], top_k=10)
    processor.rerank_chunks.assert_called_once_with("test query", chunks)
    assert result == chunks


def test_process_query_retrieval_error(mock_vector_store, mock_embedding_provider):
    mock_vector_store.retrieve.side_effect = Exception("DB error")
    processor = QueryProcessor(
        vector_store=mock_vector_store, embedding_provider=mock_embedding_provider
    )
    processor.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])

    with pytest.raises(RetrievalError):
        processor.process_query("test query")


def test_embed_query(mock_vector_store, mock_embedding_provider):
    processor = QueryProcessor(
        vector_store=mock_vector_store, embedding_provider=mock_embedding_provider
    )
    with patch(
        "yasrl.query_processor.SentenceTransformer"
    ) as mock_transformer:
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_transformer.return_value = mock_model
        embedding = processor.embed_query("test query")
        assert embedding == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with("test query")


def test_rerank_chunks(mock_vector_store, mock_embedding_provider):
    processor = QueryProcessor(
        vector_store=mock_vector_store, embedding_provider=mock_embedding_provider
    )
    chunks = [
        SourceChunk(text="chunk1", metadata={"source": "doc1"}, score=0.9),
        SourceChunk(text="chunk2", metadata={"source": "doc2"}, score=0.8),
    ]

    with patch.object(processor, "reranker") as mock_reranker:
        mock_reranker.postprocess_nodes.return_value = [
            MagicMock(text="chunk2", metadata={"source": "doc2"}, score=0.95),
            MagicMock(text="chunk1", metadata={"source": "doc1"}, score=0.91),
        ]
        reranked = processor.rerank_chunks("test query", chunks)
        assert len(reranked) == 2
        assert reranked[0].text == "chunk2"
        assert reranked[1].text == "chunk1"


def test_rerank_chunks_failure(mock_vector_store, mock_embedding_provider):
    processor = QueryProcessor(
        vector_store=mock_vector_store, embedding_provider=mock_embedding_provider
    )
    chunks = [
        SourceChunk(text="chunk1", metadata={"source": "doc1"}, score=0.9),
    ]
    with patch.object(processor, "reranker") as mock_reranker:
        mock_reranker.postprocess_nodes.side_effect = Exception("Rerank error")
        reranked = processor.rerank_chunks("test query", chunks)
        assert reranked == chunks
