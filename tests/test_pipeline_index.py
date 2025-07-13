import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock, call
from llama_index.core.schema import Document

from yasrl.pipeline import RAGPipeline
from yasrl.exceptions import IndexingError


class TestRAGPipelineIndex(unittest.TestCase):
    def setUp(self):
        # This patch will prevent the RAGPipeline from trying to load a real config
        patcher = patch("yasrl.config.manager.ConfigurationManager")
        self.addCleanup(patcher.stop)
        self.mock_config_manager = patcher.start()
        self.mock_config_manager.return_value.load_config.return_value = MagicMock(
            database=MagicMock(postgres_uri="dummy_uri", vector_dimensions=1)
        )

    @patch("yasrl.pipeline.DocumentLoader")
    @patch("yasrl.pipeline.EmbeddingProviderFactory")
    @patch("yasrl.pipeline.VectorStoreManager")
    @patch("yasrl.pipeline.TextProcessor")
    def test_index_file(
        self,
        MockTextProcessor,
        MockVectorStoreManager,
        MockEmbeddingProviderFactory,
        MockDocumentLoader,
    ):
        # Arrange
        mock_doc_loader = MockDocumentLoader.return_value
        mock_doc_loader.load_documents.return_value = [
            Document(id_="doc1", text="content1")
        ]

        mock_text_processor = MockTextProcessor.return_value
        mock_text_processor.process_documents.return_value = [MagicMock(text="chunk1")]

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(return_value=[[1.0]])
        MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider

        pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
        pipeline.db_manager = MockVectorStoreManager.return_value

        # Act
        asyncio.run(pipeline.index("dummy/path/to/file.txt"))

        # Assert
        mock_doc_loader.load_documents.assert_called_once_with(
            "dummy/path/to/file.txt"
        )
        mock_text_processor.process_documents.assert_called_once()
        mock_embedding_provider.get_embedding_model().get_text_embedding_batch.assert_called_once()
        pipeline.db_manager.upsert_documents.assert_called_once()

    @patch("yasrl.pipeline.DocumentLoader")
    @patch("yasrl.pipeline.EmbeddingProviderFactory")
    @patch("yasrl.pipeline.VectorStoreManager")
    @patch("yasrl.pipeline.TextProcessor")
    def test_index_directory(
        self,
        MockTextProcessor,
        MockVectorStoreManager,
        MockEmbeddingProviderFactory,
        MockDocumentLoader,
    ):
        # Arrange
        mock_doc_loader = MockDocumentLoader.return_value
        mock_doc_loader.load_documents.return_value = [
            Document(id_="doc1", text="content1"),
            Document(id_="doc2", text="content2"),
        ]

        mock_text_processor = MockTextProcessor.return_value
        mock_text_processor.process_documents.side_effect = [
            [MagicMock(text="chunk1")],
            [MagicMock(text="chunk2")],
        ]

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(side_effect=[[[1.0]], [[2.0]]])
        MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider

        pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
        pipeline.db_manager = MockVectorStoreManager.return_value

        # Act
        asyncio.run(pipeline.index("dummy/directory"))

        # Assert
        mock_doc_loader.load_documents.assert_called_once_with("dummy/directory")
        self.assertEqual(mock_text_processor.process_documents.call_count, 2)
        self.assertEqual(mock_embedding_provider.get_embedding_model().get_text_embedding_batch.await_count, 2)
        self.assertEqual(pipeline.db_manager.upsert_documents.call_count, 2)

    @patch("yasrl.pipeline.DocumentLoader")
    @patch("yasrl.pipeline.EmbeddingProviderFactory")
    @patch("yasrl.pipeline.VectorStoreManager")
    @patch("yasrl.pipeline.TextProcessor")
    def test_index_url(
        self,
        MockTextProcessor,
        MockVectorStoreManager,
        MockEmbeddingProviderFactory,
        MockDocumentLoader,
    ):
        # Arrange
        mock_doc_loader = MockDocumentLoader.return_value
        mock_doc_loader.load_documents.return_value = [
            Document(id_="doc1", text="content1")
        ]

        mock_text_processor = MockTextProcessor.return_value
        mock_text_processor.process_documents.return_value = [MagicMock(text="chunk1")]

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(return_value=[[1.0]])
        MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider

        pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
        pipeline.db_manager = MockVectorStoreManager.return_value

        # Act
        asyncio.run(pipeline.index("http://example.com"))

        # Assert
        mock_doc_loader.load_documents.assert_called_once_with("http://example.com")
        mock_text_processor.process_documents.assert_called_once()
        mock_embedding_provider.get_embedding_model().get_text_embedding_batch.assert_called_once()
        pipeline.db_manager.upsert_documents.assert_called_once()

    @patch("yasrl.pipeline.DocumentLoader")
    @patch("yasrl.pipeline.EmbeddingProviderFactory")
    @patch("yasrl.pipeline.VectorStoreManager")
    @patch("yasrl.pipeline.TextProcessor")
    def test_upsert_logic(
        self,
        MockTextProcessor,
        MockVectorStoreManager,
        MockEmbeddingProviderFactory,
        MockDocumentLoader,
    ):
        # Arrange
        mock_doc_loader = MockDocumentLoader.return_value
        mock_doc_loader.load_documents.return_value = [
            Document(id_="doc1", text="content1")
        ]

        mock_text_processor = MockTextProcessor.return_value
        mock_text_processor.process_documents.return_value = [MagicMock(text="chunk1")]

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(return_value=[[1.0]])
        MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider

        pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
        pipeline.db_manager = MockVectorStoreManager.return_value

        # Act
        asyncio.run(pipeline.index("dummy/path/to/file.txt"))
        asyncio.run(pipeline.index("dummy/path/to/file.txt"))

        # Assert
        self.assertEqual(mock_doc_loader.load_documents.call_count, 2)
        self.assertEqual(pipeline.db_manager.upsert_documents.call_count, 2)
        pipeline.db_manager.upsert_documents.assert_called_with(
            document_id="doc1", chunks=[MagicMock(text="chunk1", embedding=[1.0])]
        )

    @patch("yasrl.pipeline.DocumentLoader")
    @patch("yasrl.pipeline.EmbeddingProviderFactory")
    @patch("yasrl.pipeline.VectorStoreManager")
    @patch("yasrl.pipeline.TextProcessor")
    def test_index_error_handling(
        self,
        MockTextProcessor,
        MockVectorStoreManager,
        MockEmbeddingProviderFactory,
        MockDocumentLoader,
    ):
        # Arrange
        mock_doc_loader = MockDocumentLoader.return_value
        mock_doc_loader.load_documents.side_effect = [
            [Document(id_="doc1", text="content1")],
            IndexingError("Failed to load"),
        ]

        mock_text_processor = MockTextProcessor.return_value
        mock_text_processor.process_documents.return_value = [MagicMock(text="chunk1")]

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(return_value=[[1.0]])
        MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider

        pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
        pipeline.db_manager = MockVectorStoreManager.return_value

        # Act
        asyncio.run(pipeline.index("dummy/path/to/file.txt"))
        asyncio.run(pipeline.index("invalid/source"))

        # Assert
        self.assertEqual(mock_doc_loader.load_documents.call_count, 2)
        pipeline.db_manager.upsert_documents.assert_called_once() # Only called for the first, successful run


if __name__ == "__main__":
    unittest.main()
