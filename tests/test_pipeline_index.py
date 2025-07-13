import asyncio
import os
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

    @patch("yasrl.pipeline.ConfigurationManager")
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
        MockConfigManager,
    ):
        mock_config = MockConfigManager.return_value
        mock_config.load_config.return_value = MagicMock(
            database=MagicMock(postgres_uri="postgres://test:test@localhost/db")
        )
        mock_config.openai_api_key = "test-key"
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "POSTGRES_URI": "postgres://test:test@localhost/db"
        }, clear=True):
            mock_doc_loader = MockDocumentLoader.return_value
            mock_doc_loader.load_documents.return_value = [
                Document(id_="doc1", text="content1")
            ]
            mock_text_processor = MockTextProcessor.return_value
            mock_chunk = MagicMock(text="chunk1")
            def set_embedding_side_effect(val):
                mock_chunk.embedding = val
            mock_chunk.set_embedding.side_effect = set_embedding_side_effect
            mock_text_processor.process_documents.return_value = [mock_chunk]
            mock_embedding_provider = MagicMock()
            mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(return_value=[[1.0]])
            MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider
            # Fix: Make db_manager.ainit awaitable
            MockVectorStoreManager.return_value.ainit = AsyncMock()
            pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
            pipeline.db_manager = MockVectorStoreManager.return_value
            asyncio.run(pipeline.index("dummy/path/to/file.txt"))
            mock_doc_loader.load_documents.assert_called_once_with("dummy/path/to/file.txt")
            mock_text_processor.process_documents.assert_called_once()
            mock_embedding_provider.get_embedding_model().get_text_embedding_batch.assert_called_once()
            pipeline.db_manager.upsert_documents.assert_called_once()
            call_args = pipeline.db_manager.upsert_documents.call_args
            chunk = call_args.kwargs['chunks'][0]
            self.assertEqual(chunk.text, "chunk1")
            self.assertEqual(chunk.embedding, [1.0])

    @patch("yasrl.pipeline.ConfigurationManager")
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
        MockConfigManager,
    ):
        mock_config = MockConfigManager.return_value
        mock_config.load_config.return_value = MagicMock(
            database=MagicMock(postgres_uri="postgres://test:test@localhost/db")
        )
        mock_config.openai_api_key = "test-key"
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "POSTGRES_URI": "postgres://test:test@localhost/db"
        }, clear=True):
            mock_doc_loader = MockDocumentLoader.return_value
            mock_doc_loader.load_documents.return_value = [
                Document(id_="doc1", text="content1"),
                Document(id_="doc2", text="content2"),
            ]
            mock_text_processor = MockTextProcessor.return_value
            chunk1 = MagicMock(text="chunk1")
            chunk2 = MagicMock(text="chunk2")
            chunk1.set_embedding.side_effect = lambda val: setattr(chunk1, "embedding", val)
            chunk2.set_embedding.side_effect = lambda val: setattr(chunk2, "embedding", val)
            mock_text_processor.process_documents.side_effect = [
                [chunk1],
                [chunk2],
            ]
            mock_embedding_provider = MagicMock()
            mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(side_effect=[[[1.0]], [[2.0]]])
            MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider
            # Fix: Make db_manager.ainit awaitable
            MockVectorStoreManager.return_value.ainit = AsyncMock()
            pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
            pipeline.db_manager = MockVectorStoreManager.return_value
            asyncio.run(pipeline.index("dummy/directory"))
            mock_doc_loader.load_documents.assert_called_once_with("dummy/directory")
            self.assertEqual(mock_text_processor.process_documents.call_count, 2)
            self.assertEqual(mock_embedding_provider.get_embedding_model().get_text_embedding_batch.await_count, 2)
            self.assertEqual(pipeline.db_manager.upsert_documents.call_count, 2)
            # Check both chunks
            calls = pipeline.db_manager.upsert_documents.call_args_list
            self.assertEqual(calls[0].kwargs['chunks'][0].embedding, [1.0])
            self.assertEqual(calls[1].kwargs['chunks'][0].embedding, [2.0])

    @patch("yasrl.pipeline.ConfigurationManager") 
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
        MockConfigManager,
    ):
        mock_config = MockConfigManager.return_value
        mock_config.load_config.return_value = MagicMock(
            database=MagicMock(postgres_uri="postgres://test:test@localhost/db")
        )
        mock_config.openai_api_key = "test-key"
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "POSTGRES_URI": "postgres://test:test@localhost/db"
        }, clear=True):
            mock_doc_loader = MockDocumentLoader.return_value
            mock_doc_loader.load_documents.return_value = [
                Document(id_="doc1", text="content1")
            ]
            mock_text_processor = MockTextProcessor.return_value
            mock_chunk = MagicMock(text="chunk1")
            def set_embedding_side_effect(val):
                mock_chunk.embedding = val
            mock_chunk.set_embedding.side_effect = set_embedding_side_effect
            mock_text_processor.process_documents.return_value = [mock_chunk]
            mock_embedding_provider = MagicMock()
            mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(return_value=[[1.0]])
            MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider
            # Fix: Make db_manager.ainit awaitable
            MockVectorStoreManager.return_value.ainit = AsyncMock()
            pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
            pipeline.db_manager = MockVectorStoreManager.return_value
            asyncio.run(pipeline.index("http://example.com"))
            mock_doc_loader.load_documents.assert_called_once_with("http://example.com")
            mock_text_processor.process_documents.assert_called_once()
            mock_embedding_provider.get_embedding_model().get_text_embedding_batch.assert_called_once()
            pipeline.db_manager.upsert_documents.assert_called_once()
            call_args = pipeline.db_manager.upsert_documents.call_args
            chunk = call_args.kwargs['chunks'][0]
            self.assertEqual(chunk.text, "chunk1")
            self.assertEqual(chunk.embedding, [1.0])
    
    
    @patch("yasrl.pipeline.ConfigurationManager")
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
        MockConfigManager,
    ):
        mock_config = MockConfigManager.return_value
        mock_config.load_config.return_value = MagicMock(
            database=MagicMock(postgres_uri="postgres://test:test@localhost/db")
        )
        mock_config.openai_api_key = "test-key"
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "POSTGRES_URI": "postgres://test:test@localhost/db"
        }, clear=True):
            mock_doc_loader = MockDocumentLoader.return_value
            mock_doc_loader.load_documents.return_value = [
                Document(id_="doc1", text="content1")
            ]
            mock_text_processor = MockTextProcessor.return_value
            mock_chunk = MagicMock(text="chunk1")
            def set_embedding_side_effect(val):
                mock_chunk.embedding = val
            mock_chunk.set_embedding.side_effect = set_embedding_side_effect
            mock_text_processor.process_documents.return_value = [mock_chunk]
            mock_embedding_provider = MagicMock()
            mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(return_value=[[1.0]])
            MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider
            # Fix: Make db_manager.ainit awaitable
            MockVectorStoreManager.return_value.ainit = AsyncMock()
            pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
            pipeline.db_manager = MockVectorStoreManager.return_value
            asyncio.run(pipeline.index("dummy/path/to/file.txt"))
            asyncio.run(pipeline.index("dummy/path/to/file.txt"))
            self.assertEqual(mock_doc_loader.load_documents.call_count, 2)
            self.assertEqual(pipeline.db_manager.upsert_documents.call_count, 2)
            pipeline.db_manager.upsert_documents.assert_called()
            last_call = pipeline.db_manager.upsert_documents.call_args
            chunk = last_call.kwargs['chunks'][0]
            self.assertEqual(chunk.text, "chunk1")
            self.assertEqual(chunk.embedding, [1.0])


    @patch("yasrl.pipeline.ConfigurationManager")
    @patch("yasrl.pipeline.DocumentLoader")
    @patch("yasrl.pipeline.EmbeddingProviderFactory")
    @patch("yasrl.pipeline.VectorStoreManager")
    @patch("yasrl.pipeline.TextProcessor")
    def test_multiple_chunks_per_document(
        self,
        MockTextProcessor,
        MockVectorStoreManager,
        MockEmbeddingProviderFactory,
        MockDocumentLoader,
        MockConfigManager,
    ):
        mock_config = MockConfigManager.return_value
        mock_config.load_config.return_value = MagicMock(
            database=MagicMock(postgres_uri="postgres://test:test@localhost/db")
        )
        mock_config.openai_api_key = "test-key"
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "POSTGRES_URI": "postgres://test:test@localhost/db"
        }, clear=True):
            mock_doc_loader = MockDocumentLoader.return_value
            mock_doc_loader.load_documents.return_value = [
                Document(id_="doc1", text="long content that will be split")
            ]
            mock_text_processor = MockTextProcessor.return_value
            chunks = []
            for text in ["chunk1", "chunk2", "chunk3"]:
                chunk = MagicMock(text=text)
                chunk.set_embedding.side_effect = lambda val, c=chunk: setattr(c, "embedding", val)
                chunks.append(chunk)
            mock_text_processor.process_documents.return_value = chunks
            mock_embedding_provider = MagicMock()
            mock_embedding_provider.get_embedding_model.return_value.get_text_embedding_batch = AsyncMock(
                return_value=[[1.0], [2.0], [3.0]]
            )
            MockEmbeddingProviderFactory.create_provider.return_value = mock_embedding_provider
            # Fix: Make db_manager.ainit awaitable
            MockVectorStoreManager.return_value.ainit = AsyncMock()
            pipeline = asyncio.run(RAGPipeline.create(llm="openai", embed_model="openai"))
            pipeline.db_manager = MockVectorStoreManager.return_value
            asyncio.run(pipeline.index("dummy/path/to/large_file.txt"))
            mock_doc_loader.load_documents.assert_called_once_with("dummy/path/to/large_file.txt")
            mock_text_processor.process_documents.assert_called_once()
            mock_embedding_provider.get_embedding_model().get_text_embedding_batch.assert_called_once()
            pipeline.db_manager.upsert_documents.assert_called_once()
            call_args = pipeline.db_manager.upsert_documents.call_args
            self.assertEqual(call_args.kwargs['document_id'], 'doc1')
            self.assertEqual(len(call_args.kwargs['chunks']), 3)
            expected_texts = ["chunk1", "chunk2", "chunk3"]
            expected_embeddings = [[1.0], [2.0], [3.0]]
            for i, chunk in enumerate(call_args.kwargs['chunks']):
                self.assertEqual(chunk.text, expected_texts[i])
                self.assertEqual(chunk.embedding, expected_embeddings[i])


if __name__ == "__main__":
    unittest.main()