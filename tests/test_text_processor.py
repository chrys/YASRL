import unittest
from unittest.mock import MagicMock, patch

from llama_index.core.schema import Document

from yasrl.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    def test_create_nodes_from_text(self):
        text_processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        text = "This is a test sentence. This is another test sentence."
        metadata = {"source": "test"}
        nodes = text_processor.create_nodes_from_text(text, metadata)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].text, text)
        self.assertEqual(nodes[0].metadata["source"], "test")
        self.assertIn("created_at", nodes[0].metadata)

    def test_process_documents(self):
        text_processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        documents = [
            Document(
                text="This is a test sentence. This is another test sentence.",
                metadata={"source": "test1"},
            ),
            Document(
                text="This is a second document.",
                metadata={"source": "test2"},
            ),
        ]
        nodes = text_processor.process_documents(documents)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].metadata["source"], "test1")
        self.assertEqual(nodes[1].metadata["source"], "test2")

    @patch("yasrl.text_processor.EmbeddingProviderFactory")
    def test_optimize_chunk_size(self, mock_embedding_provider_factory):
        mock_provider = MagicMock()
        mock_provider.get_max_chunk_size.return_value = 512
        mock_embedding_provider_factory.get_embedding_provider.return_value = (
            mock_provider
        )

        # Test with a specified chunk size
        self.assertEqual(TextProcessor.optimize_chunk_size("test", chunk_size=1024), 1024)

        # Test without a specified chunk size
        self.assertEqual(TextProcessor.optimize_chunk_size("test"), 512)
        mock_embedding_provider_factory.get_embedding_provider.assert_called_with("test")
        mock_provider.get_max_chunk_size.assert_called_once()


if __name__ == "__main__":
    unittest.main()
