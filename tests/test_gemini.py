import unittest
from unittest.mock import MagicMock, patch
from yasrl.evaluation.cli import run_evaluation

class TestGeminiIntegration(unittest.TestCase):
    @patch("yasrl.evaluation.cli.GeminiLLMProvider")
    @patch("yasrl.evaluation.cli.GeminiEmbeddingProvider")
    @patch("yasrl.evaluation.cli.RAGPipeline")
    @patch("yasrl.evaluation.cli.load_dataset")
    @patch("yasrl.evaluation.cli.ConfigurationManager")
    def test_run_evaluation_with_gemini(
        self, mock_config_manager, mock_load_dataset, mock_rag_pipeline, mock_gemini_embed, mock_gemini_llm
    ):
        # Arrange
        mock_load_dataset.return_value = [
            {"question": "What is Gemini?", "expected_answer": "A Google LLM"}
        ]
        mock_config = MagicMock()
        mock_config.llm.model_name = "gemini-pro"
        mock_config.embedding.model_name = "gemini-embed"
        mock_config_manager.return_value.load_config.return_value = mock_config

        # Act
        run_evaluation("dummy_dataset.json", "dummy_config.yaml", ".", ["json"])

        # Assert
        mock_gemini_llm.assert_called_once()
        mock_gemini_embed.assert_called_once()
        mock_rag_pipeline.assert_called_once_with(
            llm="gemini-pro", embed_model="gemini-embed"
        )

if __name__ == "__main__":
    unittest.main()