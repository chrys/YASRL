import unittest
import os
import json
import csv
from unittest.mock import patch, MagicMock
from yasrl.evaluation.cli import main

class TestEvaluationCLI(unittest.TestCase):
    def setUp(self):
        # Create a dummy config file
        self.config_path = "test_config.yaml"
        with open(self.config_path, "w") as f:
            f.write("""
pipeline:
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
  embedding:
    provider: "openai"
    model: "text-embedding-ada-002"
""")

        # Create a dummy dataset
        self.dataset_path = "test_dataset.json"
        self.dataset = [
            {"question": "What is the capital of France?", "expected_answer": "Paris"},
            {"question": "Who wrote Hamlet?", "expected_answer": "Shakespeare"},
        ]
        with open(self.dataset_path, "w") as f:
            json.dump(self.dataset, f)

    def tearDown(self):
        os.remove(self.config_path)
        os.remove(self.dataset_path)
        if os.path.exists("report.json"):
            os.remove("report.json")
        if os.path.exists("report.html"):
            os.remove("report.html")
        if os.path.exists("report.csv"):
            os.remove("report.csv")
        if os.path.exists("synthetic_dataset.json"):
            os.remove("synthetic_dataset.json")
        if os.path.exists("comparison_report.json"):
            os.remove("comparison_report.json")

    
    @patch("yasrl.loaders.DocumentLoader.load_documents")
    def test_generate_synthetic_command(self, mock_load_documents):
        # Mock the document loader
        mock_load_documents.return_value = [
            MagicMock(text="doc1"),
            MagicMock(text="doc2"),
            MagicMock(text="doc3"),
        ]

        # Run the command
        output_file = "synthetic_dataset.json"
        with patch("sys.argv", ["cli.py", "generate-synthetic", "--docs-path", "dummy_path", "--num-questions", "2", "--output-file", output_file]):
            main()

        # Check that the output file was created and has the correct content
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertIn("question", data[0])
        self.assertIn("expected_answer", data[0])


    def test_dataset_conversion(self):
        # Create a CSV dataset
        csv_path = "test_dataset.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "expected_answer"])
            writer.writeheader()
            writer.writerows(self.dataset)

        # Run the conversion command
        output_path = "converted_dataset.json"
        with patch("sys.argv", ["cli.py", "convert-dataset", "--input-file", csv_path, "--output-file", output_path]):
            main()

        # Check that the converted file was created and has the correct content
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["question"], "What is the capital of France?")

        os.remove(csv_path)
        os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
