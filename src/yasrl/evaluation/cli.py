import argparse
import json
import logging
import os
import csv
from typing import Any, Dict, List, Optional
from yasrl.pipeline import RAGPipeline
from yasrl.evaluation.base import BaseEvaluator
from yasrl.config.manager import ConfigManager
from yasrl.loaders import load_documents
from yasrl.evaluation.ragas_evaluator import RagasEvaluator
from yasrl.providers.llm import LLMProvider
from yasrl.providers.embeddings import EmbeddingProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function for the YASRL evaluation CLI."""
    parser = argparse.ArgumentParser(description="YASRL Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand for running evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on a dataset.")
    eval_parser.add_argument("--dataset", type=str, required=True, help="Path to the evaluation dataset (JSON or CSV).")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to the pipeline configuration file.")
    eval_parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the evaluation results.")
    eval_parser.add_argument("--output-formats", nargs="+", default=["json", "html"], help="Formats to export results (json, csv, html).")

    # Subcommand for generating synthetic questions
    synth_parser = subparsers.add_parser("generate-synthetic", help="Generate synthetic evaluation questions.")
    synth_parser.add_argument("--docs-path", type=str, required=True, help="Path to the documents to generate questions from.")
    synth_parser.add_argument("--num-questions", type=int, default=10, help="Number of synthetic questions to generate.")
    synth_parser.add_argument("--output-file", type=str, required=True, help="Path to save the synthetic dataset.")

    # Subcommand for comparing pipelines
    compare_parser = subparsers.add_parser("compare-pipelines", help="Compare multiple pipeline configurations.")
    compare_parser.add_argument("--dataset", type=str, required=True, help="Path to the evaluation dataset.")
    compare_parser.add_argument("--configs", nargs="+", required=True, help="Paths to the pipeline configuration files to compare.")
    compare_parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the comparison report.")

    # Subcommand for converting datasets
    convert_parser = subparsers.add_parser("convert-dataset", help="Convert a dataset from one format to another.")
    convert_parser.add_argument("--input-file", type=str, required=True, help="Path to the input dataset file.")
    convert_parser.add_argument("--output-file", type=str, required=True, help="Path to the output dataset file.")

    args = parser.parse_args()

    if args.command == "evaluate":
        run_evaluation(args.dataset, args.config, args.output_dir, args.output_formats)
    elif args.command == "generate-synthetic":
        generate_synthetic_questions(args.docs_path, args.num_questions, args.output_file)
    elif args.command == "compare-pipelines":
        compare_pipelines(args.dataset, args.configs, args.output_dir)
    elif args.command == "convert-dataset":
        convert_dataset(args.input_file, args.output_file)

def validate_dataset(dataset: List[Dict[str, Any]]):
    """Validate that the dataset has the required fields."""
    for i, item in enumerate(dataset):
        if "question" not in item or "expected_answer" not in item:
            raise ValueError(f"Dataset item at index {i} is missing 'question' or 'expected_answer' field.")

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate a dataset from a JSON or CSV file."""
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            dataset = json.load(f)
    elif file_path.endswith(".csv"):
        with open(file_path, "r") as f:
            dataset = list(csv.DictReader(f))
    else:
        raise ValueError("Unsupported dataset format. Please use JSON or CSV.")

    validate_dataset(dataset)
    return dataset

def convert_dataset(input_file: str, output_file: str):
    """Convert a dataset from one format to another."""
    dataset = load_dataset(input_file)
    if output_file.endswith(".json"):
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=4)
    elif output_file.endswith(".csv"):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dataset[0].keys())
            writer.writeheader()
            writer.writerows(dataset)
    else:
        raise ValueError("Unsupported output format. Please use JSON or CSV.")
    logging.info(f"Dataset converted and saved to {output_file}")

def save_report(results: Dict[str, Any], output_dir: str, format: str):
    """Save the evaluation report in the specified format."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"report.{format}")

    if format == "json":
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
    elif format == "csv":
        per_question_results = results.get("per_question_results", [])
        if per_question_results:
            with open(file_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_question_results[0].keys())
                writer.writeheader()
                writer.writerows(per_question_results)
    elif format == "html":
        # Basic HTML report
        html = "<html><head><title>Evaluation Report</title></head><body>"
        html += "<h1>Evaluation Report</h1>"
        html += f"<h2>Overall Scores</h2>"
        html += "<ul>"
        for metric, score in results.get("overall_scores", {}).items():
            html += f"<li>{metric.capitalize()}: {score:.4f}</li>"
        html += "</ul>"
        html += "<h2>Per-Question Results</h2>"
        html += "<table border='1'><tr>"
        if results.get("per_question_results"):
            headers = results["per_question_results"][0].keys()
            for header in headers:
                html += f"<th>{header}</th>"
            html += "</tr>"
            for result in results["per_question_results"]:
                html += "<tr>"
                for header in headers:
                    html += f"<td>{result.get(header, '')}</td>"
                html += "</tr>"
        html += "</table></body></html>"
        with open(file_path, "w") as f:
            f.write(html)

    logging.info(f"Report saved to {file_path}")

def run_evaluation(dataset_path: str, config_path: str, output_dir: str, output_formats: List[str]):
    """Run evaluation on a dataset using a given pipeline configuration."""
    logging.info(f"Running evaluation with dataset '{dataset_path}' and config '{config_path}'")

    # Load dataset
    dataset = load_dataset(dataset_path)
    questions = [item["question"] for item in dataset]
    expected_answers = [item["expected_answer"] for item in dataset]

    # Initialize pipeline
    config_manager = ConfigManager(config_path)
    pipeline_config = config_manager.get_pipeline_config()
    llm_provider = LLMProvider(config=pipeline_config.llm)
    embedding_provider = EmbeddingProvider(config=pipeline_config.embedding)
    pipeline = RAGPipeline(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        vector_store=None,  # Not needed for this evaluation
        config=pipeline_config,
    )

    # Initialize evaluator
    evaluator = RagasEvaluator(llm_provider, embedding_provider)

    # Run evaluation
    results = evaluator.evaluate(pipeline, questions, expected_answers)

    # Save reports
    for format in output_formats:
        save_report(results, output_dir, format)

    logging.info("Evaluation finished.")

def generate_synthetic_questions(docs_path: str, num_questions: int, output_file: str):
    """Generate synthetic evaluation questions from documents."""
    logging.info(f"Generating {num_questions} synthetic questions from documents in '{docs_path}'")

    # Load documents
    documents = load_documents(docs_path)
    if not documents:
        logging.error("No documents found. Cannot generate questions.")
        return

    # For demonstration purposes, we'll just take the first `num_questions` documents
    # and generate dummy questions. A real implementation would use an LLM.
    synthetic_dataset = []
    for i, doc in enumerate(documents[:num_questions]):
        synthetic_dataset.append({
            "question": f"What is this document about: {doc.text[:50]}...",
            "expected_answer": "This is a synthetically generated answer.",
            "context": doc.text,
        })

    # Save the synthetic dataset
    with open(output_file, "w") as f:
        json.dump(synthetic_dataset, f, indent=4)

    logging.info(f"Synthetic dataset saved to {output_file}")

def compare_pipelines(dataset_path: str, config_paths: List[str], output_dir: str):
    """Compare multiple pipeline configurations."""
    logging.info(f"Comparing {len(config_paths)} pipelines using dataset '{dataset_path}'")

    comparison_results = {}
    for config_path in config_paths:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        logging.info(f"Evaluating pipeline: {config_name}")

        # Load dataset
        dataset = load_dataset(dataset_path)
        questions = [item["question"] for item in dataset]
        expected_answers = [item["expected_answer"] for item in dataset]

        # Initialize pipeline
        config_manager = ConfigManager(config_path)
        pipeline_config = config_manager.get_pipeline_config()
        llm_provider = LLMProvider(config=pipeline_config.llm)
        embedding_provider = EmbeddingProvider(config=pipeline_config.embedding)
        pipeline = RAGPipeline(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            vector_store=None,  # Not needed for this evaluation
            config=pipeline_config,
        )

        # Initialize evaluator
        evaluator = RagasEvaluator(llm_provider, embedding_provider)

        # Run evaluation
        results = evaluator.evaluate(pipeline, questions, expected_answers)
        comparison_results[config_name] = results

    # Save comparison report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(comparison_results, f, indent=4)

    logging.info(f"Comparison report saved to {report_path}")


if __name__ == "__main__":
    main()
