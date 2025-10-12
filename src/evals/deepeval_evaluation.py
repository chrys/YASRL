"""
DeepEval RAG Evaluation Demo

This script demonstrates how to evaluate a RAG pipeline using the DeepEval framework
with a Gemini model for evaluation.


"""

import asyncio
import os
import json
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from yasrl.pipeline import RAGPipeline

import logging
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

from .deepeval_gemini import Gemini as DeepEvalGemini

logger = logging.getLogger(__name__)

from llama_index.core import Document

async def generate_evaluation_dataset(
    documents: list[Document],
    model: str = "gemini-1.5-flash",
    max_questions: int = 10,
) -> EvaluationDataset:
    """
    Generates an evaluation dataset (QA pairs) from a list of documents
    using DeepEval's Synthesizer and a Gemini model.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    # Use the custom Gemini wrapper for the synthesizer
    generator_model = DeepEvalGemini(model=model, api_key=api_key)

    synthesizer = Synthesizer(
        model=generator_model,
        # You can customize the synthesizer with different prompts if needed
        # See DeepEval documentation for more details
    )

    logger.info(f"Generating up to {max_questions} questions from {len(documents)} documents...")

    # Generate the dataset
    dataset = EvaluationDataset()
    dataset.generate(
        documents=documents,
        max_questions=max_questions,
        synthesizer=synthesizer,
    )

    logger.info(f"Successfully generated {len(dataset.test_cases)} QA pairs.")
    return dataset


async def run_deepeval_evaluation():
    logger.info("🔬 DeepEval RAG Evaluation Demo with Gemini")
    logger.info("=" * 50)

    # Load projects from projects.json
    projects_path = Path(os.getenv("PROJECTS_FILE", "projects.json"))
    if not projects_path.exists():
        logger.error(f"Could not find projects file at {projects_path}")
        return

    with open(projects_path, "r") as f:
        projects_data = json.load(f)

    # Prompt user to select a project
    logger.info("Available projects:")
    project_choices = []
    for pid, info in projects_data.items():
        display = f"{info['name']} | {pid[:8]}"
        project_choices.append((display, pid))
        logger.info(f"  {len(project_choices)}. {display}")

    while True:
        try:
            selection = int(input(f"Select a project [1-{len(project_choices)}]: "))
            if 1 <= selection <= len(project_choices):
                break
            else:
                logger.warning("Invalid selection. Try again.")
        except Exception:
            logger.warning("Please enter a valid number.")

    selected_display, selected_pid = project_choices[selection - 1]
    selected_project = projects_data[selected_pid]
    logger.info(f"✅ Selected project: {selected_display}")

    # Load evaluation dataset from CSV
    import csv
    csv_path = Path(os.getenv("EVAL_DATASET_FILE", "../../data/happy_payments.csv"))
    if not csv_path.exists():
        logger.error(f"Could not find evaluation dataset CSV at {csv_path}")
        return

    eval_dataset = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Normalize keys to match expected structure
            eval_dataset.append({
                "question": row.get("question", "").strip(),
                "expected_answer": row.get("expected_answer", "").strip(),
                "ground_truth": row.get("ground_truth", "").strip(),
            })

    try:
        # 1. Initialize the RAG pipeline
        logger.info("1. Initializing RAG Pipeline...")
        from yasrl.vector_store import VectorStoreManager
        project_name = selected_project.get("name", "").strip()
        sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
        table_prefix = f"yasrl_{sanitized_name or selected_pid}"
        db_manager = VectorStoreManager(
            postgres_uri=os.getenv("POSTGRES_URI") or "",
            vector_dimensions=768,
            table_prefix=table_prefix
        )
        pipeline = await RAGPipeline.create(
            llm=selected_project.get("llm", "gemini"),
            embed_model=selected_project.get("embed_model", "gemini"),
            db_manager=db_manager
        )
        logger.info("✅ Pipeline initialized")

        # 2. Configure DeepEval to use Gemini for evaluation
        logger.info("Configuring DeepEval with Gemini model...")
        # This tells DeepEval's metrics to use Gemini for judging the results
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")

        gemini_eval_model = DeepEvalGemini(
            model_name="gemini-1.5-flash",
            api_key=gemini_api_key,
        )
        logger.info("✅ DeepEval configured")

        # 3. Run pipeline and create DeepEval Test Cases
        logger.info("Running RAG pipeline and creating test cases...")
        test_cases = []
        for item in eval_dataset:
            question = item["question"]
            expected_answer = item["expected_answer"]
            result = await pipeline.ask(question)
            contexts = [chunk.text for chunk in result.source_chunks]
            ground_truth_context = item["ground_truth"]
            
            test_case = LLMTestCase(
                input=question,
                actual_output=result.answer or "",
                expected_output = expected_answer, 
                retrieval_context=contexts, 
                context=[ground_truth_context]  # Provide ground truth as context for faithfulness check
            )
            test_cases.append(test_case)
            logger.info(f"  - Created test case for: '{question}'")

        # 4. Define metrics and run evaluation
        logger.info("Evaluating with DeepEval metrics...")
        metrics = [
            AnswerRelevancyMetric(model=gemini_eval_model, threshold=0.7),
            FaithfulnessMetric(model=gemini_eval_model, threshold=0.7),
            ContextualPrecisionMetric(model=gemini_eval_model, threshold=0.8),
            ContextualRecallMetric(model=gemini_eval_model, threshold=0.8)
        ]

        results = evaluate(test_cases, metrics)

        logger.info("📊 DeepEval Evaluation Results:")
        total_scores = {}
        for metric in metrics:
            total_scores[metric.__name__] = 0

        for result in results.test_results:
            if not result.metrics_data:
                continue
            for metric_result in result.metrics_data:
                total_scores.setdefault(metric_result.__class__.__name__, 0.0)
                total_scores[metric_result.__class__.__name__] += metric_result.score

        logger.info("Average Scores:")
        if results.test_results:
            for name, total_score in total_scores.items():
                avg_score = total_score / len(results.test_results)
                logger.info(f"  - {name}: {avg_score:.3f}")
        else:
            logger.info("  - No results to average.")

        output_file = Path("./results") / f"deepeval_evaluation_results_{sanitized_name or selected_pid}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_results = []
        for result in results.test_results:
            metric_entries = []
            if result.metrics_data:
                for m in result.metrics_data:
                    metric_entries.append({
                        "name": m.name,
                        "score": m.score,
                        "reason": m.reason,
                    })
            serializable_results.append({
                "input": result.input,
                "actual_output": result.actual_output,
                "metrics": metric_entries,
            })

        with open(output_file, "w") as f:
            json.dump({
                "evaluation_type": "deepeval",
                "project": selected_display,
                "results": serializable_results
            }, f, indent=2)

        logger.info(f"💾 DeepEval evaluation results saved to: {output_file}")

        await pipeline.cleanup()

    except Exception as e:
        logger.exception(f"Error during DeepEval evaluation: {e}")
        raise

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(run_deepeval_evaluation())