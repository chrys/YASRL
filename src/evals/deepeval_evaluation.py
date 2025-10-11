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
from UI.project_manager import get_project_manager

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel

async def run_deepeval_evaluation():
    print("🔬 DeepEval RAG Evaluation Demo with Gemini")
    print("=" * 50)

    try:
        project_manager = get_project_manager()
    except Exception as exc:
        print(f"❌ Unable to initialize ProjectManager: {exc}")
        return

    project_records = project_manager.list_projects()
    if not project_records:
        print("❌ No projects found in database.")
        return

    # Prompt user to select a project
    print("Available projects:")
    project_choices = []
    for idx, record in enumerate(project_records, start=1):
        display = f"{record['name']} | {record['id'][:8]}"
        project_choices.append((display, record))
        print(f"  {idx}. {display}")

    while True:
        try:
            selection = int(input(f"Select a project [1-{len(project_choices)}]: "))
            if 1 <= selection <= len(project_choices):
                break
            else:
                print("Invalid selection. Try again.")
        except Exception:
            print("Please enter a valid number.")

    selected_display, selected_project = project_choices[selection - 1]
    selected_pid = selected_project["id"]
    print(f"\n✅ Selected project: {selected_display}")

    # Load evaluation dataset from CSV
    import csv
    csv_path = Path(os.getenv("EVAL_DATASET_FILE", "../../data/happy_payments.csv"))
    if not csv_path.exists():
        print(f"❌ Could not find evaluation dataset CSV at {csv_path}")
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
        print("1. Initializing RAG Pipeline...")
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
        print("✅ Pipeline initialized")

        # 2. Configure DeepEval to use Gemini for evaluation
        print("\n2. Configuring DeepEval with Gemini model...")
        # This tells DeepEval's metrics to use Gemini for judging the results
        #get the Gemini API key from .env
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")
        else:
            gemini_eval_model = GeminiModel(
                model_name="gemini-2.5-flash",
                api_key=gemini_api_key,
            )
        print("✅ DeepEval configured")

        # 3. Run pipeline and create DeepEval Test Cases
        print("\n3. Running RAG pipeline and creating test cases...")
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
            print(f"  - Created test case for: '{question}'")

        # 4. Define metrics and run evaluation
        print("\n4. Evaluating with DeepEval metrics...")
        metrics = [
            # Is the generated answer relevant to the input question? (Score 0-1)
            AnswerRelevancyMetric(model=gemini_eval_model, threshold=0.7),
            
            # Does the answer stick to the facts in the retrieved context? (Score 0-1, 1 means no hallucination)
            FaithfulnessMetric(model=gemini_eval_model, threshold=0.7),
            
            # Did the retriever find all the relevant information from the ground truth to answer the question? (Score 0-1)
            ContextualPrecisionMetric(model=gemini_eval_model, threshold=0.8),
            
            # Is the retrieved context relevant to the question? (High precision means less "noise" was retrieved). (Score 0-1)
            ContextualRecallMetric(model=gemini_eval_model, threshold=0.8)
        ]

        # The evaluate function runs the metrics against the test cases
        results = evaluate(test_cases, metrics)

        print("\n📊 DeepEval Evaluation Results:")
        # The results object contains detailed information for each test case
        # For a summary, we can calculate the average scores
        total_scores = {}
        for metric in metrics:
            total_scores[metric.__name__] = 0

        for result in results.test_results:
            if not result.metrics_data:
                continue
            for metric_result in result.metrics_data:
                total_scores.setdefault(metric_result.__class__.__name__, 0.0)
                total_scores[metric_result.__class__.__name__] += metric_result.score

        print("Average Scores:")
        # Check if results were generated before dividing
        if results.test_results:
            for name, total_score in total_scores.items():
                avg_score = total_score / len(results.test_results)
                print(f"  - {name}: {avg_score:.3f}")
        else:
            print("  - No results to average.")

        # Save results to file
        output_file = Path("./results") / f"deepeval_evaluation_results_{sanitized_name or selected_pid}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # DeepEval results are not directly JSON serializable, so we format them
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

        print(f"\n💾 DeepEval evaluation results saved to: {output_file}")

        await pipeline.cleanup()

    except Exception as e:
        print(f"❌ Error during DeepEval evaluation: {e}")
        raise

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(run_deepeval_evaluation())