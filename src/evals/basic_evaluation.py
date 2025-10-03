"""
Basic RAG Evaluation Demo

This script demonstrates how to evaluate a RAG pipeline using simple metrics
like answer relevance and source attribution. It shows the fundamental
evaluation workflow with a small dataset.
"""

import asyncio
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional 

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from yasrl.pipeline import RAGPipeline
from yasrl.models import QueryResult

class BasicEvaluator:
    """
    Simple evaluator that checks basic RAG quality metrics
    """
    
    def __init__(self):
        self.metrics = ["answer_length", "has_sources", "source_count"]
    
    def evaluate_answer(self, question: str, result: QueryResult, expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single answer with basic metrics
        
        Args:
            question: The question that was asked
            result: The QueryResult from the RAG pipeline
            expected_answer: Optional ground truth answer
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        answer_text = result.answer or ""
        evaluation = {
            "question": question,
            "answer": answer_text,
            "metrics": {}
        }
        
        # Metric 1: Answer length (should be substantial but not too long)
        answer_length = len(answer_text.split())
        evaluation["metrics"]["answer_length"] = answer_length
        evaluation["metrics"]["good_length"] = 10 <= answer_length <= 200
        
        # Metric 2: Has sources (RAG should always provide sources)
        has_sources = len(result.source_chunks) > 0
        evaluation["metrics"]["has_sources"] = has_sources
        
        # Metric 3: Source count (multiple sources indicate good retrieval)
        source_count = len(result.source_chunks)
        evaluation["metrics"]["source_count"] = source_count
        
        # Metric 4: Answer relevance (simple keyword matching)
        if expected_answer:
            answer_lower = answer_text.lower()
            expected_lower = expected_answer.lower()
            # Simple relevance check: count common words
            answer_words = set(answer_lower.split())
            expected_words = set(expected_lower.split())
            common_words = answer_words.intersection(expected_words)
            relevance_score = len(common_words) / max(len(expected_words), 1)
            evaluation["metrics"]["keyword_relevance"] = relevance_score
        
        # Overall quality score (simple average)
        quality_score = (
            int(evaluation["metrics"]["good_length"]) +
            int(evaluation["metrics"]["has_sources"]) +
            min(evaluation["metrics"]["source_count"] / 2, 1)  # Cap at 1
        ) / 3
        evaluation["metrics"]["overall_quality"] = quality_score
        
        return evaluation

async def run_basic_evaluation():
    """
    Main evaluation function that demonstrates basic RAG evaluation
    """
    print("🔍 Basic RAG Evaluation Demo")
    print("=" * 50)
    
        # Load projects from projects.json
    projects_path = Path(os.getenv("PROJECTS_FILE", "projects.json"))
    if not projects_path.exists():
        print(f"❌ Could not find projects file at {projects_path}")
        return

    with open(projects_path, "r") as f:
        projects_data = json.load(f)

    if not projects_data:
        print("❌ No projects found in projects.json.")
        return

    # List available projects
    print("Available projects:")
    project_choices = []
    for pid, info in projects_data.items():
        display = f"{info['name']} | {pid[:8]}"
        project_choices.append((display, pid))
        print(f"  {len(project_choices)}. {display}")

    # Prompt user to select a project
    while True:
        try:
            selection = int(input(f"Select a project [1-{len(project_choices)}]: "))
            if 1 <= selection <= len(project_choices):
                break
            else:
                print("Invalid selection. Try again.")
        except Exception:
            print("Please enter a valid number.")

    selected_display, selected_pid = project_choices[selection - 1]
    selected_project = projects_data[selected_pid]
    print(f"\n✅ Selected project: {selected_display}")
    
    # Create evaluation dataset
    eval_dataset = [
        {
            "question": "What is Happy Payments?",
            "expected_answer": "Happy Payments is a company that processes payments"
        },
        {
            "question": "What are the benefits of ISO 8583?",
            "expected_answer": "ISO 8583 provides standardized messaging for financial transactions"
        },
        {
            "question": "How does payment processing work?",
            "expected_answer": "Payment processing involves authorization, clearing, and settlement"
        }
    ]
    
    try:
        # Initialize the RAG pipeline for the selected project
        print("1. Initializing RAG Pipeline for selected project...")
        from yasrl.vector_store import VectorStoreManager

        # Use the same table naming logic as in the UI/app
        project_name = selected_project.get("name", "").strip()
        sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
        sanitized_name = "_".join(filter(None, sanitized_name.split("_")))
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
        
        # Initialize evaluator
        evaluator = BasicEvaluator()
        
        # Run evaluation on each question
        print("\n2. Running Evaluation...")
        results = []
        
        for i, item in enumerate(eval_dataset, 1):
            question = item["question"]
            expected = item["expected_answer"]
            
            print(f"\n📝 Question {i}: {question}")
            
            # Get answer from RAG pipeline
            result = await pipeline.ask(question)
            
            # Evaluate the result
            evaluation = evaluator.evaluate_answer(question, result, expected)
            results.append(evaluation)
            
            # Print evaluation results
            print(f"💡 Answer: {evaluation['answer'][:100]}...")
            print(f"📊 Metrics:")
            for metric, value in evaluation["metrics"].items():
                print(f"   - {metric}: {value}")
        
        # Calculate overall statistics
        print("\n3. Overall Evaluation Results")
        print("-" * 30)
        
        total_questions = len(results)
        avg_quality = sum(r["metrics"]["overall_quality"] for r in results) / total_questions
        sources_provided = sum(1 for r in results if r["metrics"]["has_sources"]) / total_questions
        good_length_ratio = sum(1 for r in results if r["metrics"]["good_length"]) / total_questions
        
        print(f"📈 Total Questions Evaluated: {total_questions}")
        print(f"📈 Average Quality Score: {avg_quality:.2f}")
        print(f"📈 Questions with Sources: {sources_provided:.1%}")
        print(f"📈 Good Answer Length: {good_length_ratio:.1%}")
        
        # Save results to file
        
        output_file = Path("./results") / "basic_evaluation_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump({
                "evaluation_type": "basic",
                "project": selected_display,
                "total_questions": total_questions,
                "overall_metrics": {
                    "average_quality": avg_quality,
                    "sources_provided_ratio": sources_provided,
                    "good_length_ratio": good_length_ratio
                },
                "detailed_results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Cleanup
        await pipeline.cleanup()
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the evaluation
    asyncio.run(run_basic_evaluation())