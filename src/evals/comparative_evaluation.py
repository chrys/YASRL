"""
Comparative RAG Evaluation Demo

This script demonstrates how to compare different RAG configurations
(different models, chunk sizes, etc.) to find the best performing setup.
It compares multiple pipeline configurations side by side.
"""

import asyncio
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from yasrl.pipeline import RAGPipeline
from yasrl.models import QueryResult

class ComparativeEvaluator:
    """
    Evaluator that compares multiple RAG configurations
    """
    
    def __init__(self):
        self.metrics = [
            "response_time", 
            "answer_length", 
            "source_count", 
            "answer_completeness"
        ]
    
    def evaluate_response(self, question: str, result: QueryResult, response_time: float) -> Dict[str, Any]:
        """
        Evaluate a single response with comparative metrics
        
        Args:
            question: The question asked
            result: The QueryResult from the pipeline
            response_time: Time taken to generate the response
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Response time metric (lower is better)
        time_score = max(0, 1 - (response_time / 30))  # Penalize responses > 30 seconds
        
        # Answer length metric (should be substantial)
        answer_length = len(result.answer.split())
        length_score = min(answer_length / 50, 1)  # Optimal around 50 words
        
        # Source count metric (more sources = better retrieval)
        source_count = len(result.source_chunks)
        source_score = min(source_count / 3, 1)  # Optimal around 3 sources
        
        # Answer completeness (checks for key indicators)
        completeness_indicators = [
            "." in result.answer,  # Has proper sentences
            len(result.answer) > 20,  # Substantial content
            not result.answer.lower().startswith("i don't"),  # Not a "don't know" response
            not result.answer.lower().startswith("sorry"),  # Not an apology
        ]
        completeness_score = sum(completeness_indicators) / len(completeness_indicators)
        
        return {
            "response_time": response_time,
            "response_time_score": time_score,
            "answer_length": answer_length,
            "answer_length_score": length_score,
            "source_count": source_count,
            "source_score": source_score,
            "completeness_score": completeness_score,
            "overall_score": (time_score + length_score + source_score + completeness_score) / 4
        }


async def run_comparative_evaluation():
    """
    Main function that compares different RAG configurations
    """
    print("⚖️  Comparative RAG Evaluation Demo")
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

    # Define test questions
    test_questions = [
        "What is Happy Payments?",
        "What are the main features of ISO 8583?",
        "How does payment processing work?",
        "What are the benefits of standardized payment messages?"
    ]

    # Define configurations to compare
    configurations = [
        {
            "name": "Gemini-Standard",
            "llm": selected_project.get("llm", "gemini"),
            "embed_model": selected_project.get("embed_model", "gemini"),
            "description": "Standard Gemini configuration"
        },
        # Add more configurations as needed
    ]

    print(f"📋 Testing {len(configurations)} configuration(s) with {len(test_questions)} questions")
    print(f"📋 Total evaluations: {len(configurations) * len(test_questions)}")

    # Use the same table naming logic as in the UI/app
    project_name = selected_project.get("name", "").strip()
    sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
    sanitized_name = "_".join(filter(None, sanitized_name.split("_")))
    table_prefix = f"yasrl_{sanitized_name or selected_pid}"

    from yasrl.vector_store import VectorStoreManager

    # Run evaluation for each configuration
    comparison_results = []

    for config in configurations:
        # Use the selected project's table for all configs
        db_manager = VectorStoreManager(
            postgres_uri=os.getenv("POSTGRES_URI") or "",
            vector_dimensions=768,
            table_prefix=table_prefix
        )
        result = await evaluate_pipeline_config(
            config["name"],
            config["llm"],
            config["embed_model"],
            test_questions,
            db_manager=db_manager
        )
        result["description"] = config["description"]
        comparison_results.append(result)

    # Analyze and compare results
    print("\n📊 Comparative Analysis")
    print("=" * 50)

    # Sort by average score
    valid_results = [r for r in comparison_results if "error" not in r]
    valid_results.sort(key=lambda x: x["average_score"], reverse=True)

    print("🏆 Rankings by Overall Score:")
    for i, result in enumerate(valid_results, 1):
        print(f"{i}. {result['config_name']}")
        print(f"   Average Score: {result['average_score']:.3f}")
        print(f"   Avg Response Time: {result['average_response_time']:.2f}s")
        print(f"   Avg Sources: {result['average_source_count']:.1f}")
        print(f"   Description: {result['description']}")
        print()

    # Performance comparison
    if len(valid_results) > 1:
        best = valid_results[0]
        print(f"🥇 Best Performer: {best['config_name']}")
        print(f"   Score: {best['average_score']:.3f}")
        print(f"   Response Time: {best['average_response_time']:.2f}s")

        # Compare with others
        for result in valid_results[1:]:
            score_diff = ((best['average_score'] - result['average_score']) / result['average_score']) * 100
            time_diff = ((result['average_response_time'] - best['average_response_time']) / best['average_response_time']) * 100
            print(f"\n   vs {result['config_name']}:")
            print(f"     {score_diff:+.1f}% better score")
            print(f"     {time_diff:+.1f}% faster response time")

    # Save detailed comparison results
    output_file = Path("./results") / f"comparative_evaluation_results_{sanitized_name or selected_pid}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "evaluation_type": "comparative",
            "project": selected_display,
            "configurations_tested": len(configurations),
            "questions_per_config": len(test_questions),
            "test_questions": test_questions,
            "results": comparison_results,
            "ranking": [r["config_name"] for r in valid_results]
        }, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")

# Update evaluate_pipeline_config to accept db_manager
async def evaluate_pipeline_config(config_name: str, llm: str, embed_model: str, questions: List[str], db_manager=None) -> Dict[str, Any]:
    """
    Evaluate a specific pipeline configuration

    Args:
        config_name: Name identifier for this configuration
        llm: LLM provider to use
        embed_model: Embedding model to use
        questions: List of questions to evaluate
        db_manager: VectorStoreManager instance to use

    Returns:
        Dictionary with configuration results
    """
    print(f"\n🔧 Testing Configuration: {config_name}")
    print(f"   LLM: {llm}, Embeddings: {embed_model}")

    try:
        # Initialize pipeline with specific configuration
        start_init = time.time()
        if db_manager is not None:
            pipeline = await RAGPipeline.create(llm=llm, embed_model=embed_model, db_manager=db_manager)
        else:
            pipeline = await RAGPipeline.create(llm=llm, embed_model=embed_model)
        init_time = time.time() - start_init

        evaluator = ComparativeEvaluator()
        results = []
        total_time = 0

        # Evaluate each question
        for i, question in enumerate(questions, 1):
            print(f"   Question {i}/{len(questions)}: {question[:50]}...")

            # Time the response
            start_time = time.time()
            result = await pipeline.ask(question)
            response_time = time.time() - start_time
            total_time += response_time

            # Evaluate the response
            evaluation = evaluator.evaluate_response(question, result, response_time)
            evaluation["question"] = question
            evaluation["answer"] = result.answer
            results.append(evaluation)

            print(f"   ⏱️  Response time: {response_time:.2f}s")
            print(f"   📊 Score: {evaluation['overall_score']:.2f}")

        # Calculate aggregate metrics
        avg_score = sum(r["overall_score"] for r in results) / len(results)
        avg_response_time = total_time / len(questions)
        avg_source_count = sum(r["source_count"] for r in results) / len(results)

        await pipeline.cleanup()

        return {
            "config_name": config_name,
            "llm": llm,
            "embed_model": embed_model,
            "initialization_time": init_time,
            "average_response_time": avg_response_time,
            "average_score": avg_score,
            "average_source_count": avg_source_count,
            "total_questions": len(questions),
            "detailed_results": results
        }

    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        return {
            "config_name": config_name,
            "llm": llm,
            "embed_model": embed_model,
            "error": str(e),
            "average_score": 0
        }

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Run the comparative evaluation
    asyncio.run(run_comparative_evaluation())