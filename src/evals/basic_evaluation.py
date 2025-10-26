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
from yasrl.text_processor import TextProcessor

# LlamaIndex imports for dataset generation
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.core.evaluation import DatasetGenerator

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
        )
        evaluation["metrics"]["overall_quality"] = quality_score

        
        # Eyeball test: show sources as a single line
        sources = [chunk.metadata.get("source", "Unknown") for chunk in result.source_chunks]
        eyeball_sources = ", ".join(sources) if sources else "No sources"
        evaluation["metrics"]["eyeball_sources"] = eyeball_sources 

        return evaluation
    
    def create_evaluation_dataset(self, document_path: str, num_questions: int = 10) -> List[Dict[str, str]]:
        """
        Create evaluation dataset using RagDatasetGenerator
        
        Args:
            document_path: Path to the input document or directory
            num_questions: Number of questions to generate
            
        Returns:
            List of dictionaries with 'question' and 'reference_answer' keys
        """
        print(f"📄 Creating evaluation dataset from: {document_path}")
        print(f"🎯 Target questions: {num_questions}")
        
        try:
            # Step 1: Use Gemini as LLM
            llm = Gemini(
                model="gemini-2.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY")
            )
            print("✅ Gemini LLM initialized")
            
            # Step 2: Load the data using SimpleDirectoryReader
            if os.path.isdir(document_path):
                documents = SimpleDirectoryReader(document_path).load_data()
            else:
                documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
            
            print(f"✅ Loaded {len(documents)} document(s)")
            
            # Step 3: Create a node parser to split the document into chunks
            text_processor = TextProcessor(
                chunk_size=1024,  # Reasonable chunk size for question generation
                chunk_overlap=200
            )
            
            # Process documents into nodes
            nodes = text_processor.process_documents(documents)
            print(f"✅ Created {len(nodes)} text chunks")
            
            # Step 4: Use RagDatasetGenerator to generate questions for each node
            dataset_generator = DatasetGenerator(
                nodes=nodes,
                llm=llm,
                num_questions_per_chunk=max(1, num_questions // len(nodes)),  # Distribute questions across chunks
                text_question_template=None,  # Use default template
                text_qa_template=None  # Use default template
            )
            
            print("🔄 Generating questions...")
            eval_questions = dataset_generator.generate_questions_from_nodes()
            
            # Convert to our expected format
            dataset = []
            for i, question in enumerate(eval_questions[:num_questions]):  # Limit to requested number
                dataset.append({
                    "question": question,
                    "reference_answer": f"Generated question {i+1} from document chunks"
                })
            
            print(f"✅ Generated {len(dataset)} evaluation questions")
            
            # Save the generated dataset
            output_dir = Path("./results")
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset_file = output_dir / f"generated_dataset_{len(dataset)}_questions.json"
            
            with open(dataset_file, "w") as f:
                json.dump({
                    "source_document": document_path,
                    "num_questions": len(dataset),
                    "generation_method": "RagDatasetGenerator with Gemini",
                    "questions": dataset
                }, f, indent=2)
            
            print(f"💾 Dataset saved to: {dataset_file}")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Error creating evaluation dataset: {e}")
            raise

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
            "question": "What is the credit car issuer response code 0?",
            "expected_answer": "Response code 0 means the transaction was approved"
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

async def demo_dataset_generation():
    """
    Demo function showing dataset generation capabilities
    """
    print("\n" + "=" * 60)
    print("🎯 DATASET GENERATION DEMO")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = BasicEvaluator()
    
    # Ask user for document path
    print("\n📁 Document Input Options:")
    print("1. Use a specific file path")
    print("2. Use a directory path")
    print("3. Skip demo")
    
    choice = input("Choose an option [1-3]: ").strip()
    
    if choice == "3":
        print("⏭️ Skipping dataset generation demo")
        return
    
    if choice == "1":
        doc_path = input("Enter file path: ").strip()
    elif choice == "2":
        doc_path = input("Enter directory path: ").strip()
    else:
        print("ℹ️ Using default demo document")
        # Create a demo document for testing
        demo_dir = Path("./demo_docs")
        demo_dir.mkdir(exist_ok=True)
        demo_file = demo_dir / "sample.txt"
        
        demo_content = """
        Welcome to YASRL - Yet Another Search and Retrieval Library.
        
        YASRL is a powerful RAG (Retrieval-Augmented Generation) system that combines 
        vector search with large language models to provide accurate, source-backed answers.
        
        Key Features:
        - Multi-modal document processing
        - Advanced chunking strategies
        - Hybrid search capabilities
        - Comprehensive evaluation tools
        
        The system supports various embedding models including OpenAI, Gemini, and local models.
        It can process PDFs, web pages, and text documents efficiently.
        
        For evaluation, YASRL provides both automated metrics and human feedback collection.
        """
        
        with open(demo_file, "w") as f:
            f.write(demo_content)
        
        doc_path = str(demo_file)
        print(f"📝 Created demo document at: {doc_path}")
    
    if not os.path.exists(doc_path):
        print(f"❌ Path does not exist: {doc_path}")
        return
    
    # Generate dataset
    try:
        num_questions = int(input("Number of questions to generate [5]: ").strip() or "5")
        
        print(f"\n🚀 Generating evaluation dataset...")
        dataset = evaluator.create_evaluation_dataset(doc_path, num_questions)
        
        # Show sample questions
        print(f"\n📋 Sample Generated Questions:")
        for i, item in enumerate(dataset[:3], 1):  # Show first 3
            print(f"{i}. {item['question']}")
        
        if len(dataset) > 3:
            print(f"... and {len(dataset) - 3} more questions")
        
        print(f"\n✅ Dataset generation complete!")
        
    except Exception as e:
        print(f"❌ Error in dataset generation demo: {e}")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    async def main():
        """Main function with demo options"""
        print("🚀 YASRL Evaluation Demo")
        print("=" * 50)
        print("1. Run basic evaluation with existing questions")
        print("2. Demo dataset generation")
        print("3. Run both")
        
        choice = input("Choose an option [1-3]: ").strip()
        
        if choice in ["1", "3"]:
            await run_basic_evaluation()
        
        if choice in ["2", "3"]:
            await demo_dataset_generation()
        
        print("\n🎉 Demo complete!")
    
    # Run the main demo
    asyncio.run(main())