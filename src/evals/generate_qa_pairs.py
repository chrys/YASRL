import os
import logging
import argparse
from typing import List, Dict, Any
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import Document
from markitdown import MarkItDown
from llama_index.core.evaluation import ContextRelevancyEvaluator


# ⚠️ Set your API key as an environment variable (recommended)
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_markdown(source_path: str) -> tuple[str, str]:
    """
    Converts the file at the source path into a clean Markdown string.
    """
    # Initialize MarkItDown
    md = MarkItDown()
    
    # Convert the source file
    converted_result = md.convert(source_path)
    
    # Extract the clean Markdown text
    markdown_text = converted_result.text_content
    #logger.info(f"Converted {source_path} to markdown ({len(markdown_text)} chars)")
    #logger.info(markdown_text[:500] + "..." if len(markdown_text) > 500 else markdown_text)
    
    # Return the text and the original path (the source input)
    return markdown_text, source_path

def load_data(source_path: str) -> List[Document]:
    """
    Load documents from a source file and convert to markdown.
    
    Args:
        source_path: Path to the source file to load and convert
        
    Returns:
        List of loaded documents
    """
    logger.info(f"Loading and converting document from {source_path}...")
    
    # Convert the source file to markdown
    markdown_text, converted_path = convert_to_markdown(source_path)
    logger.info(f"Converted {source_path} to markdown ({len(markdown_text)} chars)")
    
    # Create a temporary markdown file
    temp_markdown_path = source_path.replace(os.path.splitext(source_path)[1], "_converted.md")
    with open(temp_markdown_path, "w") as f:
        f.write(markdown_text)
    logger.info(f"Created temporary markdown file at {temp_markdown_path}")
    
    # Load the markdown file with SimpleDirectoryReader
    temp_dir = os.path.dirname(temp_markdown_path)
    documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
    logger.info(f"Loaded {len(documents)} documents")
    
    return documents


def generate_qa_dataset(
    documents: List[Document],
    num_questions_per_chunk: int = 2,
    workers: int = 2,
    llm_model: str = "gemini-2.5-flash",
    embed_model: str = "gemini-embedding-001"
) -> Any:
    """
    Generate QA pairs from documents using LlamaIndex DatasetGenerator.
    
    Args:
        documents: List of documents to generate QA pairs from
        num_questions_per_chunk: Number of questions to generate per document chunk
        workers: Number of workers for parallel processing
        llm_model: LLM model to use for generation
        embed_model: Embedding model to use
        
    Returns:
        Dictionary containing generated QA datasets
    """
    logger.info("Setting up LLM and embedding models...")
    
    # Initialize LLM and embedding model
    my_llm = GoogleGenAI(model=llm_model)
    Settings.llm = my_llm
    
    my_embed_model = GoogleGenAIEmbedding(model_name=embed_model)
    Settings.embed_model = my_embed_model
    
    logger.info(f"Generating QA datasets from {len(documents)} documents...")
    
    # Create index from documents
    index = VectorStoreIndex.from_documents(documents, embed_model=my_embed_model)
    logger.info("Index created successfully")
    
    # Initialize dataset generator
    dataset_generator = RagDatasetGenerator.from_documents(
        documents,
        llm=Settings.llm,
        question_gen_query="Using the provided context, formulate a single question and its answer",
        num_questions_per_chunk=num_questions_per_chunk,
        workers=workers,
    )
    logger.info("Dataset generator initialized")
    
    # Generate QA datasets
    qa_datasets = dataset_generator.generate_dataset_from_nodes()
    logger.info(f"Generated QA datasets: {type(qa_datasets)}")
    
    return qa_datasets


def print_scored_qa_pairs(results: List[Dict[str, Any]]) -> None:
    """
    Print scored QA pairs in a formatted manner.
    
    Args:
        results: List of dictionaries containing 'query', 'score', and 'example'
    """
    logger.info("Printing scored QA pairs...")
    
    if not results:
        logger.warning("No scored QA pairs to print")
        return
    
    for idx, result in enumerate(results, 1):
        question = result.get("query", "N/A")
        score = result.get("score", "N/A")
        example = result.get("example", None)
        answer = getattr(example, 'reference_answer', getattr(example, 'answer', 'N/A')) if example else 'N/A'
        
        print(f"\nQuestion {idx}: {question}")
        print(f"Answer {idx}: {answer}")
        print(f"Relevancy Score: {score}")
        print("-" * 80)

def print_qa_pairs(qa_datasets: Any) -> None:
    """
    Print QA pairs in a formatted manner.
    
    Args:
        qa_datasets: Generated QA dataset containing questions and answers
    """
    logger.info("Printing QA pairs...")
    
    if not hasattr(qa_datasets, 'examples') or not qa_datasets.examples:
        logger.warning("No QA pairs found in the dataset")
        return
    
    examples = qa_datasets.examples
    for idx, example in enumerate(examples, 1):
        question = getattr(example, 'query', getattr(example, 'question', 'N/A'))
        answer = getattr(example, 'reference_answer', getattr(example, 'answer', 'N/A'))
        
        print(f"\nQuestion {idx}: {question}")
        print(f"Answer {idx}: {answer}")
        print("-" * 80)

def evaluate_qa_pairs(qa_datasets: Any): 
    """
    Evaluate the generated QA pairs for context relevancy.
    
    Args:
        qa_datasets: Generated QA dataset containing questions and answers
    """
    
    #1 Evaluator initialisation 
    logger.info("Evaluating QA pairs for context relevancy...")
    judge_llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0)
    evaluator = ContextRelevancyEvaluator(llm=judge_llm)
    results = [] 

    # 2 Iterate through dataset and provide sore
    for i, example in enumerate(qa_datasets.examples): 
        eval_result = evaluator.evaluate(
            query = example.query,
            contexts= example.reference_contexts,
        )  
        # Store the results
        results.append({
            "query": example.query,
            "score": eval_result.score, # The numerical score assigned by the LLM
            "example": example # Keep the original example object
        })    
    return results    

def get_top_n_results(results: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """
    Get the top N results based on the score.
    
    Args:
        results: List of dictionaries containing 'query', 'score', and 'example'
        n: Number of top results to return
        
    Returns:
        List of top N results
    """
    # Sort results by score in descending order
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Return the top N results
    return sorted_results[:n]

def main():
    """
    Main function demonstrating the use of data loading and dataset generation functions.
    Supports command-line arguments for source file and number of QA pairs.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from a source document using LlamaIndex DatasetGenerator"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to the source file (PDF, txt, etc.) to generate QA pairs from"
    )
    parser.add_argument(
        "--total",
        type=int,
        default=2,
        help="Total number of QA pairs to generate for the entire document (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file for results (optional, for programmatic use)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting QA pair generation pipeline...")
    
    # Determine source file
    source_file = args.source
    
    if source_file is None:
        # Create a dummy file for the example if no source is provided
        # In a real scenario, you'd have your PDF, markdown, or other files here.
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)
        
        source_file = os.path.join(data_dir, "gemini_summary.txt")
        if not os.path.exists(source_file):
            with open(source_file, "w") as f:
                f.write(
                    "Gemini is a family of multimodal large language models developed by Google AI. "
                    "It includes Gemini Ultra, the largest and most capable model; "
                    "Gemini Pro, a model optimized for scaling across a wide range of tasks; "
                    "and Gemini Flash, a light-weight and efficient model for faster applications. "
                    "Gemini models are natively multimodal, meaning they were trained from the start "
                    "to understand and operate across text, images, and other modalities."
                )
            logger.info(f"Created dummy file at {source_file}")
    
    # Verify source file exists
    if not os.path.exists(source_file):
        logger.error(f"Source file not found: {source_file}")
        return
    
    logger.info(f"Using source file: {source_file}")
    
    pairs = args.total
    
    # Step 1: Load data from the source file
    documents = load_data(source_path=source_file)
    
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
    

    # Step 2: Generate QA dataset
    qa_datasets = generate_qa_dataset(
         documents=documents,
         num_questions_per_chunk=pairs,
         workers=2,
         llm_model="gemini-2.5-flash",
         embed_model="gemini-embedding-001"
    )
    
    # # Step 3: Print QA pairs
    
    results = evaluate_qa_pairs(qa_datasets)
    sorted_results = get_top_n_results(results, n=pairs)
    
    # Step 4: Output results
    if args.output:
        # Write to JSON file for programmatic use
        import json
        output_data = []
        for result in sorted_results:
            example = result.get("example")
            if example:
                question = getattr(example, 'query', 'N/A')
                answer = getattr(example, 'reference_answer', getattr(example, 'answer', 'N/A'))
                contexts = getattr(example, 'reference_contexts', [])
                context = ' '.join(contexts) if contexts else ''
                
                output_data.append({
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'score': result.get('score', 0.0)
                })
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results written to {args.output}")
    else:
        # Print to console for interactive use
        print_scored_qa_pairs(sorted_results)
    
    logger.info(f"QA pair generation completed for pairs requested: {pairs}.")


if __name__ == "__main__":
    main()
    