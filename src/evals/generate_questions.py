import os
import json
import asyncio
from typing import List, Dict, Any

from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.schema import Document
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv

from src.yasrl.pipeline import RAGPipeline
from src.yasrl.vector_store import VectorStoreManager

load_dotenv()

# Constants
QA_GEN_PROMPT_TMPL = """
Here is a context from a document.
Your task is to generate a question and a concise answer based ONLY on the provided context.
Do not use any information outside of the context.

The output should be in a JSON format with two keys: "question" and "answer".

Context:
{context_str}
JSON Output:
"""

EVOLVE_PROMPT_TMPL = """
You are an expert test case designer for a RAG system.
Your goal is to rewrite a simple question to make it more complex and require reasoning, while ensuring it can still be answered by the given context.

Here are some ways to make it more complex:

Change it from "what" to "what is the significance of".
Add a condition or a constraint.
Ask for a summary or implication instead of a direct fact.
Do not ask for information that is not present in the context.

Context: "{context}"
Simple Question: "{simple_question}"

Rewrite the question to be more complex.
Evolved Question:
"""

def select_project(project_id: str) -> Dict[str, Any]:
    """
    Selects a project configuration from the projects.json file.

    Args:
        project_id: The ID of the project to select.

    Returns:
        The project configuration dictionary.
    """
    with open("src/evals/projects.json", "r") as f:
        projects = json.load(f)

    for project in projects:
        if project["id"] == project_id:
            return project

    raise ValueError(f"Project with ID '{project_id}' not found.")

async def generate_questions_with_datasetgenerator(project: Dict[str, Any], num_questions: int = 3) -> List[Dict[str, str]]:
    """
    Generates a dataset of questions and answers using DatasetGenerator.

    Args:
        project: The project configuration.
        num_questions: The number of questions to generate.

    Returns:
        A list of dictionaries, where each dictionary has "question", "answer", and "context" keys.
    """
    print("Generating questions with DatasetGenerator...")

    # Initialize the RAG pipeline
    pipeline = await RAGPipeline.create(llm="gemini", embed_model="gemini")

    # Index documents
    for source in project["sources"]:
        await pipeline.index(source, project_id=project["id"])

    # Get all indexed nodes for the project
    nodes = await pipeline.db_manager.get_all_chunks(project_id=project["id"])
    if not nodes:
        print(f"No content found for project {project['id']}. Skipping question generation.")
        await pipeline.cleanup()
        return []

    # Generate dataset from a subset of nodes
    nodes_to_process = nodes[:num_questions] # Process up to num_questions chunks
    dataset_generator = DatasetGenerator.from_nodes(
        nodes=nodes_to_process,
        llm=Gemini(),
        num_questions_per_chunk=1, # Generate 1 question per chunk
    )

    rag_dataset = await dataset_generator.agenerate_dataset_from_nodes()

    # Format the output
    dataset = []
    for item in rag_dataset:
        # Assuming the first context is the relevant one
        context = item.reference_contexts[0] if item.reference_contexts else ""
        dataset.append({
            "question": item.query,
            "answer": item.reference_answer,
            "context": context
        })

    await pipeline.cleanup()
    print("Finished generating questions with DatasetGenerator.")
    return dataset

async def generate_questions_with_custom_prompt(project: Dict[str, Any], num_questions: int = 3) -> List[Dict[str, str]]:
    """
    Generates a dataset of questions and answers using a custom prompt for each chunk.

    Args:
        project: The project configuration.
        num_questions: The number of questions to generate.

    Returns:
        A list of dictionaries, where each dictionary has "question", "answer", and "context" keys.
    """
    print("Generating questions with custom prompt...")

    # Initialize the RAG pipeline
    pipeline = await RAGPipeline.create(llm="gemini", embed_model="gemini")
    llm = Gemini()

    # Index documents if they are not already indexed, or if we need to force re-indexing.
    # For this script, we'll index every time to ensure freshness.
    for source in project["sources"]:
        await pipeline.index(source, project_id=project["id"])

    # Get all indexed nodes for the project
    nodes = await pipeline.db_manager.get_all_chunks(project_id=project["id"])
    if not nodes:
        print(f"No content found for project {project['id']}. Skipping question generation.")
        await pipeline.cleanup()
        return []

    # Generate questions from a subset of nodes
    dataset = []
    nodes_to_process = nodes[:num_questions] # Process up to num_questions chunks

    for node in nodes_to_process:
        context = node.get_content()
        prompt = QA_GEN_PROMPT_TMPL.format(context_str=context)
        response = await llm.acomplete(prompt)

        try:
            qa_pair = json.loads(response.text)
            qa_pair["context"] = context # Add context to the output
            dataset.append(qa_pair)
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Could not decode JSON from LLM response: {response.text}")
            continue

    await pipeline.cleanup()
    print("Finished generating questions with custom prompt.")
    return dataset

async def evolve_questions(datasets: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    Evolves simple questions to be more complex using an LLM.

    Args:
        datasets: A list of datasets, where each dataset is a list of QA pairs.
                  Each QA pair dictionary must contain a "context" key.

    Returns:
        A list of dictionaries, where each dictionary has "evolved_question" and "original_question" keys.
    """
    print("Evolving questions...")

    llm = Gemini()

    evolved_dataset = []
    for dataset in datasets:
        for qa_pair in dataset:
            simple_question = qa_pair["question"]
            context = qa_pair["context"] # Use context from the dataset

            prompt = EVOLVE_PROMPT_TMPL.format(context=context, simple_question=simple_question)
            response = await llm.acomplete(prompt)

            evolved_question = response.text.strip()

            evolved_dataset.append({
                "original_question": simple_question,
                "evolved_question": evolved_question
            })

    print("Finished evolving questions.")
    return evolved_dataset

async def main():
    """
    Main function to demonstrate the question generation and evolution workflow.
    """
    # Select a project
    project_id = "project_1"
    project = select_project(project_id)

    # Generate questions using both methods
    dataset1 = await generate_questions_with_datasetgenerator(project, num_questions=3)
    dataset2 = await generate_questions_with_custom_prompt(project, num_questions=3)

    print("\n--- Generated Questions (DatasetGenerator) ---")
    for qa in dataset1:
        print(f"Q: {qa['question']}\nA: {qa['answer']}\nContext: {qa['context']}\n")

    print("\n--- Generated Questions (Custom Prompt) ---")
    for qa in dataset2:
        print(f"Q: {qa['question']}\nA: {qa['answer']}\nContext: {qa['context']}\n")

    # Evolve the questions
    evolved_dataset = await evolve_questions([dataset1, dataset2])

    print("\n--- Evolved Questions ---")
    for item in evolved_dataset:
        print(f"Original: {item['original_question']}")
        print(f"Evolved:  {item['evolved_question']}\n")

if __name__ == "__main__":
    # To run this script, you need to have a PostgreSQL database running
    # and the required environment variables set in a .env file.
    # See the .env.example file for more details.

    # Note: This script requires API keys with access to Gemini models.
    # If you get an error, make sure your keys are correct and have the
    # necessary permissions.

    asyncio.run(main())