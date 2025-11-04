import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# --- Step 1: Configuration and Data Loading ---

# ⚠️ Set your API key as an environment variable (recommended)
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Create a dummy file for the example
# In a real scenario, you'd have your PDF, markdown, or other files here.
with open("data/gemini_summary.txt", "w") as f:
    f.write(
        "Gemini is a family of multimodal large language models developed by Google AI. "
        "It includes Gemini Ultra, the largest and most capable model; "
        "Gemini Pro, a model optimized for scaling across a wide range of tasks; "
        "and Gemini Flash, a light-weight and efficient model for faster applications. "
        "Gemini models are natively multimodal, meaning they were trained from the start "
        "to understand and operate across text, images, and other modalities."
    )

# Load the documents from the directory
my_llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.llm = my_llm
my_embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
Settings.embed_model = my_embed_model
documents = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model = my_embed_model)
query_engine = index.as_query_engine(llm=my_llm)

# response = query_engine.query(
#     "Provide a concise summary of the Gemini model family from Google AI."
# )
# print("Response from LLM:")
# print(response)

hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine=query_engine, query_transform=hyde)

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=hyde_query_engine,
        metadata=ToolMetadata(
            name="docs",
            description="Useful for answering questions about the Gemini model family from Google AI.",
        ),
    )
]

# wrap query engine tool in Sub-Question Query Engine object - Final Query Engine
SQ_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

# --- Step 2: Index Creation ---
# Settings.llm = my_llm
# Settings.embed_model = my_embed_model 
# index = VectorStoreIndex.from_documents(documents)
# print("Index created successfully.")

# --- Step 3: Dataset Generation ---
print("Generating QA datasets from the indexed documents...")
dataset_generator =  RagDatasetGenerator.from_documents(
     documents,
     llm = Settings.llm,
     question_gen_query = "Using the provided context, formulate a single question and its answer",
     num_questions_per_chunk=2,
     workers=2,
 )
print("Dataset generator initialized.")
qa_datasets = dataset_generator.generate_dataset_from_nodes()
print("The QA datasets are structured as follows:")
print(qa_datasets)
    