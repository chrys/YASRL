
### **Project Specification: The "SimpleRAG" Python Library**

**Version:** 1.0

**Date:** 12 July, 2025

**Target Audience:** Senior Software Architect / Lead Developer

#### **1. Executive Summary**

This document outlines the specifications for a Python library, named "YASRL," (Yet Another Simple RAG Library) designed to provide a streamlined, developer-first experience for building and deploying Retrieval-Augmented Generation (RAG) pipelines.

The library's core value proposition is **simplicity through opinionated design**. It will act as a high-level orchestration layer (Facade) over the powerful LlamaIndex framework, abstracting away its boilerplate and complexity for the most common use cases. The end goal is to allow a developer to create a production-ready, conversational RAG system with minimal configuration and code.

#### **2. Core Philosophy & Guiding Principles**

The architect should ensure all design and implementation decisions adhere to these principles:

*   **Developer Experience First:** The primary goal is to minimize developer friction. The API should be intuitive, simple, and require very little code to achieve powerful results.
*   **Simplicity Over Configuration:** The library will favor convention over configuration. It will make intelligent, best-practice decisions on behalf of the user, eliminating the need for complex configuration objects.
*   **Opinionated Defaults:** The library will come pre-configured with sensible defaults for models, chunking strategies, and other parameters. These defaults will be chosen to provide a high-quality result out-of-the-box.
*   **Extensible for the Future:** While prioritizing simplicity now, the underlying architecture should be modular to allow for future expansion (e.g., adding new evaluators or providers).

#### **3. High-Level Architecture**

The library will be an **asynchronous-first facade** over the LlamaIndex framework. It will consist of a primary orchestrator class (`RAGPipeline`) that manages underlying LlamaIndex components.

**Data Flow Diagram:**

```
[Developer Code] -> pipeline.index(source) -> [RAGPipeline Orchestrator]
    |                                                  |
    |                                                  v
    |                                          [LlamaIndex Loaders] -> [LlamaIndex TextSplitter (NodeParser)]
    |                                                  |                         (Chunk size decided by pipeline)
    |                                                  v
    |                                          [LlamaIndex VectorStore (PGVector)] <- [LlamaIndex Embeddings]
    |                                                  ^ (Upsert Logic)                   (Model decided by pipeline)
    v
[Developer Code] -> pipeline.ask(query) -> [RAGPipeline Orchestrator]
                                                   |
                                                   v
                                          [Retriever (w/ Re-ranker)] -> [LlamaIndex LLM] -> [QueryResult] -> [Developer]
```

#### **4. Core Components & API Specification**

##### **4.1. `RAGPipeline` Orchestrator**

This is the main entry point for the user.

```python
# Signature
class RAGPipeline:
    def __init__(self, llm: str, embed_model: str):
        # ... implementation ...

    async def index(self, source: str | list[str]):
        # ... implementation ...

    async def ask(self, query: str, conversation_history: list[dict] | None = None) -> QueryResult:
        # ... implementation ...
```

*   **`__init__(self, llm: str, embed_model: str)`:**
    *   Initializes the pipeline with simple string identifiers.
    *   `llm`: The generative model provider. Supported strings: `"openai"`, `"gemini"`, `"ollama"`.
    *   `embed_model`: The embedding model provider. Supported strings: `"openai"`, `"gemini"`, `"opensource"`.
    *   The library is responsible for translating these strings into the appropriate, pre-configured LlamaIndex objects (e.g., `llm="openai"` might instantiate `llama_index.llms.openai.OpenAI(model="gpt-4o-mini")`).

##### **4.2. Configuration Management**

*   **Secrets & Keys:** All API keys and connection strings **must** be loaded from environment variables. The library **must not** accept keys as direct arguments.
    *   `OPENAI_API_KEY`: For OpenAI services.
    *   `GOOGLE_API_KEY`: For Gemini services.
    *   `POSTGRES_URI`: For the `pgvector` database connection (e.g., `postgresql+psycopg2://user:pass@host:port/dbname`).
    *   `OLLAMA_HOST`: (Optional) For connecting to a non-default Ollama instance.
*   The library should raise a clear `ConfigurationError` if a required environment variable is missing for a selected service.

##### **4.3. Data Ingestion & Indexing (`index` method)**

*   **Signature:** `async def index(self, source: str | list[str])`
*   **Intelligent Source Handling:** The method must automatically detect the type of `source` and use the appropriate LlamaIndex loader:
    *   If `source` is a path to a directory, use `SimpleDirectoryReader` to recursively load all supported files.
    *   If `source` is a path to a single file (e.g., `.../doc.pdf`), load that specific file.
    *   If `source` is a list of URLs, load them.
*   **Upsert Logic:** The indexing process must be idempotent and support updates.
    1.  For each loaded document, a unique, deterministic `document_id` **must** be generated (e.g., using the absolute file path or URL).
    2.  Before indexing a new document, the library **must** check the `pgvector` store for any existing chunks with the same `document_id`.
    3.  If existing chunks are found, they **must** be deleted from the vector store.
    4.  The new document is then chunked, embedded, and its new chunks are stored, all associated with the `document_id`. This ensures new documents are added and existing documents are updated seamlessly.

##### **4.4. Data Retrieval & Generation (`ask` method)**

*   **Signature:** `async def ask(self, query: str, conversation_history: list[dict] | None = None) -> QueryResult`
*   **Functionality:**
    1.  Takes the user's `query`.
    2.  Embeds the query using the configured embedding model.
    3.  Retrieves relevant chunks from the `pgvector` store.
    4.  **Re-ranking:** The retrieved chunks should be passed through a re-ranker (e.g., LlamaIndex's `CohereRerank` or a cross-encoder) to improve context quality before being passed to the LLM. This should be an internal, enabled-by-default feature.
    5.  Constructs a final prompt using a built-in template. This template **must** correctly format the `conversation_history`, the retrieved context chunks, and the new `query`.
    6.  Sends the prompt to the configured generative LLM.
    7.  Returns a `QueryResult` object.

##### **4.5. Data Models (Output)**

The library must use `dataclasses` for its return objects to ensure a clear and type-safe API contract.

```python
from dataclasses import dataclass, field

@dataclass
class SourceChunk:
    text: str
    metadata: dict = field(default_factory=dict) # Includes original source path/URL
    score: float | None = None # Relevance score from retriever/re-ranker

@dataclass
class QueryResult:
    answer: str
    source_chunks: list[SourceChunk] = field(default_factory=list)
```

#### **5. Internal Logic & Automated Decisions**

*   **Dynamic Chunk Size Management:** The chunk size **must not** be a direct user parameter. Instead, the `RAGPipeline` will internally maintain a mapping to determine the optimal chunk size based on the chosen embedding model.
    *   *Example Mapping:* `{"openai": 1024, "gemini": 1024, "opensource": 512}`.
*   **Database Schema:** The library will operate on a "one index per database" model. It will use a default, prefixed set of table names (e.g., `srag__data`, `srag__metadata`) to store its data within the provided PostgreSQL database. The schema **must** include a column to store the `document_id` for each vector chunk to enable the upsert logic.

**All projects will use a separate table that will be created after the project is created. 

#### **6. Evaluation**

*   To provide evaluation capabilities, the library will include a lightweight, extensible evaluation module.
*   **Architecture:** Define a `BaseEvaluator` abstract class. The first implementation will be a `RagasEvaluator`, which acts as a convenient wrapper around the `ragas` library. This design allows for other evaluation tools (like TruLens) to be integrated in the future by creating new classes that inherit from `BaseEvaluator`.

#### **7. Error Handling & Logging**

*   The library must implement a configurable logging system using Python's standard `logging` module. This will allow developers to set the log level (DEBUG, INFO, etc.) to see detailed information about the pipeline's execution, which is crucial for debugging.
*   Custom, specific exceptions should be used (e.g., `ConfigurationError`, `IndexingError`) to allow for programmatic error handling by the user.

#### **8. Testing Strategy**

A robust testing suite is required for reliability.

*   **Unit Tests:** Each individual component (e.g., the logic for choosing a chunk size, the source type detection) must have comprehensive unit tests.
*   **Integration Tests:** The full `RAGPipeline` flow (`index` and `ask` methods) must be tested. All external services (LLM APIs, database calls) **must be mocked** using a library like `unittest.mock`. These tests ensure that all components work together correctly without relying on live network connections or credentials.

#### **9. Packaging & Dependency Management**

*   The library will be packaged using a `pyproject.toml` file.
*   **Core Dependencies:** The core installation (`pip install yasrl`) will be lightweight.
*   **Optional Dependencies:** Heavy or provider-specific dependencies **must** be managed as "extras."
    *   `pip install yasrl[openai]`
    *   `pip install yasrl[gemini]`
    *   `pip install yasrl[ollama]`
    *   `pip install yasrl[postgres]` (for the `pgvector` client)

#### **10. Documentation**

*   Documentation is a first-class feature.
*   All public classes and methods **must** have comprehensive, well-formatted docstrings (e.g., Google or NumPy style).
*   The project **must** be configured to use **Sphinx** to automatically generate a professional documentation website from these docstrings.