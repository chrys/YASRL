# yasrl Development Blueprint & Implementation Checklist

## Overview
This document provides a step-by-step implementation plan for the yasrl (Yet Another RAG Library) project. Each section contains detailed prompts for code generation that build incrementally on previous work.

## Phase 1: Project Foundation & Core Infrastructure

### Step 1.1: Project Structure and Configuration Setup

**Objective**: Establish the basic project structure, packaging configuration, and development environment.

```
Create a Python library project structure for a package called "yasrl" (Yet Another RAG Library). Set up the following:

1. Create a proper Python package structure with:
   - Root directory structure with src/yasrl/ package layout
   - pyproject.toml with build system configuration using setuptools
   - Basic package metadata (name: yasrl, version: 0.1.0, description, author info)
   - Define optional dependency groups: [openai], [gemini], [ollama], [postgres], [dev], [test], [docs]

2. Core dependencies in pyproject.toml:
   - llama-index-core
   - asyncio support packages
   - Standard logging and dataclass support

3. Development infrastructure:
   - .gitignore appropriate for Python projects
   - Basic README.md with installation instructions

4. Create basic package files:
   - src/yasrl/__init__.py with version info
   - src/yasrl/py.typed for type hint support
   - Empty tests/ directory structure

Ensure the project follows modern Python packaging standards and is ready for development.
```

### Step 1.2: Custom Exceptions and Error Handling

**Objective**: Define the custom exception hierarchy for clear error handling.

```
Create a comprehensive exception handling system for the yasrl library. Implement the following in src/yasrl/exceptions.py:

1. Base exception class yasrlError that inherits from Exception
2. Specific exception classes:
   - ConfigurationError: For missing environment variables or invalid configurations
   - IndexingError: For failures during document indexing
   - RetrievalError: For failures during query retrieval
   - EvaluationError: For evaluation-related failures

3. Each exception should:
   - Have a clear docstring explaining when it's raised
   - Accept an optional message parameter
   - Include a default message that describes the error type

4. Create comprehensive unit tests in tests/test_exceptions.py that:
   - Test each exception can be raised and caught properly
   - Verify exception messages work correctly
   - Test inheritance hierarchy

5. Update src/yasrl/__init__.py to expose these exceptions for public use

Focus on clear, actionable error messages that help developers understand what went wrong and how to fix it.
```

### Step 1.3: Configuration Management System

**Objective**: Implement environment variable-based configuration with clear error handling.

```
Create a configuration management system for the yasrl library in src/yasrl/config.py:

1. Implement a ConfigManager class that:
   - Loads all required environment variables on initialization
   - Validates that required variables are present based on selected providers
   - Provides methods to get configuration values
   - Raises ConfigurationError for missing required variables

2. Required environment variables to support:
   - OPENAI_API_KEY (for OpenAI services)
   - GOOGLE_API_KEY (for Gemini services)  
   - POSTGRES_URI (for pgvector database)
   - OLLAMA_HOST (optional, with default to localhost:11434)

3. Implement validation methods:
   - validate_llm_config(llm_provider: str) -> validates required env vars for LLM
   - validate_embed_config(embed_provider: str) -> validates required env vars for embeddings
   - validate_postgres_config() -> validates database connection string

4. Create comprehensive unit tests in tests/test_config.py using unittest.mock to:
   - Test successful configuration loading
   - Test ConfigurationError raising for missing variables
   - Test environment variable validation for different providers
   - Mock environment variables for testing

5. Include clear docstrings and type hints throughout

The configuration system should fail fast with clear error messages about missing environment variables.
```

### Step 1.4: Data Models and Type Definitions

**Objective**: Define the core data models using dataclasses for type safety.

```
Create the core data models for the yasrl library in src/yasrl/models.py:

1. Implement the following dataclasses with proper type hints:

   @dataclass
   class SourceChunk:
       text: str
       metadata: dict = field(default_factory=dict)
       score: float | None = None

   @dataclass  
   class QueryResult:
       answer: str
       source_chunks: list[SourceChunk] = field(default_factory=list)

2. Add comprehensive docstrings for each class and field explaining:
   - What each field represents
   - Expected data types and formats
   - Example usage

3. Add validation methods to each dataclass:
   - SourceChunk.validate() -> validates text is not empty, score is in valid range
   - QueryResult.validate() -> validates answer is not empty, source_chunks is valid list

4. Add convenience methods:
   - SourceChunk.from_dict(data: dict) -> creates instance from dictionary
   - QueryResult.to_dict() -> converts to dictionary for serialization
   - QueryResult.get_sources() -> returns list of unique source metadata

5. Create comprehensive unit tests in tests/test_models.py:
   - Test dataclass creation and field access
   - Test validation methods with valid and invalid data
   - Test convenience methods and edge cases
   - Test serialization/deserialization

6. Export these models in src/yasrl/__init__.py

Focus on immutability where possible and clear type definitions for all fields.
```

## Phase 2: Provider Abstraction Layer

### Step 2.1: LLM Provider Factory

**Objective**: Create a factory system for instantiating LLM providers from string identifiers.

```
Create an LLM provider factory system in src/yasrl/providers/llm.py:

1. Define an abstract base class LLMProvider with:
   - Abstract method get_llm() -> returns configured LlamaIndex LLM instance
   - Abstract method validate_config() -> validates required environment variables
   - Property model_name -> returns the model name being used

2. Implement concrete provider classes:
   - OpenAILLMProvider: Creates OpenAI LLM instance (model: gpt-4o-mini)
   - GeminiLLMProvider: Creates Gemini LLM instance  
   - OllamaLLMProvider: Creates Ollama LLM instance

3. Each provider should:
   - Check for required environment variables in __init__
   - Raise ConfigurationError if required variables are missing
   - Use opinionated defaults for model selection
   - Support async operations

4. Create LLMProviderFactory class with:
   - Static method create_provider(provider_name: str) -> LLMProvider
   - Supported providers: "openai", "gemini", "ollama"
   - Clear error messages for unsupported providers

5. Unit tests in tests/test_providers_llm.py:
   - Mock environment variables and LlamaIndex imports
   - Test each provider creation and validation
   - Test factory method with valid and invalid provider names
   - Test error handling for missing environment variables

6. Create src/yasrl/providers/__init__.py and export the factory

Focus on clean abstractions and comprehensive error handling.
```

### Step 2.2: Embedding Provider Factory

**Objective**: Create a factory system for embedding model providers with chunk size management.

```
Create an embedding provider factory system in src/yasrl/providers/embeddings.py:

1. Define an abstract base class EmbeddingProvider with:
   - Abstract method get_embedding_model() -> returns configured LlamaIndex embedding model
   - Abstract method validate_config() -> validates required environment variables
   - Property chunk_size -> returns optimal chunk size for this embedding model
   - Property model_name -> returns the embedding model name

2. Implement concrete provider classes:
   - OpenAIEmbeddingProvider: Creates OpenAI embedding model (chunk_size: 1024)
   - GeminiEmbeddingProvider: Creates Gemini embedding model (chunk_size: 1024)
   - OpenSourceEmbeddingProvider: Creates open-source embedding model (chunk_size: 512)

3. Each provider should:
   - Define optimal chunk size as a class constant
   - Check for required environment variables
   - Use opinionated model defaults
   - Handle rate limiting and async operations properly

4. Create EmbeddingProviderFactory class with:
   - Static method create_provider(provider_name: str) -> EmbeddingProvider
   - Supported providers: "openai", "gemini", "opensource"
   - Cache provider instances to avoid recreating them

5. Add chunk size mapping functionality:
   - Static method get_chunk_size(provider_name: str) -> int
   - Mapping: {"openai": 1024, "gemini": 1024, "opensource": 512}

6. Unit tests in tests/test_providers_embeddings.py:
   - Mock LlamaIndex embedding models
   - Test provider creation and chunk size retrieval
   - Test caching behavior
   - Test configuration validation

Focus on performance optimization through caching and clear chunk size management.
```

### Step 2.3: Document Loader Factory

**Objective**: Create intelligent document loading with automatic source type detection.

```
Create a document loader system in src/yasrl/loaders.py:

1. Define DocumentLoader class with methods:
   - detect_source_type(source: str | list[str]) -> str: Returns "file", "directory", "url", or "url_list"
   - load_documents(source: str | list[str]) -> list[Document]: Loads documents using appropriate LlamaIndex loader
   - generate_document_id(source: str) -> str: Creates deterministic ID from file path or URL

2. Source type detection logic:
   - Single string ending in file extension -> "file"
   - Single string pointing to existing directory -> "directory" 
   - Single string starting with http/https -> "url"
   - List of strings with URLs -> "url_list"

3. Document loading implementation:
   - Use LlamaIndex SimpleDirectoryReader for directories
   - Use appropriate LlamaIndex loaders for individual files
   - Use LlamaIndex web loaders for URLs
   - Handle common file types: .txt, .pdf, .docx, .md

4. Document ID generation:
   - For files: use absolute file path with hash for uniqueness
   - For URLs: use URL with hash for uniqueness
   - Ensure IDs are deterministic for the same source

5. Error handling:
   - Raise IndexingError for unsupported file types
   - Raise IndexingError for non-existent files/directories
   - Raise IndexingError for invalid URLs

6. Unit tests in tests/test_loaders.py:
   - Mock file system operations and LlamaIndex loaders
   - Test source type detection for all scenarios
   - Test document loading for different source types
   - Test document ID generation consistency
   - Test error handling for invalid inputs
   - Do not run the tests. I will do it locally.

Focus on robust source detection and clear error messages for unsupported formats.
```

## Phase 3: Core Pipeline Components

### Step 3.1: Vector Store Management

**Objective**: Implement PostgreSQL vector store with upsert capabilities.

```
Create vector store management in src/yasrl/vector_store.py:

1. Implement VectorStoreManager class with:
   - __init__(postgres_uri: str): Initialize connection to PostgreSQL with pgvector
   - setup_schema(): Create required tables with proper indexing
   - upsert_documents(document_id: str, chunks: list): Delete existing chunks for document_id, then insert new ones
   - retrieve_chunks(query_embedding: list[float], top_k: int = 10): Retrieve most similar chunks
   - delete_document(document_id: str): Remove all chunks for a document

2. Database schema design:
   - Table: yasrl_chunks
   - Columns: id (uuid), document_id (text), chunk_text (text), embedding (vector), metadata (jsonb), created_at (timestamp)
   - Index on document_id for fast upsert operations
   - Vector index on embedding for fast similarity search

3. Upsert logic implementation:
   - Check if chunks exist for document_id
   - Delete existing chunks in a transaction
   - Insert new chunks with same document_id
   - Ensure atomic operations (all succeed or all fail)

4. Integration with LlamaIndex:
   - Use LlamaIndex PGVectorStore as underlying implementation
   - Wrap it with additional upsert functionality
   - Handle connection pooling and error recovery

5. Error handling:
   - Raise IndexingError for database connection failures
   - Raise RetrievalError for query failures
   - Comprehensive logging for debugging

6. Unit tests in tests/test_vector_store.py:
   - Mock PostgreSQL connections and operations
   - Test schema creation and table setup
   - Test upsert operations (insert new, update existing)
   - Test retrieval with various query scenarios
   - Test error handling for database failures

Focus on data consistency and atomic operations for the upsert functionality.
```

### Step 3.2: Text Processing and Chunking

**Objective**: Implement text chunking with provider-specific chunk sizes.

```
Create text processing system in src/yasrl/text_processor.py:

1. Implement TextProcessor class with:
   - __init__(chunk_size: int, chunk_overlap: int = 200): Configure chunking parameters
   - process_documents(documents: list[Document]) -> list[Node]: Process documents into chunks
   - create_nodes_from_text(text: str, metadata: dict) -> list[Node]: Create nodes from text
   - optimize_chunk_size(embedding_provider: str) -> int: Get optimal chunk size

2. Chunking strategy:
   - Use LlamaIndex SentenceSplitter for intelligent splitting
   - Respect sentence boundaries when possible
   - Maintain metadata throughout the chunking process
   - Ensure chunk overlap for context preservation

3. Metadata preservation:
   - Include original document source in each chunk
   - Add chunk position and total chunks for document
   - Preserve any existing metadata from source documents
   - Add timestamp for when chunk was created

4. Integration with embedding providers:
   - Get chunk size from EmbeddingProviderFactory
   - Configure chunking parameters based on provider
   - Validate that chunks don't exceed model limits

5. Performance optimization:
   - Process documents in batches for large datasets
   - Use async processing where possible
   - Memory-efficient chunking for large documents

6. Unit tests in tests/test_text_processor.py:
   - Test chunking with different chunk sizes
   - Verify metadata preservation through processing
   - Test with various document types and sizes
   - Test integration with embedding provider chunk sizes
   - Test error handling for malformed documents

Focus on intelligent chunking that preserves context while optimizing for retrieval quality.
```

### Phase 2 -> Step 3.3: Query Processing and Re-ranking

**Objective**: Implement query processing with retrieval and re-ranking capabilities.

```
Create query processing system in src/yasrl/query_processor.py:

1. Implement QueryProcessor class with:
   - __init__(vector_store: VectorStoreManager, embedding_provider: EmbeddingProvider, reranker_model: str = "BAAI/bge-reranker-base")
   - process_query(query: str, top_k: int = 10, rerank_top_k: int = 5) -> list[SourceChunk]
   - embed_query(query: str) -> list[float]: Convert query to embedding
   - rerank_chunks(query: str, chunks: list[SourceChunk]) -> list[SourceChunk]: Improve ranking

2. Query processing pipeline:
   - Embed the user query using the configured embedding model
   - Retrieve top_k chunks from vector store using similarity search
   - Apply re-ranking to improve relevance ordering
   - Return top rerank_top_k chunks as SourceChunk objects

3. Re-ranking implementation:
   - Use LlamaIndex re-ranking capabilities (CohereRerank or sentence-transformers)
   - Fall back gracefully if re-ranking fails
   - Preserve original similarity scores and add rerank scores
   - Handle cases where fewer chunks are available than requested

4. Error handling and logging:
   - Log query processing steps for debugging
   - Handle embedding model failures gracefully
   - Raise RetrievalError for vector store issues
   - Provide fallback behavior for re-ranking failures

5. Performance considerations:
   - Cache embeddings for repeated queries
   - Optimize vector store queries
   - Handle large result sets efficiently

6. Unit tests in tests/test_query_processor.py:
   - Mock vector store and embedding providers
   - Test full query processing pipeline
   - Test re-ranking with various chunk scenarios
   - Test error handling and fallback behavior
   - Verify SourceChunk creation and metadata

Focus on retrieval quality through effective re-ranking while maintaining performance.
```

## Phase 4: Main Pipeline Implementation

### Step 4.1: RAGPipeline Core Class - Initialization

**Objective**: Implement the main RAGPipeline class initialization and setup.

```
Create the main RAGPipeline class in src/yasrl/pipeline.py:

1. Implement RAGPipeline class __init__ method:
   - Accept llm: str and embed_model: str parameters
   - Initialize ConfigurationManager and validate required environment variables
   - Create LLM and embedding providers using the factory classes
   - Initialize VectorStoreManager with PostgreSQL connection
   - Set up TextProcessor with appropriate chunk size
   - Initialize QueryProcessor.
   - Do not implement re-ranking capabilities. This will be done in phase2.
   - Set up logging with configurable levels

2. Class structure and dependencies:
   - Store all initialized components as instance variables
   - Ensure proper dependency injection between components
   - Handle initialization errors gracefully with clear messages
   - Support async initialization for database connections

3. Configuration validation:
   - Validate that required environment variables exist for chosen providers
   - Test database connectivity during initialization
   - Verify embedding model and LLM accessibility
   - Raise ConfigurationError with specific guidance for missing setup

4. Logging setup:
   - Use Python's standard logging module
   - Default to INFO level, configurable via environment variable
   - Log initialization steps and component setup
   - Include performance timing for initialization steps

5. Resource management:
   - Implement proper cleanup methods for database connections
   - Support context manager protocol for resource management
   - Handle graceful shutdown of async components

6. Unit tests in tests/test_pipeline_init.py:
   - Mock all external dependencies and providers
   - Test successful initialization with valid configuration
   - Test ConfigurationError raising for invalid setups
   - Test component wiring and dependency injection
   - Test logging configuration and output
   - Do not run the tests. I will do it locally.

Focus on robust initialization with clear error messages and proper resource management.
```

### Step 4.2: RAGPipeline Index Method Implementation

**Objective**: Implement the document indexing functionality with upsert capabilities.

```
Implement the index method for RAGPipeline in src/yasrl/pipeline.py:

1. Add the async index method to RAGPipeline:
   - Signature: async def index(self, source: str | list[str]) -> None
   - Use DocumentLoader to detect source type and load documents
   - Process documents through TextProcessor to create chunks
   - Generate embeddings for all chunks using embedding provider
   - Store chunks in vector store with upsert logic

2. Indexing workflow implementation:
   - Detect source type (file, directory, URL, URL list)
   - Load documents using appropriate LlamaIndex loaders
   - Generate deterministic document_id for each source
   - Check existing chunks in vector store for each document_id
   - Delete existing chunks before inserting new ones (upsert logic)
   - Process documents into optimally-sized chunks
   - Generate embeddings for chunks in batches
   - Store chunks with metadata in PostgreSQL vector store

3. Progress tracking and logging:
   - Log indexing progress for large document sets
   - Provide feedback on number of documents processed
   - Track timing for performance monitoring
   - Log any documents that fail to process with reasons

4. Error handling and recovery:
   - Handle individual document failures without stopping entire process
   - Provide detailed error messages for debugging
   - Support partial success scenarios
   - Implement retry logic for transient failures

5. Batch processing optimization:
   - Process embeddings in configurable batch sizes
   - Use async operations for I/O-bound tasks
   - Memory-efficient processing for large document sets
   - Support cancellation for long-running operations

6. Unit tests in tests/test_pipeline_index.py:
   - Mock all external dependencies (loaders, embeddings, vector store)
   - Test indexing various source types (files, directories, URLs)
   - Test upsert logic (updating existing documents)
   - Test error handling for various failure scenarios
   - Test batch processing and performance optimizations
   - Do not run the tests. I will do it locally.

Focus on reliability, performance, and clear progress feedback for the indexing process.
```

### Step 4.3: RAGPipeline Ask Method Implementation

**Objective**: Implement the query processing and answer generation functionality.

```
Implement the ask method for RAGPipeline in src/yasrl/pipeline.py:

1. Add the async ask method to RAGPipeline:
   - Signature: async def ask(self, query: str, conversation_history: list[dict] | None = None) -> QueryResult
   - Use QueryProcessor to retrieve relevant chunks
   - Format prompt with conversation history and retrieved context
   - Generate answer using configured LLM
   - Return QueryResult with answer and source chunks

2. Query processing workflow:
   - Validate input query is not empty
   - Process query through QueryProcessor to get relevant chunks
   - Apply re-ranking to improve chunk relevance
   - Construct context from top-ranked chunks
   - Format conversation history if provided
   - Build comprehensive prompt for LLM

3. Prompt template design:
   - Create effective prompt template that includes:
     * Clear instruction for the AI assistant
     * Conversation history (if provided)
     * Retrieved context chunks with source attribution
     * Current user query
     * Instructions for citing sources
   - Ensure prompt fits within LLM context limits
   - Handle cases with no relevant chunks found

4. LLM interaction and response processing:
   - Send formatted prompt to configured LLM
   - Handle LLM API errors gracefully
   - Parse and validate LLM response
   - Extract answer and maintain source attribution
   - Create QueryResult with answer and source chunks

5. Conversation history handling:
   - Accept conversation history in standardized format
   - Integrate previous context appropriately
   - Limit conversation history to stay within token limits
   - Maintain conversation flow and context

6. Unit tests in tests/test_pipeline_ask.py:
   - Mock QueryProcessor, LLM provider, and all dependencies
   - Test successful query processing with various scenarios
   - Test conversation history integration
   - Test error handling for empty results and LLM failures
   - Test prompt formatting and token limit handling
   - Verify QueryResult creation and source attribution
   - Do not run the tests. I will do it locally.

Focus on generating high-quality answers with proper source attribution and conversation context.
```

### Step 4.4: Pipeline Integration and Public API

**Objective**: Complete the RAGPipeline implementation and expose the public API.

```
Complete the RAGPipeline implementation and set up the public API in src/yasrl/__init__.py:

1. Finalize RAGPipeline class in src/yasrl/pipeline.py:
   - Add context manager support (__aenter__, __aexit__)
   - Implement cleanup method for proper resource disposal
   - Add pipeline health check method
   - Add method to get pipeline statistics (indexed documents, etc.)
   - Ensure all methods have comprehensive docstrings with examples

2. Resource management:
   - Implement proper cleanup for database connections
   - Handle graceful shutdown of async operations
   - Support pipeline reinitialization if needed
   - Add connection pooling for database operations

3. Public API design in src/yasrl/__init__.py:
   - Export RAGPipeline as the main class
   - Export QueryResult and SourceChunk data models
   - Export custom exceptions for error handling
   - Set __version__ and __all__ for clean imports
   - Add module-level docstring with usage examples

4. Usage examples and documentation:
   - Create comprehensive docstrings with code examples
   - Document all parameters and return types
   - Include common usage patterns in docstrings
   - Add type hints throughout for IDE support

5. Integration testing setup:
   - Create integration test that exercises full pipeline
   - Mock all external services (LLM APIs, database)
   - Test complete workflow: initialize -> index -> ask
   - Verify end-to-end functionality works correctly

6. Unit tests in tests/test_pipeline_complete.py:
   - Test resource management and cleanup
   - Test context manager functionality
   - Test pipeline statistics and health checks
   - Test complete integration scenarios
   - Verify public API exports work correctly
   - Do not run the tests. I will do it locally.

Focus on a clean, intuitive API that handles resource management automatically and provides clear documentation.
```

## Phase 5: Evaluation Framework

### Step 5.1: Base Evaluator Interface

**Objective**: Create extensible evaluation framework with abstract base class.

```
Create the evaluation framework in src/yasrl/evaluation/base.py:

1. Define abstract BaseEvaluator class:
   - Abstract method evaluate(questions: list[str], expected_answers: list[str], pipeline: RAGPipeline) -> dict
   - Abstract method evaluate_single(question: str, expected_answer: str, pipeline: RAGPipeline) -> dict
   - Abstract property supported_metrics -> list[str]: Return list of metrics this evaluator supports
   - Method generate_report(results: dict) -> str: Generate human-readable evaluation report

2. Evaluation result format:
   - Standardize evaluation result dictionary structure
   - Include overall scores and per-question breakdowns
   - Support multiple evaluation metrics per evaluator
   - Include timing information and metadata

3. Error handling for evaluation:
   - Define EvaluationError for evaluation-specific failures
   - Handle pipeline errors gracefully during evaluation
   - Provide partial results if some evaluations fail
   - Log evaluation progress and any issues

4. Base evaluation utilities:
   - Common helper methods for result aggregation
   - Utility functions for metric calculation
   - Support for async evaluation of multiple questions
   - Configuration validation for evaluation parameters

5. Documentation and type hints:
   - Comprehensive docstrings explaining evaluation interface
   - Type hints for all methods and return values
   - Examples of how to implement custom evaluators
   - Guidelines for evaluation best practices

6. Unit tests in tests/test_evaluation_base.py:
   - Test abstract class cannot be instantiated directly
   - Test base utility methods work correctly
   - Test error handling for evaluation failures
   - Test result format validation
   - Mock implementations to test interface compliance
   - Do not run the tests. I will do it locally.

Focus on creating a flexible interface that can support various evaluation libraries and metrics.
```

### Step 5.2: RAGAS Evaluator Implementation FAILED

**Objective**: Implement RAGAS-based evaluator as first concrete evaluation tool.

```
Create RAGAS evaluator implementation in src/yasrl/evaluation/ragas_evaluator.py:

1. Implement RagasEvaluator class inheriting from BaseEvaluator:
   - __init__(metrics: list[str] = None): Initialize with specific RAGAS metrics
   - evaluate() method: Run RAGAS evaluation on question set
   - evaluate_single() method: Evaluate single question-answer pair
   - Integration with RAGAS library for standard RAG metrics

2. RAGAS metrics integration:
   - Support key RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall
   - Handle RAGAS dataset format conversion
   - Configure RAGAS with appropriate LLM and embedding models
   - Map RAGAS results to standardized evaluation format

3. Data preparation for RAGAS:
   - Convert pipeline Q&A format to RAGAS dataset format
   - Extract contexts from QueryResult source chunks
   - Handle missing ground truth gracefully
   - Support both synthetic and human-labeled evaluation datasets

4. Performance and reliability:
   - Implement batch evaluation for efficiency
   - Add retry logic for LLM API failures during evaluation
   - Cache evaluation results to avoid re-computation
   - Support async evaluation for large datasets

5. Configuration and customization:
   - Allow custom RAGAS metric configuration
   - Support different LLM models for evaluation
   - Configurable evaluation parameters (batch size, timeouts)
   - Integration with pipeline's existing LLM providers

6. Unit tests in tests/test_evaluation_ragas.py:
   - Mock RAGAS library and its evaluation functions
   - Test metric calculation with sample data
   - Test data format conversion between pipeline and RAGAS
   - Test error handling for RAGAS failures
   - Test batch evaluation and performance optimizations
   - Do not run the tests. I will do it locally.

Focus on robust integration with RAGAS while maintaining the flexibility of the base evaluation interface.
```

### Step 5.3: Evaluation CLI and Utilities

**Objective**: Create command-line tools and utilities for running evaluations.

```
Create evaluation utilities and CLI in src/yasrl/evaluation/cli.py:

1. Implement evaluation CLI commands:
   - Command to run evaluation on a dataset file
   - Command to generate synthetic evaluation questions from indexed documents
   - Command to compare multiple pipeline configurations
   - Command to export evaluation results in various formats

2. Dataset handling utilities:
   - Support common evaluation dataset formats (JSON, CSV)
   - Validation for required fields (questions, expected_answers, contexts)
   - Conversion between different dataset formats
   - Generation of evaluation datasets from existing documents

3. Evaluation pipeline integration:
   - Easy setup of pipeline for evaluation purposes
   - Configuration management for evaluation runs
   - Support for evaluating multiple configurations in parallel
   - Integration with existing pipeline initialization

4. Result reporting and visualization:
   - Generate comprehensive evaluation reports
   - Export results to JSON, CSV, and HTML formats
   - Create summary statistics and metric comparisons
   - Support for custom report templates

5. CLI argument parsing and validation:
   - Use argparse for command-line interface
   - Validate file paths and configuration parameters
   - Provide helpful error messages for invalid inputs
   - Support configuration files for complex evaluation setups

6. Unit tests in tests/test_evaluation_cli.py:
   - Mock file system operations and pipeline creation
   - Test CLI argument parsing and validation
   - Test dataset loading and format conversion
   - Test evaluation execution and result generation
   - Test error handling for invalid configurations
   - Do not run the tests. I will do it locally.

Focus on creating user-friendly tools that make evaluation accessible and automate common evaluation workflows.
```

## Phase 2 Phase 6: Advanced Features and Optimization

### Step 6.1: Logging and Monitoring System

**Objective**: Implement comprehensive logging and monitoring for production use.

```
Create advanced logging and monitoring system in src/yasrl/logging.py:

1. Implement yasrlLogger class:
   - Configurable log levels (DEBUG, INFO, WARNING, ERROR)
   - Structured logging with JSON format option
   - Context-aware logging with operation IDs
   - Integration with Python's standard logging module

2. Performance monitoring:
   - Track timing for indexing operations
   - Monitor query processing latency
   - Log embedding generation performance
   - Track vector store operation metrics

3. Operational logging:
   - Log pipeline initialization and configuration
   - Track document processing progress and failures
   - Monitor LLM API usage and costs
   - Log database operations and connection health

4. Error tracking and debugging:
   - Detailed error logging with stack traces
   - Context preservation for error investigation
   - Integration with exception handling system
   - Support for log aggregation systems

5. Configuration and customization:
   - Environment variable configuration (yasrl_LOG_LEVEL, yasrl_LOG_FORMAT)
   - Support for multiple log handlers (file, console, remote)
   - Configurable log rotation and retention
   - Integration with external monitoring systems

6. Unit tests in tests/test_logging.py:
   - Test log level configuration and filtering
   - Test structured logging format
   - Test performance metric collection
   - Test error logging and context preservation
   - Mock external logging systems for integration tests
   - Do not run the tests. I will do it locally.

Focus on providing production-ready logging that aids in debugging and monitoring pipeline performance.
```

### Step 6.2: Caching and Performance Optimization

**Objective**: Implement caching strategies to improve pipeline performance.

```
Create caching system for performance optimization in src/yasrl/cache.py:

1. Implement CacheManager class:
   - Support multiple cache backends (memory, Redis, file-based)
   - Configurable cache TTL and eviction policies
   - Async-compatible caching operations
   - Cache key generation and management

2. Embedding caching:
   - Cache document embeddings to avoid recomputation
   - Cache query embeddings for repeated queries
   - Intelligent cache invalidation for updated documents
   - Memory-efficient storage for large embedding vectors

3. Query result caching:
   - Cache query results for identical queries
   - Support cache invalidation when index changes
   - Configurable cache size and retention policies
   - Consider conversation history in cache keys

4. Provider instance caching:
   - Cache LLM and embedding provider instances
   - Avoid repeated initialization overhead
   - Support provider configuration changes
   - Memory management for cached instances

5. Performance optimization strategies:
   - Batch processing for embeddings
   - Connection pooling for database operations
   - Async operation optimization
   - Memory usage monitoring and optimization

6. Unit tests in tests/test_cache.py:
   - Test different cache backend implementations
   - Test cache hit/miss scenarios
   - Test cache invalidation logic
   - Test performance improvements with caching
   - Test memory usage and resource management
   - Do not run the tests. I will do it locally.

Focus on significant performance improvements while maintaining data consistency and freshness.
```

### Step 6.3: Advanced Configuration and Customization

**Objective**: Provide advanced configuration options for power users.

```
Create advanced configuration system in src/yasrl/advanced_config.py:

1. Implement AdvancedConfig class:
   - Support for configuration files (YAML, JSON)
   - Runtime configuration updates
   - Configuration validation and type checking
   - Environment-specific configuration profiles

2. Advanced pipeline options:
   - Custom chunk size and overlap configuration
   - Configurable retrieval parameters (top_k, rerank_top_k)
   - Custom prompt templates for answer generation
   - Advanced embedding and LLM model parameters

3. Database and vector store configuration:
   - Custom PostgreSQL connection parameters
   - Vector index configuration options
   - Batch size and connection pooling settings
   - Custom table names and schema options

4. Provider-specific configurations:
   - OpenAI model parameters and API settings
   - Gemini configuration and quota management
   - Ollama custom model and host configuration
   - Custom embedding model parameters

5. Performance and scaling options:
   - Async operation configuration
   - Memory usage limits and monitoring
   - Timeout and retry configurations
   - Load balancing and failover options

6. Unit tests in tests/test_advanced_config.py:
   - Test configuration file parsing and validation
   - Test runtime configuration updates
   - Test provider-specific configuration options
   - Test performance setting impacts
   - Test configuration error handling
   - Do not run the tests. I will do it locally.

Focus on providing flexibility for advanced users while maintaining the simplicity of the basic API.
```

## Phase 7: Testing and Quality Assurance

### Step 7.1: Comprehensive Integration Testing

**Objective**: Create thorough integration tests that validate end-to-end functionality.

```
Create comprehensive integration testing in tests/integration/:

1. Full pipeline integration tests:
   - Test complete workflow: initialize -> index -> ask
   - Use mock services for all external dependencies
   - Validate data flow through all components
   - Test error propagation and handling

2. Provider integration testing:
   - Test all LLM provider implementations with mocked APIs
   - Test all embedding provider implementations
   - Validate provider switching and configuration
   - Test provider error handling and fallbacks

3. Database integration testing:
   - Test PostgreSQL vector store operations with test database
   - Validate upsert logic with real database operations
   - Test concurrent access and transaction handling
   - Test schema creation and migration

4. Performance integration testing:
   - Test with large document sets (mocked)
   - Validate memory usage patterns
   - Test concurrent indexing and querying
   - Benchmark operation timing

5. Configuration integration testing:
   - Test various configuration combinations
   - Validate environment variable handling
   - Test configuration error scenarios
   - Test runtime configuration changes

6. Error scenario testing:
   - Test network failure scenarios
   - Test partial failures and recovery
   - Test resource exhaustion scenarios
   - Validate error propagation and logging

Use pytest fixtures for setup/teardown and mock external services comprehensively.
```

### Step 7.2: Performance Testing and Benchmarks

**Objective**: Establish performance baselines and identify optimization opportunities.

```
Create performance testing framework in tests/performance/:

1. Benchmarking infrastructure:
   - Create performance test framework using pytest-benchmark
   - Establish baseline metrics for key operations
   - Support for continuous performance monitoring
   - Performance regression detection

2. Indexing performance tests:
   - Benchmark document processing speed
   - Test embedding generation performance
   - Measure vector store insertion rates
   - Test batch processing efficiency

3. Query performance tests:
   - Benchmark query processing latency
   - Test retrieval and re-ranking performance
   - Measure LLM response times
   - Test concurrent query handling

4. Memory usage testing:
   - Profile memory usage during large indexing operations
   - Test memory efficiency of caching systems
   - Validate memory cleanup and garbage collection
   - Test for memory leaks in long-running operations

5. Scalability testing:
   - Test performance with increasing document counts
   - Validate query performance with large indexes
   - Test concurrent user scenarios
   - Test resource usage scaling patterns

6. Performance optimization validation:
   - Verify caching improves performance
   - Test batch processing benefits
   - Validate async operation efficiency
   - Measure impact of configuration changes

Create automated performance monitoring that can be run in CI/CD pipelines.
```

### Step 7.3: Documentation and Examples

**Objective**: Create comprehensive documentation and practical examples.

```
Create comprehensive documentation system:

1. Setup Sphinx documentation in docs/:
   - Configure Sphinx with modern theme (sphinx-rtd-theme)
   - Set up automatic API documentation generation
   - Configure documentation build process
   - Set up documentation hosting preparation

2. API documentation:
   - Comprehensive docstring coverage for all public methods
   - Type hint documentation and examples
   - Error handling documentation
   - Configuration option documentation

3. User guides and tutorials:
   - Quick start guide with simple examples
   - Advanced configuration guide
   - Performance optimization guide
   - Troubleshooting and FAQ section

4. Code examples and recipes:
   - Basic RAG pipeline setup example
   - Multi-source indexing examples
   - Conversation-aware query examples
   - Evaluation workflow examples
   - Custom configuration examples

5. Development documentation:
   - Architecture overview and design decisions
   - Contributing guidelines and development setup
   - Testing strategy and best practices
   - Release process and versioning

6. Example applications:
   - Create example/demo_basic.py: Simple document Q&A
   - Create example/demo_advanced.py: Advanced configuration showcase
   - Create example/demo_evaluation.py: Evaluation workflow
   - Create example/demo_conversation.py: Conversational RAG

Ensure all documentation is accurate, up-to-date, and includes working code examples.
```

## Phase 8: Packaging and Distribution

### Step 8.1: Final Package Configuration

**Objective**: Finalize packaging configuration for distribution.

```
Complete package configuration for distribution:

1. Finalize pyproject.toml:
   - Complete package metadata (description, keywords, classifiers)
   - Finalize dependency specifications with version constraints
   - Configure optional dependency groups properly
   - Set up build system configuration

2. Distribution preparation:
   - Create MANIFEST.in for additional file inclusion
   - Configure package data and resource files
   - Set up proper package versioning strategy
   - Prepare license file and legal documentation

3. Quality assurance tools:
   - Configure pre-commit hooks for code quality
   - Set up GitHub Actions for CI/CD
   - Configure automated testing on multiple Python versions
   - Set up code coverage reporting

4. Security and dependency management:
   - Security scanning for dependencies
   - Dependency vulnerability checking
   - Pin dependency versions for stability
   - Configure automated dependency updates

5. Documentation build:
   - Automate documentation building and deployment
   - Configure documentation versioning
   - Set up documentation hosting
   - Test documentation build process

6. Release preparation:
   - Create release workflow and checklist
   - Configure automated PyPI publishing
   - Set up semantic versioning and changelog
   - Prepare release announcement templates

Focus on creating a professional, maintainable package ready for public distribution.
```

### Step 8.2: Final Testing and Validation

**Objective**: Comprehensive final testing before release.

```
Perform final testing and validation:

1. End-to-end testing:
   - Test installation from PyPI test instance
   - Validate all optional dependency combinations
   - Test on multiple Python versions (3.8+)
   - Test on different operating systems

2. Documentation validation:
   - Test all code examples in documentation
   - Verify documentation completeness
   - Test documentation build process
   - Validate API documentation accuracy

3. Performance validation:
   - Run full performance test suite
   - Validate performance meets requirements
   - Test resource usage under load
   - Verify no performance regressions

4. Security testing:
   - Security scan of all dependencies
   - Validate input sanitization
   - Test configuration security
   - Verify no credential leakage

5. User experience testing:
   - Test installation experience
   - Validate error messages are helpful
   - Test common usage scenarios
   - Verify configuration process is smooth

6. Release readiness checklist:
   - All tests passing
   - Documentation complete and accurate
   - Performance benchmarks met
   - Security requirements satisfied
   - Package metadata complete

Create comprehensive release validation checklist and automated testing workflow.
```

## Implementation Notes

### Development Best Practices

1. **Test-Driven Development**: Each step should begin with writing tests before implementation
2. **Incremental Progress**: Each step should build on previous work without breaking existing functionality
3. **Clear Documentation**: Every public method should have comprehensive docstrings with examples
4. **Error Handling**: Implement robust error handling with clear, actionable error messages
5. **Performance Consideration**: Consider performance implications of each implementation choice
6. **Security First**: Never accept credentials as parameters; always use environment variables

### Key Integration Points

1. **Provider Factories**: Ensure all providers implement consistent interfaces
2. **Configuration Management**: Centralize all configuration logic for maintainability
3. **Error Propagation**: Ensure errors bubble up with sufficient context for debugging
4. **Async Operations**: Maintain async compatibility throughout the pipeline
5. **Resource Management**: Implement proper cleanup and resource management
6. **Logging Integration**: Ensure consistent logging throughout all components

### Testing Strategy

1. **Unit Tests**: Test individual components in isolation with comprehensive mocking
2. **Integration Tests**: Test component interactions with minimal external dependencies
3. **Performance Tests**: Establish baselines and monitor for regressions
4. **End-to-End Tests**: Validate complete workflows with mocked external services

### Quality Gates

- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Code coverage > 90%
- [ ] Documentation complete
- [ ] Security scan clean
- [ ] Type checking passes
- [ ] Linting passes

This blueprint provides a comprehensive, step-by-step approach to building the yasrl library with strong testing, clear documentation, and robust error handling throughout.
