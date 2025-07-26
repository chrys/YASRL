"""
Configuration data models for YASRL library.
Defines structured configuration classes with validation and type safety.
"""
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from yasrl.exceptions import ConfigurationError


@dataclass
class LLMModelConfig:
    """
    Configuration for Large Language Models.
    
    Attributes:
        provider: LLM provider name ("openai", "gemini", "ollama")
        model_name: Specific model to use (e.g., "gpt-4o-mini")
        temperature: Sampling temperature for response generation (0.0-2.0)
        max_tokens: Maximum tokens in generated response
        timeout: Request timeout in seconds
        api_version: API version to use (optional)
        custom_params: Additional provider-specific parameters
    
    Example:
        >>> config = LLMModelConfig(
        ...     provider="openai",
        ...     model_name="gpt-4o-mini",
        ...     temperature=0.7
        ... )
    """
    provider: str
    model_name: str = "models/gemini-2.0-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 30
    api_version: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate LLM configuration parameters."""
        if self.provider not in ["openai", "gemini", "ollama"]:
            raise ConfigurationError(f"Unsupported LLM provider: {self.provider}")
        
        if not self.model_name or not self.model_name.strip():
            raise ConfigurationError("LLM model_name cannot be empty")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError("LLM temperature must be between 0.0 and 2.0")
        
        if self.max_tokens <= 0:
            raise ConfigurationError("LLM max_tokens must be positive")
        
        if self.timeout <= 0:
            raise ConfigurationError("LLM timeout must be positive")


@dataclass
class EmbeddingModelConfig:
    """
    Configuration for embedding models.
    
    Attributes:
        provider: Embedding provider name ("openai", "gemini", "opensource")
        model_name: Specific embedding model to use
        chunk_size: Optimal chunk size for this embedding model
        batch_size: Number of texts to embed in a single batch
        timeout: Request timeout in seconds
        custom_params: Additional provider-specific parameters
    
    Example:
        >>> config = EmbeddingModelConfig(
        ...     provider="openai",
        ...     model_name="text-embedding-3-small",
        ...     chunk_size=1024
        ... )
    """
    provider: str
    model_name: str
    chunk_size: int = 1024
    batch_size: int = 100
    timeout: int = 30
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate embedding configuration parameters."""
        if self.provider not in ["openai", "gemini", "opensource"]:
            raise ConfigurationError(f"Unsupported embedding provider: {self.provider}")
        
        if not self.model_name or not self.model_name.strip():
            raise ConfigurationError("Embedding model_name cannot be empty")
        
        if self.chunk_size <= 0:
            raise ConfigurationError("Embedding chunk_size must be positive")
        
        if self.batch_size <= 0:
            raise ConfigurationError("Embedding batch_size must be positive")
        
        if self.timeout <= 0:
            raise ConfigurationError("Embedding timeout must be positive")


@dataclass
class DatabaseConfig:
    """
    Configuration for database connections.
    
    Attributes:
        postgres_uri: PostgreSQL connection string
        table_prefix: Prefix for database table names
        connection_pool_size: Size of connection pool
        vector_dimensions: Dimensionality of embedding vectors
        index_type: Type of vector index to use
    
    Example:
        >>> config = DatabaseConfig(
        ...     postgres_uri="postgres://user:pass@localhost/db",
        ...     table_prefix="yasrl"
        ... )
    """
    postgres_uri: str
    table_prefix: str = "yasrl"
    connection_pool_size: int = 10
    vector_dimensions: int = 768
    index_type: str = "ivfflat"
    
    def validate(self) -> None:
        """Validate database configuration parameters."""
        if not self.postgres_uri or not self.postgres_uri.strip():
            raise ConfigurationError("Database postgres_uri cannot be empty")
        
        if not self.postgres_uri.startswith(("postgres://", "postgresql://")):
            raise ConfigurationError("Database postgres_uri must be a valid PostgreSQL URI")
        
        if not self.table_prefix or not self.table_prefix.strip():
            raise ConfigurationError("Database table_prefix cannot be empty")
        
        if self.connection_pool_size <= 0:
            raise ConfigurationError("Database connection_pool_size must be positive")
        
        if self.vector_dimensions <= 0:
            raise ConfigurationError("Database vector_dimensions must be positive")
        
        if self.index_type not in ["ivfflat", "hnsw"]:
            raise ConfigurationError(f"Unsupported database index_type: {self.index_type}")


@dataclass
class AdvancedConfig:
    """
    Complete advanced configuration for YASRL pipeline.
    
    Attributes:
        llm: Large Language Model configuration
        embedding: Embedding model configuration
        database: Database connection configuration
        retrieval_top_k: Number of chunks to retrieve from vector store
        rerank_top_k: Number of chunks after re-ranking
        chunk_overlap: Overlap between text chunks in characters
        batch_processing_size: Size of batches for processing operations
        cache_enabled: Whether to enable caching
        async_processing: Whether to use async processing
        log_level: Logging level
        structured_logging: Whether to use structured (JSON) logging
    
    Example:
        >>> config = AdvancedConfig(
        ...     llm=LLMModelConfig(provider="openai", model_name="gpt-4o-mini"),
        ...     embedding=EmbeddingModelConfig(provider="openai", model_name="text-embedding-3-small"),
        ...     database=DatabaseConfig(postgres_uri="postgres://user:pass@localhost/db")
        ... )
    """
    llm: LLMModelConfig
    embedding: EmbeddingModelConfig
    database: DatabaseConfig
    
    # Pipeline settings
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    chunk_overlap: int = 200
    
    # Performance settings
    batch_processing_size: int = 50
    cache_enabled: bool = True
    async_processing: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    structured_logging: bool = False
    
    # API Keys and service endpoints
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_host: Optional[str] = None
    
    def validate(self) -> None:
        """Validate complete configuration."""
        # Validate nested configurations
        self.llm.validate()
        self.embedding.validate()
        self.database.validate()
        
        # Validate API keys based on providers
        if self.llm.provider == "gemini" and not self.google_api_key:
            raise ConfigurationError("GOOGLE_API_KEY is required for Gemini LLM provider")
        
        if self.embedding.provider == "gemini" and not self.google_api_key:
            raise ConfigurationError("GOOGLE_API_KEY is required for Gemini embedding provider")
        
        if self.llm.provider == "openai" and not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI LLM provider")
        
        if self.embedding.provider == "openai" and not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI embedding provider")
        
        if self.llm.provider == "ollama" and not self.ollama_host:
            raise ConfigurationError("OLLAMA_HOST is required for Ollama LLM provider")
        
        # Validate pipeline settings
        if self.retrieval_top_k <= 0:
            raise ConfigurationError("retrieval_top_k must be positive")
        
        if self.rerank_top_k <= 0:
            raise ConfigurationError("rerank_top_k must be positive")
        
        if self.rerank_top_k > self.retrieval_top_k:
            raise ConfigurationError("rerank_top_k cannot be greater than retrieval_top_k")
        
        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap cannot be negative")
        
        # Validate performance settings
        if self.batch_processing_size <= 0:
            raise ConfigurationError("batch_processing_size must be positive")
        
        # Validate logging settings
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of: {valid_log_levels}")