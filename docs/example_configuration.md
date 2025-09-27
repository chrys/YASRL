# Example YASRL Configuration File
# Copy this to yasrl.yaml and customize for your needs

llm:
  provider: "openai"  # "openai", "gemini", "ollama"
  model_name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  api_version: null
  custom_params:
    frequency_penalty: 0.0
    presence_penalty: 0.0

embedding:
  provider: "openai"  # "openai", "gemini", "opensource"
  model_name: "text-embedding-3-small"
  chunk_size: 1024
  batch_size: 100
  timeout: 30
  custom_params: {}

database:
  # postgres_uri should be set via POSTGRES_URI environment variable
  table_prefix: "yasrl"
  connection_pool_size: 10
  vector_dimensions: 768
  index_type: "ivfflat"  # "ivfflat" or "hnsw"

# Retrieval settings
retrieval_top_k: 10
rerank_top_k: 5
chunk_overlap: 200

# Performance settings
batch_processing_size: 50
cache_enabled: true
async_processing: true

# Logging settings
log_level: "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
structured_logging: false


{
  "llm": {
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 4096,
    "timeout": 30,
    "api_version": null,
    "custom_params": {
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    }
  },
  "embedding": {
    "provider": "openai",
    "model_name": "text-embedding-3-small",
    "chunk_size": 1024,
    "batch_size": 100,
    "timeout": 30,
    "custom_params": {}
  },
  "database": {
    "table_prefix": "yasrl",
    "connection_pool_size": 10,
    "vector_dimensions": 1536,
    "index_type": "ivfflat"
  },
  "retrieval_top_k": 10,
  "rerank_top_k": 5,
  "chunk_overlap": 200,
  "batch_processing_size": 50,
  "cache_enabled": true,
  "async_processing": true,
  "log_level": "INFO",
  "structured_logging": false
}