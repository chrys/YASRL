import pytest
from yasrl.providers.embeddings import (
    EmbeddingProviderFactory,
    OpenAIEmbeddingProvider,
    GeminiEmbeddingProvider,
    OpenSourceEmbeddingProvider,
)
from yasrl.exceptions import ConfigurationError

@pytest.fixture(autouse=True)
def clear_embedding_provider_cache():
    from yasrl.providers.embeddings import EmbeddingProviderFactory
    EmbeddingProviderFactory._instance_cache.clear()

class DummyConfig:
    openai_api_key = "test-openai-key"
    openai_embedding_model = "text-embedding-3-small"
    google_api_key = "test-google-key"
    gemini_embedding_model = "embedding-001"
    opensource_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

class MissingOpenAIConfig:
    openai_api_key = None
    google_api_key = "test-google-key"
    gemini_embedding_model = "embedding-001"
    opensource_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

class MissingGeminiConfig:
    google_api_key = None
    openai_api_key = "test-openai-key"
    openai_embedding_model = "text-embedding-3-small"
    opensource_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

class MinimalConfig:
    pass

def test_factory_creates_openai_provider_and_caches():
    config = DummyConfig()
    provider1 = EmbeddingProviderFactory.create_provider("openai", config)
    provider2 = EmbeddingProviderFactory.create_provider("openai", config)
    assert isinstance(provider1, OpenAIEmbeddingProvider)
    assert provider1 is provider2  # caching works
    assert provider1.model_name == config.openai_embedding_model
    assert provider1.chunk_size == 1024
    assert provider1.get_embedding_model() == f"MockOpenAIEmbedding(model={config.openai_embedding_model})"

def test_factory_creates_opensource_provider_and_caches():
    config = DummyConfig()
    provider1 = EmbeddingProviderFactory.create_provider("opensource", config)
    provider2 = EmbeddingProviderFactory.create_provider("opensource", config)
    assert isinstance(provider1, OpenSourceEmbeddingProvider)
    assert provider1 is provider2
    assert provider1.model_name == config.opensource_embedding_model
    assert provider1.chunk_size == 512
    assert provider1.get_embedding_model() == f"MockOpenSourceEmbedding(model={config.opensource_embedding_model})"

def test_factory_raises_for_unknown_provider():
    config = DummyConfig()
    with pytest.raises(ValueError):
        EmbeddingProviderFactory.create_provider("unknown", config)
    with pytest.raises(ValueError):
        EmbeddingProviderFactory.get_chunk_size("unknown")

def test_opensource_provider_does_not_require_api_key():
    config = MinimalConfig()
    provider = EmbeddingProviderFactory.create_provider("opensource", config)
    assert isinstance(provider, OpenSourceEmbeddingProvider)
    assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert provider.chunk_size == 512

def test_chunk_size_map():
    assert EmbeddingProviderFactory.get_chunk_size("openai") == 1024
    assert EmbeddingProviderFactory.get_chunk_size("gemini") == 1024
    assert EmbeddingProviderFactory.get_chunk_size("opensource") == 512
