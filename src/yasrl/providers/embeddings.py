from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from yasrl.exceptions import ConfigurationError

class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding model providers.
    """
    @abstractmethod
    def get_embedding_model(self) -> Any:
        """
        Returns a configured LlamaIndex embedding model instance.
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validates required environment variables for the provider.
        """
        pass

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """
        Returns the optimal chunk size for this embedding model.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the embedding model name.
        """
        pass

    def get_max_chunk_size(self) -> int:
        """
        Returns the optimal chunk size for this embedding model.
        """
        return self.chunk_size

class OpenAIEmbeddingProvider(EmbeddingProvider):
    CHUNK_SIZE = 1024
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, config):
        self.config = config
        self.validate_config()

    def get_embedding_model(self) -> Any:
        # Replace with actual LlamaIndex OpenAI embedding instantiation
        return f"MockOpenAIEmbedding(model={self.model_name})"

    def validate_config(self) -> None:
        api_key = getattr(self.config, "openai_api_key", None)
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI embedding provider.")

    @property
    def chunk_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def model_name(self) -> str:
        return getattr(self.config, "openai_embedding_model", self.DEFAULT_MODEL)

    def get_max_chunk_size(self) -> int:
        return self.CHUNK_SIZE

class GeminiEmbeddingProvider(EmbeddingProvider):
    CHUNK_SIZE = 1024
    DEFAULT_MODEL = "embedding-001"

    def __init__(self, config):
        self.config = config
        self.validate_config()

    def get_embedding_model(self) -> Any:
        # Replace with actual LlamaIndex Gemini embedding instantiation
        return f"MockGeminiEmbedding(model={self.model_name})"

    def validate_config(self) -> None:
        api_key = getattr(self.config, "google_api_key", None)
        if not api_key:
            raise ConfigurationError("GOOGLE_API_KEY is required for Gemini embedding provider.")

    @property
    def chunk_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def model_name(self) -> str:
        return getattr(self.config, "gemini_embedding_model", self.DEFAULT_MODEL)

    def get_max_chunk_size(self) -> int:
        return self.CHUNK_SIZE

class OpenSourceEmbeddingProvider(EmbeddingProvider):
    CHUNK_SIZE = 512
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, config):
        self.config = config
        # No required API key for open-source models

    def get_embedding_model(self) -> Any:
        # Replace with actual LlamaIndex open-source embedding instantiation
        return f"MockOpenSourceEmbedding(model={self.model_name})"

    def validate_config(self) -> None:
        # No required environment variables for open-source
        pass

    @property
    def chunk_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def model_name(self) -> str:
        return getattr(self.config, "opensource_embedding_model", self.DEFAULT_MODEL)

    def get_max_chunk_size(self) -> int:
        return self.CHUNK_SIZE

class EmbeddingProviderFactory:
    """
    Factory for creating and caching embedding provider instances.
    """
    _provider_map: Dict[str, Type[EmbeddingProvider]] = {
        "openai": OpenAIEmbeddingProvider,
        "gemini": GeminiEmbeddingProvider,
        "opensource": OpenSourceEmbeddingProvider,
    }
    _chunk_size_map: Dict[str, int] = {
        "openai": 1024,
        "gemini": 1024,
        "opensource": 512,
    }
    _instance_cache: Dict[str, EmbeddingProvider] = {}

    @staticmethod
    def create_provider(provider_name: str, config) -> EmbeddingProvider:
        name = provider_name.lower()
        if name not in EmbeddingProviderFactory._provider_map:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")
        if name in EmbeddingProviderFactory._instance_cache:
            return EmbeddingProviderFactory._instance_cache[name]
        provider_cls = EmbeddingProviderFactory._provider_map[name]
        instance = provider_cls(config)
        EmbeddingProviderFactory._instance_cache[name] = instance
        return instance

    @staticmethod
    def get_chunk_size(provider_name: str) -> int:
        name = provider_name.lower()
        if name not in EmbeddingProviderFactory._chunk_size_map:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")
        return EmbeddingProviderFactory._chunk_size_map[name]

    @staticmethod
    def get_embedding_provider(provider_name: str, config=None) -> EmbeddingProvider:
        """
        Returns an embedding provider instance (for compatibility with some usages).
        """
        return EmbeddingProviderFactory.create_provider(provider_name, config)