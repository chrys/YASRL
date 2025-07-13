import os
from abc import ABC, abstractmethod
from typing import Any
from yasrl.config import ConfigurationManager
from yasrl.exceptions import ConfigurationError

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    @abstractmethod
    def get_llm(self) -> Any:
        """
        Returns a configured LlamaIndex LLM instance.
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validates required environment variables for the provider.
        Raises ConfigurationError if missing.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the model name being used.
        """
        pass

class OpenAILLMProvider(LLMProvider):
    """
    LLM provider for OpenAI (model: gpt-4o-mini).
    """
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.validate_config()

    def get_llm(self) -> Any:
        # Placeholder for actual LlamaIndex OpenAI LLM instantiation
        return f"OpenAI LLM: {self.model_name}"

    def validate_config(self) -> None:
        # Check environment variable directly for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI LLM provider.")

    @property
    def model_name(self) -> str:
        try:
            config_obj = self.config.load_config()
            return config_obj.llm.model_name
        except:
            return "gpt-4o-mini"  # fallback default

class GeminiLLMProvider(LLMProvider):
    """
    LLM provider for Gemini.
    """
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.validate_config()

    def get_llm(self) -> Any:
        # Placeholder for actual LlamaIndex Gemini LLM instantiation
        return f"Gemini LLM: {self.model_name}"
    
    def validate_config(self) -> None:
        # Check environment variable directly for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigurationError("GOOGLE_API_KEY is required for Gemini LLM provider.")

    @property
    def model_name(self) -> str:
        try:
            config_obj = self.config.load_config()
            return config_obj.llm.model_name
        except:
            return "gemini-pro"  # fallback default

class OllamaLLMProvider(LLMProvider):
    """
    LLM provider for Ollama.
    """
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.validate_config()

    def get_llm(self) -> Any:
        # Placeholder for actual LlamaIndex Ollama LLM instantiation
        return f"Ollama LLM: {self.model_name}"
        
    def validate_config(self) -> None:
        # Check environment variable directly for Ollama host
        ollama_host = os.getenv("OLLAMA_HOST")
        if not ollama_host:
            raise ConfigurationError("OLLAMA_HOST is required for Ollama LLM provider.")

    @property
    def model_name(self) -> str:
        try:
            config_obj = self.config.load_config()
            return config_obj.llm.model_name
        except:
            return "llama3"  # fallback default

class LLMProviderFactory:
    """
    Factory for creating LLMProvider instances from string identifiers.
    """
    @staticmethod
    def create_provider(provider_name: str, config: ConfigurationManager) -> LLMProvider:
        """
        Creates an LLMProvider instance for the given provider name.
        Supported: "openai", "gemini", "ollama"
        Raises ValueError for unsupported providers.
        """
        name = provider_name.lower()
        if name == "openai":
            return OpenAILLMProvider(config)
        elif name == "gemini":
            return GeminiLLMProvider(config)
        elif name == "ollama":
            return OllamaLLMProvider(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")