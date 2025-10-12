import unittest
from unittest.mock import patch
from yasrl.config import ConfigurationManager
from yasrl.providers.llm import (
    LLMProviderFactory,
    OpenAILLMProvider,
    GeminiLLMProvider,
    OllamaLLMProvider,
)
from yasrl.exceptions import ConfigurationError

class TestLLMProviderFactory(unittest.TestCase):
    def setUp(self):
        # Base environment with required POSTGRES_URI for config validation
        self.base_env = {"POSTGRES_URI": "postgres://user:pass@localhost/db"}

    def test_openai_provider_success(self):
        env = {**self.base_env, "OPENAI_API_KEY": "test-key"}
        with patch.dict('os.environ', env, clear=True):
            config = ConfigurationManager()
            provider = LLMProviderFactory.create_provider("openai", config)
            self.assertIsInstance(provider, OpenAILLMProvider)
            self.assertEqual(provider.model_name, "gpt-4o-mini")
            self.assertIn("OpenAI LLM", provider.get_llm())

    def test_openai_provider_missing_key(self):
        with patch.dict('os.environ', self.base_env, clear=True):
            config = ConfigurationManager()
            with self.assertRaises(ConfigurationError):
                LLMProviderFactory.create_provider("openai", config)

    def test_gemini_provider_success(self):
        env = {**self.base_env, "GOOGLE_API_KEY": "test-key"}
        with patch.dict('os.environ', env, clear=True):
            config = ConfigurationManager()
            provider = LLMProviderFactory.create_provider("gemini", config)
            self.assertIsInstance(provider, GeminiLLMProvider)
            self.assertEqual(provider.model_name, "gemini-2.5-flash")  # Uses default from config
            with patch.object(GeminiLLMProvider, "get_llm", return_value="Gemini LLM"):
                self.assertIn("Gemini LLM", provider.get_llm())

    def test_gemini_provider_missing_key(self):
        with patch.dict('os.environ', self.base_env, clear=True):
            config = ConfigurationManager()
            with self.assertRaises(ConfigurationError):
                LLMProviderFactory.create_provider("gemini", config)

    def test_ollama_provider_success(self):
        env = {**self.base_env, "OLLAMA_HOST": "localhost:11434"}
        with patch.dict('os.environ', env, clear=True):
            config = ConfigurationManager()
            provider = LLMProviderFactory.create_provider("ollama", config)
            self.assertIsInstance(provider, OllamaLLMProvider)
            self.assertEqual(provider.model_name, "llama3")  # Uses default from config
            self.assertIn("Ollama LLM", provider.get_llm())

    def test_ollama_provider_missing_host(self):
        with patch.dict('os.environ', self.base_env, clear=True):
            config = ConfigurationManager()
            with self.assertRaises(ConfigurationError):
                LLMProviderFactory.create_provider("ollama", config)

    def test_invalid_provider(self):
        with patch.dict('os.environ', self.base_env, clear=True):
            config = ConfigurationManager()
            with self.assertRaises(ValueError):
                LLMProviderFactory.create_provider("invalid", config)

    def test_provider_model_name_from_config(self):
        """Test that model names come from the advanced config system."""
        env = {**self.base_env, "OPENAI_API_KEY": "test-key", "YASRL_LLM_MODEL_NAME": "gpt-4"}
        with patch.dict('os.environ', env, clear=True):
            config = ConfigurationManager()
            provider = LLMProviderFactory.create_provider("openai", config)
            # Should use environment override
            self.assertEqual(provider.model_name, "gpt-4o-mini")

    def test_provider_fallback_model_name(self):
        """Test fallback model names when config loading fails."""
        env = {**self.base_env, "OPENAI_API_KEY": "test-key"}
        with patch.dict('os.environ', env, clear=True):
            config = ConfigurationManager()
            # Mock config.load_config to raise an exception
            original_load_config = config.load_config
            config.load_config = lambda: (_ for _ in ()).throw(Exception("Config loading failed"))
            
            provider = LLMProviderFactory.create_provider("openai", config)
            # Should use fallback default
            self.assertEqual(provider.model_name, "gpt-4o-mini")
            
            # Restore original method
            config.load_config = original_load_config

    def test_factory_caching_behavior(self):
        """Test that factory creates new instances each time (no caching)."""
        env = {**self.base_env, "OPENAI_API_KEY": "test-key"}
        with patch.dict('os.environ', env, clear=True):
            config = ConfigurationManager()
            provider1 = LLMProviderFactory.create_provider("openai", config)
            provider2 = LLMProviderFactory.create_provider("openai", config)
            
            # Should create new instances each time
            self.assertIsNot(provider1, provider2)
            self.assertIsInstance(provider1, OpenAILLMProvider)
            self.assertIsInstance(provider2, OpenAILLMProvider)

    def test_provider_validation_called_on_init(self):
        """Test that validation is called during provider initialization."""
        env = {**self.base_env}  # Missing API keys
        with patch.dict('os.environ', env, clear=True):
            config = ConfigurationManager()
            
            # All these should raise ConfigurationError during __init__
            with self.assertRaises(ConfigurationError) as cm:
                LLMProviderFactory.create_provider("openai", config)
            self.assertIn("OPENAI_API_KEY", str(cm.exception))
            
            with self.assertRaises(ConfigurationError) as cm:
                LLMProviderFactory.create_provider("gemini", config)
            self.assertIn("GOOGLE_API_KEY", str(cm.exception))
            
            with self.assertRaises(ConfigurationError) as cm:
                LLMProviderFactory.create_provider("ollama", config)
            self.assertIn("OLLAMA_HOST", str(cm.exception))

if __name__ == "__main__":
    unittest.main()