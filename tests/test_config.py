import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yasrl.config import (
    ConfigurationManager, 
    AdvancedConfig, 
    LLMModelConfig, 
    EmbeddingModelConfig, 
    DatabaseConfig
)
from yasrl.exceptions import ConfigurationError


class TestConfigurationModels(unittest.TestCase):
    """Test configuration data models."""
    
    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        # Valid configuration
        config = LLMModelConfig(provider="openai", model_name="gpt-4o-mini")
        config.validate()  # Should not raise
        
        # Invalid provider
        config = LLMModelConfig(provider="invalid", model_name="model")
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Invalid temperature
        config = LLMModelConfig(provider="openai", model_name="model", temperature=3.0)
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Invalid max_tokens
        config = LLMModelConfig(provider="openai", model_name="model", max_tokens=-1)
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Empty model name
        config = LLMModelConfig(provider="openai", model_name="")
        with self.assertRaises(ConfigurationError):
            config.validate()
    
    def test_embedding_config_validation(self):
        """Test embedding configuration validation."""
        # Valid configuration
        config = EmbeddingModelConfig(provider="openai", model_name="text-embedding-3-small")
        config.validate()  # Should not raise
        
        # Invalid provider
        config = EmbeddingModelConfig(provider="invalid", model_name="model")
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Invalid chunk size
        config = EmbeddingModelConfig(provider="openai", model_name="model", chunk_size=-1)
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Invalid batch size
        config = EmbeddingModelConfig(provider="openai", model_name="model", batch_size=0)
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Empty model name
        config = EmbeddingModelConfig(provider="openai", model_name="")
        with self.assertRaises(ConfigurationError):
            config.validate()
    
    def test_database_config_validation(self):
        """Test database configuration validation."""
        # Valid configuration
        config = DatabaseConfig(postgres_uri="postgres://user:pass@localhost/db")
        config.validate()  # Should not raise
        
        # Valid postgresql:// URI
        config = DatabaseConfig(postgres_uri="postgresql://user:pass@localhost/db")
        config.validate()  # Should not raise
        
        # Invalid URI
        config = DatabaseConfig(postgres_uri="invalid-uri")
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Empty URI
        config = DatabaseConfig(postgres_uri="")
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Invalid connection pool size
        config = DatabaseConfig(postgres_uri="postgres://user:pass@localhost/db", connection_pool_size=0)
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Invalid vector dimensions
        config = DatabaseConfig(postgres_uri="postgres://user:pass@localhost/db", vector_dimensions=-1)
        with self.assertRaises(ConfigurationError):
            config.validate()
        
        # Invalid index type
        config = DatabaseConfig(postgres_uri="postgres://user:pass@localhost/db", index_type="invalid")
        with self.assertRaises(ConfigurationError):
            config.validate()
    

class TestConfigurationManager(unittest.TestCase):
    """Test configuration manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_config_path = Path(self.temp_dir) / "test_config.yaml"

    def tearDown(self):
        """Clean up test fixtures."""
        for file in Path(self.temp_dir).glob("*"):
            if file.is_file():
                file.unlink()
        Path(self.temp_dir).rmdir()

    def test_default_config_loading(self):
        """Test loading default configuration."""
        with patch.dict(os.environ, {
            "POSTGRES_URI": "postgres://user:pass@localhost/db",
            "OPENAI_API_KEY": "sk-test",
            "YASRL_LLM_PROVIDER": "openai",
            "YASRL_LLM_MODEL_NAME": "gpt-4o-mini",
            "YASRL_LLM_TEMPERATURE": "0.7",
            "YASRL_EMBEDDING_PROVIDER": "openai",
            "YASRL_EMBEDDING_MODEL_NAME": "text-embedding-3-small",
            "YASRL_RETRIEVAL_TOP_K": "10",
            "YASRL_RERANK_TOP_K": "5"
            }, clear=True):
            manager = ConfigurationManager()
            config = manager.load_config()
            self.assertIsInstance(config, AdvancedConfig)
            self.assertEqual(config.llm.provider, "openai")
            self.assertEqual(config.llm.model_name, "gpt-4o-mini")
            self.assertEqual(config.llm.temperature, 0.7)
            self.assertEqual(config.embedding.provider, "openai")
            self.assertEqual(config.embedding.model_name, "text-embedding-3-small")
            self.assertEqual(config.database.postgres_uri, "postgres://user:pass@localhost/db")
            self.assertEqual(config.retrieval_top_k, 10)
            self.assertEqual(config.rerank_top_k, 5)

    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
        llm:
            provider: "gemini"
            model_name: "gemini-pro"
            temperature: 0.5
            max_tokens: 2048
        embedding:
            provider: "gemini"
            model_name: "embedding-001"
            chunk_size: 512
        database:
            postgres_uri: "postgres://test:test@localhost/test"
            table_prefix: "test_yasrl"
        retrieval_top_k: 15
        rerank_top_k: 8
        cache_enabled: false
        log_level: "DEBUG"
        """
        with open(self.temp_config_path, 'w') as f:
            f.write(yaml_content)

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            manager = ConfigurationManager(config_file=self.temp_config_path)
            config = manager.load_config()
            self.assertEqual(config.llm.provider, "gemini")
            self.assertEqual(config.llm.model_name, "gemini-pro")
            self.assertEqual(config.llm.temperature, 0.5)
            self.assertEqual(config.llm.max_tokens, 2048)
            self.assertEqual(config.embedding.provider, "gemini")
            self.assertEqual(config.embedding.model_name, "embedding-001")
            self.assertEqual(config.embedding.chunk_size, 512)
            self.assertEqual(config.database.table_prefix, "test_yasrl")
            self.assertEqual(config.retrieval_top_k, 15)
            self.assertEqual(config.rerank_top_k, 8)
            self.assertEqual(config.cache_enabled, False)
            self.assertEqual(config.log_level, "DEBUG")

    def test_environment_override(self):
        """Test environment variable override."""
        yaml_content = """
            llm:
                provider: "openai"
                model_name: "gpt-4o-mini"
                temperature: 0.7
            database:
                postgres_uri: "postgres://test:test@localhost/test"
            retrieval_top_k: 10
            """
        with open(self.temp_config_path, 'w') as f:
            f.write(yaml_content)

        with patch.dict(os.environ, {
            "YASRL_LLM_PROVIDER": "gemini",
            "YASRL_LLM_TEMPERATURE": "0.9",
            "YASRL_LLM_MAX_TOKENS": "8192",
            "YASRL_EMBEDDING_PROVIDER": "gemini",
            "YASRL_RETRIEVAL_TOP_K": "20",
            "YASRL_CACHE_ENABLED": "false", 
            "GOOGLE_API_KEY": "test-key",
        }):
            manager = ConfigurationManager(config_file=self.temp_config_path)
            config = manager.load_config()
            self.assertEqual(config.llm.provider, "gemini")
            self.assertEqual(config.llm.temperature, 0.9)
            self.assertEqual(config.llm.max_tokens, 8192)
            self.assertEqual(config.embedding.provider, "gemini")
            self.assertEqual(config.retrieval_top_k, 20)
            self.assertEqual(config.cache_enabled, False)
            self.assertEqual(config.llm.model_name, "gpt-4o-mini")

    def test_custom_env_prefix(self):
        """Test custom environment variable prefix."""
        with patch.dict(os.environ, {
            "MYAPP_LLM_PROVIDER": "ollama",
            "MYAPP_LLM_MODEL": "llama3",
            "MYAPP_LLM_TEMPERATURE": "0.2",
            "MYAPP_EMBEDDING_PROVIDER": "opensource",
            "POSTGRES_URI": "postgres://user:pass@localhost/db", 
            "OLLAMA_HOST": "http://localhost:11434",
        }):
            manager = ConfigurationManager(env_prefix="MYAPP")
            config = manager.load_config()
            self.assertEqual(config.llm.provider, "ollama")
            self.assertEqual(config.llm.model_name, "llama3")
            self.assertEqual(config.llm.temperature, 0.2)
            self.assertEqual(config.embedding.provider, "opensource")

    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        # Non-existent file
        with self.assertRaises(ConfigurationError):
            ConfigurationManager(config_file="non_existent.yaml")

        # Invalid YAML content
        with open(self.temp_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        manager = ConfigurationManager(config_file=self.temp_config_path)
        with self.assertRaises(ConfigurationError):
            manager.load_config()

        # Unsupported file format
        unsupported_config = Path(self.temp_dir) / "config.txt"
        with open(unsupported_config, 'w') as f:
            f.write("some content")
        with self.assertRaises(ConfigurationError):
            ConfigurationManager(config_file=unsupported_config)

    def test_invalid_configuration_values(self):
        """Test handling of invalid configuration values."""
        yaml_content = """
            llm:
                provider: "invalid_provider"
                model_name: "test"
            database:
                postgres_uri: "postgres://user:pass@localhost/db"
            """
        with open(self.temp_config_path, 'w') as f:
            f.write(yaml_content)
        manager = ConfigurationManager(config_file=self.temp_config_path)
        with self.assertRaises(ConfigurationError):
            manager.load_config()

    def test_config_caching(self):
        """Test configuration caching."""
        with patch.dict(os.environ, {
            "POSTGRES_URI": "postgres://user:pass@localhost/db",
            "OPENAI_API_KEY": "sk-test",
            }):
            manager = ConfigurationManager()
            config1 = manager.load_config()
            config2 = manager.load_config()
            self.assertIs(config1, config2)
            manager.clear_cache()
            config3 = manager.load_config()
            self.assertIsNot(config1, config3)
            self.assertEqual(config1.llm.provider, config3.llm.provider)
            self.assertEqual(config1.embedding.provider, config3.embedding.provider)

    def test_config_sources(self):
        """Test getting configuration sources."""
        manager = ConfigurationManager()
        sources = manager.get_config_sources()
        self.assertEqual(len(sources), 2)
        self.assertIn("Environment variables", sources)
        self.assertIn("Default values", sources)
        with open(self.temp_config_path, 'w') as f:
            f.write("llm:\n  provider: openai\ndatabase:\n  postgres_uri: postgres://test/db")
        manager = ConfigurationManager(config_file=self.temp_config_path)
        sources = manager.get_config_sources()
        self.assertEqual(len(sources), 3)
        self.assertIn("Environment variables", sources)
        self.assertIn(f"Config file: {self.temp_config_path}", sources)
        self.assertIn("Default values", sources)

    def test_config_file_search_order(self):
        """Test configuration file search order."""
        local_config = Path("yasrl.yaml")
        try:
            with open(local_config, 'w') as f:
                f.write("""
                    llm:
                    provider: "local_file"
                    database:
                    postgres_uri: "postgres://local/db"
                    """)
            manager = ConfigurationManager()
            self.assertEqual(manager.config_file, local_config)
        finally:
            if local_config.exists():
                local_config.unlink()

    def test_missing_required_config(self):
        """Test handling of missing required configuration."""
        with patch.dict(os.environ, {}, clear=True):
            manager = ConfigurationManager()
            with self.assertRaises(ConfigurationError):
                manager.load_config()

if __name__ == "__main__":
    unittest.main()