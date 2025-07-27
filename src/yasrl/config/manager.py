"""
Simplified configuration manager for YASRL library.
Supports only YAML config files and environment variables.
"""
import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from yasrl.exceptions import ConfigurationError
from .models import AdvancedConfig, LLMModelConfig, EmbeddingModelConfig, DatabaseConfig

class ConfigurationManager:
    """
    Simplified configuration manager for YASRL.
    Loads configuration from:
    1. Environment variables (highest priority)
    2. Local YAML config file (yasrl.yaml/yasrl.yml)
    3. Global YAML config file (~/.yasrl/config.yaml or config.yml)
    4. Defaults (lowest priority)
    """

    def __init__(
        self, 
        config_file: Optional[Union[str, Path]] = None,
        env_prefix: str = "YASRL"
    ) -> None:
        self.env_prefix = env_prefix
        self.config_file = self._find_config_file(config_file)
        self._config_cache: Optional[AdvancedConfig] = None

    def _find_config_file(self, config_file: Optional[Union[str, Path]]) -> Optional[Path]:
        if config_file:
            path = Path(config_file)
            if not path.exists():
                raise ConfigurationError(f"Specified config file not found: {config_file}")
            if path.suffix not in ['.yaml', '.yml']:
                raise ConfigurationError(f"Unsupported config file format: {path.suffix}")
            return path

        # Search order: local -> global -> none
        search_paths = [
            Path("yasrl.yaml"),
            Path("yasrl.yml"),
            Path.home() / ".yasrl" / "config.yaml",
            Path.home() / ".yasrl" / "config.yml",
        ]
        for path in search_paths:
            if path.exists():
                return path
        return None

    def load_config(self) -> AdvancedConfig:
        if self._config_cache:
            return self._config_cache

        config_dict = self._get_default_config()

        # Override with file config
        if self.config_file:
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                config_dict = self._merge_configs(config_dict, file_config)
            except Exception as e:
                raise ConfigurationError(f"Failed to load config file {self.config_file}: {e}")

        # Override with environment variables
        env_config = self._load_env_config()
        config_dict = self._merge_configs(config_dict, env_config)

        try:
            self._config_cache = self._create_config_object(config_dict)
            return self._config_cache
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "llm": {
                "provider": "openai",
                "model_name": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 4096,
                "timeout": 30,
                "api_version": None,
                "custom_params": {}
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
                "postgres_uri": os.getenv("POSTGRES_URI", ""),
                "table_prefix": "yasrl",
                "connection_pool_size": 10,
                "vector_dimensions": 1536,
                "index_type": "ivfflat"
            },
            "retrieval_top_k": 10,
            "rerank_top_k": 5,
            "chunk_overlap": 200,
            "batch_processing_size": 50,
            "cache_enabled": True,
            "async_processing": True,
            "log_level": "INFO",
            "structured_logging": False,
            "log_output": "file",
            "log_file": "yasrl.log"
        }

    def _load_env_config(self) -> Dict[str, Any]:
        env_config = {}
        prefix = f"{self.env_prefix}_"
        env_mappings = {
            f"{prefix}LLM_PROVIDER": ["llm", "provider"],
            f"{prefix}LLM_MODEL": ["llm", "model_name"],
            f"{prefix}LLM_TEMPERATURE": ["llm", "temperature"],
            f"{prefix}LLM_MAX_TOKENS": ["llm", "max_tokens"],
            f"{prefix}LLM_TIMEOUT": ["llm", "timeout"],
            f"{prefix}LLM_API_VERSION": ["llm", "api_version"],
            f"{prefix}EMBEDDING_PROVIDER": ["embedding", "provider"],
            f"{prefix}EMBEDDING_MODEL": ["embedding", "model_name"],
            f"{prefix}CHUNK_SIZE": ["embedding", "chunk_size"],
            f"{prefix}BATCH_SIZE": ["embedding", "batch_size"],
            f"{prefix}EMBEDDING_TIMEOUT": ["embedding", "timeout"],
            f"{prefix}POSTGRES_URI": ["database", "postgres_uri"],
            f"{prefix}TABLE_PREFIX": ["database", "table_prefix"],
            f"{prefix}CONNECTION_POOL_SIZE": ["database", "connection_pool_size"],
            f"{prefix}VECTOR_DIMENSIONS": ["database", "vector_dimensions"],
            f"{prefix}INDEX_TYPE": ["database", "index_type"],
            f"{prefix}RETRIEVAL_TOP_K": ["retrieval_top_k"],
            f"{prefix}RERANK_TOP_K": ["rerank_top_k"],
            f"{prefix}CHUNK_OVERLAP": ["chunk_overlap"],
            f"{prefix}BATCH_PROCESSING_SIZE": ["batch_processing_size"],
            f"{prefix}CACHE_ENABLED": ["cache_enabled"],
            f"{prefix}ASYNC_PROCESSING": ["async_processing"],
            f"{prefix}LOG_LEVEL": ["log_level"],
            f"{prefix}STRUCTURED_LOGGING": ["structured_logging"],
            f"{prefix}LOG_OUTPUT": ["log_output"],
            f"{prefix}LOG_FILE": ["log_file"],
            "GOOGLE_API_KEY": ["google_api_key"],
            "OPENAI_API_KEY": ["openai_api_key"],
            "OLLAMA_HOST": ["ollama_host"],
            "POSTGRES_URI": ["database", "postgres_uri"],
        }
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(env_config, config_path, self._convert_env_value(value))
        return env_config

    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any) -> None:
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def _create_config_object(self, config_dict: Dict[str, Any]) -> AdvancedConfig:
        try:
            llm_config = LLMModelConfig(**config_dict["llm"])
            embedding_config = EmbeddingModelConfig(**config_dict["embedding"])
            database_config = DatabaseConfig(**config_dict["database"])
            config = AdvancedConfig(
                llm=llm_config,
                embedding=embedding_config,
                database=database_config,
                retrieval_top_k=config_dict["retrieval_top_k"],
                rerank_top_k=config_dict["rerank_top_k"],
                chunk_overlap=config_dict["chunk_overlap"],
                batch_processing_size=config_dict["batch_processing_size"],
                cache_enabled=config_dict["cache_enabled"],
                async_processing=config_dict["async_processing"],
                log_level=config_dict["log_level"],
                structured_logging=config_dict["structured_logging"],
                google_api_key=config_dict.get("google_api_key"),
                openai_api_key=config_dict.get("openai_api_key"),
                ollama_host=config_dict.get("ollama_host")
            )
            config.validate()
            return config
        except KeyError as e:
            raise ConfigurationError(f"Missing required configuration key: {e}")
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration type: {e}")

    def clear_cache(self) -> None:
        """Clear cached configuration to force reload on next access."""
        self._config_cache = None

    def get_config_sources(self) -> List[str]:
        sources = ["Environment variables"]
        if self.config_file:
            sources.append(f"Config file: {self.config_file}")
        sources.append("Default values")
        return sources