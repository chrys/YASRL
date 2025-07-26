"""
Configuration manager for YASRL library.
Supports hierarchical configuration with environment variables, config files, and defaults.
"""
import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from yasrl.exceptions import ConfigurationError
from .models import AdvancedConfig, LLMModelConfig, EmbeddingModelConfig, DatabaseConfig


class ConfigurationManager:
    """
    Manages configuration from multiple sources with priority:
    1. Environment variables (highest priority)
    2. Local config file (yasrl.yaml/yasrl.json)
    3. Global config file (~/.yasrl/config.yaml)
    4. Default values (lowest priority)
    
    Example:
        >>> # Initialize with default search paths
        >>> config_manager = ConfigurationManager()
        >>> config = config_manager.load_config()
        
        >>> # Initialize with specific config file
        >>> config_manager = ConfigurationManager(config_file="my_config.yaml")
        >>> config = config_manager.load_config()
        
        >>> # Initialize with custom environment prefix
        >>> config_manager = ConfigurationManager(env_prefix="MYAPP")
        >>> config = config_manager.load_config()
    """
    
    def __init__(
        self, 
        config_file: Optional[Union[str, Path]] = None,
        env_prefix: str = "YASRL"
    ) -> None:
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file. If None, searches for default locations.
            env_prefix: Prefix for environment variables (e.g., YASRL_LLM_MODEL)
        
        Raises:
            ConfigurationError: If specified config file doesn't exist
        """
        self.env_prefix = env_prefix
        self.config_file = self._find_config_file(config_file)
        self._config_cache: Optional[AdvancedConfig] = None
        
    def _find_config_file(self, config_file: Optional[Union[str, Path]]) -> Optional[Path]:
        """
        Find configuration file in order of preference.
        
        Args:
            config_file: User-specified config file path
            
        Returns:
            Path to config file, or None if no config file found
            
        Raises:
            ConfigurationError: If specified config file doesn't exist
        """
        if config_file:
            path = Path(config_file)
            if not path.exists():
                raise ConfigurationError(f"Specified config file not found: {config_file}")
            # Validate extension
            if path.suffix not in ['.yaml', '.yml', '.json']:
                raise ConfigurationError(f"Unsupported config file format: {path.suffix}")
            return path
        
        # Search order: local -> global -> none
        search_paths = [
            Path("yasrl.yaml"),
            Path("yasrl.yml"),
            Path("yasrl.json"),
            Path.home() / ".yasrl" / "config.yaml",
            Path.home() / ".yasrl" / "config.yml",
            Path.home() / ".yasrl" / "config.json"
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def load_config(self) -> AdvancedConfig:
        """
        Load configuration from all sources with proper precedence.
        
        Returns:
            Complete configuration object
            
        Raises:
            ConfigurationError: If configuration is invalid or cannot be loaded
        """
        if self._config_cache:
            return self._config_cache
        
        # Start with defaults
        config_dict = self._get_default_config()
        
        # Override with file config
        if self.config_file:
            try:
                file_config = self._load_config_file(self.config_file)
                config_dict = self._merge_configs(config_dict, file_config)
            except Exception as e:
                raise ConfigurationError(f"Failed to load config file {self.config_file}: {e}")
        
        # Override with environment variables
        env_config = self._load_env_config()
        config_dict = self._merge_configs(config_dict, env_config)
        
        # Validate and create config object
        try:
            self._config_cache = self._create_config_object(config_dict)
            return self._config_cache
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Dictionary with default configuration
        """
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
            "structured_logging": False
        }
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Dictionary with configuration from file
            
        Raises:
            ConfigurationError: If file format is unsupported or cannot be parsed
        """
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_file.suffix == '.json':
                    return json.load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported config file format: {config_file.suffix}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read config file: {e}")
    
    def _load_env_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
            
        Returns:
            Dictionary with configuration from environment variables
        """
        env_config = {}
        prefix = f"{self.env_prefix}_"
            
        # Map environment variables to config structure
        env_mappings = {
            # LLM configuration
            f"{prefix}LLM_PROVIDER": ["llm", "provider"],
            f"{prefix}LLM_MODEL": ["llm", "model_name"],
            f"{prefix}LLM_TEMPERATURE": ["llm", "temperature"],
            f"{prefix}LLM_MAX_TOKENS": ["llm", "max_tokens"],
            f"{prefix}LLM_TIMEOUT": ["llm", "timeout"],
            f"{prefix}LLM_API_VERSION": ["llm", "api_version"],
            
            # Embedding configuration
            f"{prefix}EMBEDDING_PROVIDER": ["embedding", "provider"],
            f"{prefix}EMBEDDING_MODEL": ["embedding", "model_name"],
            f"{prefix}CHUNK_SIZE": ["embedding", "chunk_size"],
            f"{prefix}BATCH_SIZE": ["embedding", "batch_size"],
            f"{prefix}EMBEDDING_TIMEOUT": ["embedding", "timeout"],
            
            # Database configuration
            f"{prefix}POSTGRES_URI": ["database", "postgres_uri"],
            f"{prefix}TABLE_PREFIX": ["database", "table_prefix"],
            f"{prefix}CONNECTION_POOL_SIZE": ["database", "connection_pool_size"],
            f"{prefix}VECTOR_DIMENSIONS": ["database", "vector_dimensions"],
            f"{prefix}INDEX_TYPE": ["database", "index_type"],
            
            # General configuration
            f"{prefix}RETRIEVAL_TOP_K": ["retrieval_top_k"],
            f"{prefix}RERANK_TOP_K": ["rerank_top_k"],
            f"{prefix}CHUNK_OVERLAP": ["chunk_overlap"],
            f"{prefix}BATCH_PROCESSING_SIZE": ["batch_processing_size"],
            f"{prefix}CACHE_ENABLED": ["cache_enabled"],
            f"{prefix}ASYNC_PROCESSING": ["async_processing"],
            f"{prefix}LOG_LEVEL": ["log_level"],
            f"{prefix}STRUCTURED_LOGGING": ["structured_logging"],
            
            # API Keys - these should be read directly without prefix
            "GOOGLE_API_KEY": ["google_api_key"],
            "OPENAI_API_KEY": ["openai_api_key"],
            "OLLAMA_HOST": ["ollama_host"],
            "POSTGRES_URI": ["database", "postgres_uri"],  # Also allow direct POSTGRES_URI
        }
            
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(env_config, config_path, self._convert_env_value(value))
        
        return env_config

    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any) -> None:
        """
        Set a nested value in configuration dictionary.

        Args:
                config: Configuration dictionary to modify
                path: List of keys representing path to value
                value: Value to set
            """
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value with appropriate type
        """
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Number conversion
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _create_config_object(self, config_dict: Dict[str, Any]) -> AdvancedConfig:
        """
        Create AdvancedConfig object from dictionary with validation.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Validated AdvancedConfig object
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
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
                # Add API keys to config object
                google_api_key=config_dict.get("google_api_key"),
                openai_api_key=config_dict.get("openai_api_key"),
                ollama_host=config_dict.get("ollama_host")
            )
            
            # Validate the complete configuration
            config.validate()
            return config
            
        except KeyError as e:
            raise ConfigurationError(f"Missing required configuration key: {e}")
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration type: {e}")
    
    def save_config(self, config: AdvancedConfig, config_file: Optional[Path] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
            config_file: Path to save configuration (optional)
            
        Raises:
            ConfigurationError: If unable to save configuration
        """
        if not config_file:
            config_file = self.config_file or Path("yasrl.yaml")
        
        config_dict = {
            "llm": {
                "provider": config.llm.provider,
                "model_name": config.llm.model_name,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "api_version": config.llm.api_version,
                "custom_params": config.llm.custom_params
            },
            "embedding": {
                "provider": config.embedding.provider,
                "model_name": config.embedding.model_name,
                "chunk_size": config.embedding.chunk_size,
                "batch_size": config.embedding.batch_size,
                "timeout": config.embedding.timeout,
                "custom_params": config.embedding.custom_params
            },
            "database": {
                "table_prefix": config.database.table_prefix,
                "connection_pool_size": config.database.connection_pool_size,
                "vector_dimensions": config.database.vector_dimensions,
                "index_type": config.database.index_type
                # Note: postgres_uri is not saved to file for security
            },
            "retrieval_top_k": config.retrieval_top_k,
            "rerank_top_k": config.rerank_top_k,
            "chunk_overlap": config.chunk_overlap,
            "batch_processing_size": config.batch_processing_size,
            "cache_enabled": config.cache_enabled,
            "async_processing": config.async_processing,
            "log_level": config.log_level,
            "structured_logging": config.structured_logging
        }
        
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if config_file.suffix in ['.yaml', '.yml']:
                with open(config_file, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_file.suffix == '.json':
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported config file format: {config_file.suffix}")
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save config file: {e}")
    
    def clear_cache(self) -> None:
        """Clear cached configuration to force reload on next access."""
        self._config_cache = None
    
    def get_config_sources(self) -> List[str]:
        """
        Get list of configuration sources in order of precedence.
        
        Returns:
            List of configuration source descriptions
        """
        sources = ["Environment variables"]
        
        if self.config_file:
            sources.append(f"Config file: {self.config_file}")
        
        sources.append("Default values")
        return sources