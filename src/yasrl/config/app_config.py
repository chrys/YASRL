import logging
import os
from dataclasses import dataclass, field

from .manager import ConfigurationManager
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """
    Handles application configuration and logging setup.
    """

    llm: str
    embed_model: str
    config_manager: ConfigurationManager = field(default_factory=ConfigurationManager)

    def __post_init__(self):
        self.config = self.config_manager.load_config()
        self._setup_logging()
        self._validate_env_vars()
        logger.info("Application configuration loaded and logging is set up.")

    def _setup_logging(self):
        """Sets up logging for the application."""
        log_level = getattr(self.config, "log_level", "INFO").upper()
        log_output = getattr(self.config, "log_output", "console")  # Default to console
        log_file = getattr(self.config, "log_file", "yasrl.log")

        # Remove all handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if log_output == "file":
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                filename=log_file,
                filemode="a",
            )
        else:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        logger.info(f"Logging configured: level={log_level}, output={log_output}")

    def _validate_env_vars(self):
        """Validates that the required environment variables are set."""
        missing_vars = []
        if self.llm == "openai" and not os.getenv("OPENAI_API_KEY"):
            missing_vars.append("OPENAI_API_KEY")
        if self.embed_model == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            missing_vars.append("GOOGLE_API_KEY")
        if not self.config.database.postgres_uri:
            missing_vars.append("POSTGRES_URI")

        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)